import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

from datetime import datetime
from matplotlib.dates import DateFormatter
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# If using GPU:
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# For CUDA determinism (can slow things down a bit):
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1) Paths
TRAIN_PATH = "/Users/guillermogildeavallebellido/Desktop/preprocessed/Turbine2.parquet"
TEST_PATH  = "/Users/guillermogildeavallebellido/Desktop/preprocessed/Turbine1.parquet"

# 2) Data Loading & Preprocessing
def parse_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def clean_dataframe(df):
    # Define the columns we want to keep
    allowed_cols = ["utc_datetime", "rms", "kurtosis", "skewness",
                    "peak", "crest_factor", "impact_factor", "shape_factor"]

    # Parse datetime if available
    if "utc_datetime" in df.columns:
        df["utc_datetime"] = parse_datetime(df["utc_datetime"])

    # Retain only the allowed columns that exist in the dataframe
    existing_allowed = [col for col in allowed_cols if col in df.columns]
    df = df[existing_allowed].copy()

    # Drop rows where all values (except utc_datetime) are NaN
    df.dropna(how="all", subset=[col for col in existing_allowed if col != "utc_datetime"], inplace=True)

    # Sort by datetime if available
    if "utc_datetime" in df.columns:
        df.sort_values(by="utc_datetime", inplace=True, ignore_index=True)

    return df

df_train = pd.read_parquet(TRAIN_PATH)
df_test  = pd.read_parquet(TEST_PATH)

df_train = clean_dataframe(df_train)
df_test  = clean_dataframe(df_test)

print("Train columns:", df_train.columns)
print("Test columns:", df_test.columns)

feature_cols = [c for c in df_train.columns if c != "utc_datetime"]

# 3) Define VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=8, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

def vae_loss_function(reconstructed, x, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss()(reconstructed, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld)
    return recon_loss + beta * kld

def train_vae(model, data_tensor, epochs=20, lr=0.001, batch_size=512, beta=1.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, in dataloader:
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch_x)
            loss = vae_loss_function(reconstructed, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    return model

# 4) Train VAE + IF, then apply to test
def run_vae_if(df_train, df_test, feature_cols, latent_dim=4):

    X_train_raw = df_train[feature_cols].dropna().values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)

    input_dim = X_train.shape[1]
    vae = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    vae = train_vae(vae, X_train_torch, epochs=90, lr=0.0003711474 , batch_size=128, beta=0.24998934)
    vae.eval()

    with torch.no_grad():
        _, mu_train, _ = vae(X_train_torch)
    mu_train_np = mu_train.numpy()

    iso = IsolationForest(n_estimators=250, max_samples=256, random_state=42)
    iso.fit(mu_train_np)

    df_test_proc = df_test.copy()
    X_test_raw = df_test_proc[feature_cols].dropna().values
    X_test = scaler.transform(X_test_raw)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        _, mu_test, _ = vae(X_test_torch)
    mu_test_np = mu_test.numpy()

    test_scores = iso.decision_function(mu_test_np)  # Higher => more "normal"
    anomaly_scores = -test_scores
    train_scores = iso.decision_function(mu_train_np)
    train_anomaly_scores = -train_scores
    dynamic_threshold = np.percentile(train_anomaly_scores, 99)  # top 1%

    anomaly_labels = (anomaly_scores > dynamic_threshold).astype(int)

    valid_idx = df_test_proc[feature_cols].dropna().index
    df_test_proc["vae_if_anomaly_score"] = np.nan
    df_test_proc["vae_if_anomaly_label"] = np.nan

    df_test_proc.loc[valid_idx, "vae_if_anomaly_score"] = anomaly_scores
    df_test_proc.loc[valid_idx, "vae_if_anomaly_label"] = anomaly_labels

    return df_test_proc, dynamic_threshold

df_test_vae, dyn_threshold = run_vae_if(df_train, df_test, feature_cols, latent_dim=4)
print(f"Dynamic threshold (top 1% anomaly) = {dyn_threshold:.4f}")

# 5) t-SNE on Test (VAE approach)
def plot_tsne_test_latent(df_test_vae, feature_cols):
    test_valid = df_test_vae.dropna(subset=feature_cols).copy()
    if len(test_valid) == 0:
        print("No valid data for TSNE. Skipping.")
        return

    test_valid["cluster"] = test_valid["vae_if_anomaly_label"].fillna(0).astype(int)
    sc = StandardScaler()
    X_test = sc.fit_transform(test_valid[feature_cols].values.astype(np.float32))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(X_test)

    test_valid["tsne_0"] = embedded[:, 0]
    test_valid["tsne_1"] = embedded[:, 1]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=test_valid, x="tsne_0", y="tsne_1",
        hue="cluster", palette="Set1", alpha=0.7, legend=False  # Remove legend
    )
    plt.title("Turbine 1: t-SNE based on VAE-IF Fault Classification")
    plt.show()

plot_tsne_test_latent(df_test_vae, feature_cols)

# 6) Time Series - VAE-IF
def plot_timeseries_anomalies(df_test_vae, method_label="vae_if", downsample_frac=0.05, threshold=None):
    dfp = df_test_vae.dropna(subset=["utc_datetime", f"{method_label}_anomaly_score", f"{method_label}_anomaly_label"]).copy()
    if len(dfp) == 0:
        print(f"No valid data for {method_label} timeseries plot.")
        return

    # Downsample if specified
    if downsample_frac is not None and 0 < downsample_frac < 1:
        dfp = dfp.sample(frac=downsample_frac).sort_index()

    dfp.set_index("utc_datetime", inplace=True)
    dfp.sort_index(inplace=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfp.index, dfp[f"{method_label}_anomaly_score"], label="Anomaly Score", color="blue", alpha=0.6)

    anomalies = dfp[dfp[f"{method_label}_anomaly_label"] == 1]
    ax.scatter(anomalies.index, anomalies[f"{method_label}_anomaly_score"], color="red", label="Fault (Anomaly)", alpha=0.8)

    # Plot the threshold if provided
    if threshold is not None:
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"Flexible hreshold = {threshold:.2f}")
        """
        ax.text(
            dfp.index[0],  # Annotate at the start of the plot
            threshold + 0.02 * (dfp[f"{method_label}_anomaly_score"].max() - dfp[f"{method_label}_anomaly_score"].min()),  # Offset slightly above the line
            f"Flexible Threshold = {threshold:.2f}",
            color="green", fontsize=10, ha="left"
        )
        """

    ax.set_xlabel("Date")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Turbine 1: VAE-IF Fault Scores in Time Series")
    ax.legend(loc="lower left")  # Legend on bottom-right
    ax.grid(True)

    # Format x-axis to only show date (no hours)
    date_fmt = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_fmt)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    total_faults = len(anomalies)
    total_samples = len(dfp)
    fault_pct = 100.0 * total_faults / total_samples if total_samples > 0 else 0.0

    print(f"Total {method_label} 'faults' found: {total_faults} / {total_samples} ({fault_pct:.2f}%)")
    if total_faults>0:
        print("First fault:", anomalies.index[0], "Last fault:", anomalies.index[-1])

plot_timeseries_anomalies(df_test_vae, method_label="vae_if", threshold=dyn_threshold)

# 7) Anomaly Distribution
def plot_anomaly_distribution(df_test_vae, method_label="vae_if", threshold=None):
    dfp = df_test_vae.dropna(subset=[f"{method_label}_anomaly_score"]).copy()
    if len(dfp)==0:
        print("No valid data for anomaly distribution.")
        return

    plt.figure(figsize=(6,4))
    sns.histplot(dfp[f"{method_label}_anomaly_score"], bins=50, kde=True, color="orange")
    plt.title("Turbine 1: Fault Score Distribution VAE-IF")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")

    # Use provided threshold if available, otherwise calculate
    if threshold is not None:
        plt.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5)
        # plt.legend([f"Threshold ~ {threshold:.4f}"])
    else:
        guess = np.percentile(dfp[f"{method_label}_anomaly_score"], 99)
        plt.axvline(x=guess, color="red", linestyle="--", linewidth=1.5)
        # plt.legend([f"Threshold ~ {guess:.4f}"])

    # Annotate the threshold line
    plt.text(
        threshold - 0.04,  # x position
        plt.gca().get_ylim()[1] * 0.9,  # y position, slightly below the top of the plot
        f"Flexible Threshold = {threshold:.2f}",  # Label text
        color="red", fontsize=10, ha="center"
    )

    plt.tight_layout()
    plt.show()

plot_anomaly_distribution(df_test_vae, method_label="vae_if", threshold=dyn_threshold)

# 8) Static Threshold Approach
bc_threshold = 0.765
cd_threshold = 1.223

df_test_thresholds = df_test_vae.copy()
if "rms" not in df_test_thresholds.columns:
    print("No 'rms' column found in test data. Threshold-based approach skipped.")
else:

    # Add predictions
    df_test_thresholds["bc_pred"] = (df_test_thresholds["rms"] > bc_threshold).astype(int)
    df_test_thresholds["cd_pred"] = (df_test_thresholds["rms"] > cd_threshold).astype(int)


    def plot_timeseries_thresholds_combined(df_in, bc_val, cd_val, downsample_frac=0.3):    # Clutter fixation
        """
        Single plot showing RMS + two threshold lines (B/C, C/D)
        and highlighted faults for each threshold, with priority-based labeling.
        """
        dfp = df_in.dropna(subset=["utc_datetime", "rms"]).copy()
        if len(dfp) == 0:
            print("No valid data for threshold time series.")
            return

        dfp.set_index("utc_datetime", inplace=True)
        dfp.sort_index(inplace=True)

        # Downsample if specified
        if downsample_frac is not None and 0 < downsample_frac < 1:
            dfp = dfp.sample(frac=downsample_frac).sort_index()

        # Assign priority-based fault labels (0 = No Fault, 1 = B/C Fault, 2 = C/D Fault)
        dfp["final_label"] = 0  # Default: no fault
        dfp.loc[dfp["bc_pred"] == 1, "final_label"] = 1  # Mark B/C faults
        dfp.loc[dfp["cd_pred"] == 1, "final_label"] = 2  # Override with C/D faults (higher risk)

        # Separate faults for plotting
        bc_faults = dfp[dfp["final_label"] == 1]
        cd_faults = dfp[dfp["final_label"] == 2]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4))

        # RMS line
        ax.plot(dfp.index, dfp["rms"], label="RMS", color="blue", alpha=0.6)

        # Horizontal lines for thresholds
        ax.axhline(y=bc_val, color="orange", linestyle="--", label=f"B/C = {bc_val}")
        ax.axhline(y=cd_val, color="red", linestyle="--", label=f"C/D = {cd_val}")

        # Fault points based on final label
        ax.scatter(bc_faults.index, bc_faults["rms"], color="orange", marker="o", label="Fault B/C", alpha=0.8)
        ax.scatter(cd_faults.index, cd_faults["rms"], color="red", marker="o", label="Fault C/D", alpha=0.8)

        # Axis labels, title, and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("RMS")
        ax.set_title("Turbine 1: ISO Threshold on Time Series (B/C and C/D)")
        ax.legend(loc="lower left")  # Legend on bottom-right
        ax.grid(True)

        # Format x-axis to only show date (no hours)
        date_fmt = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_fmt)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Print stats
        num_samples = len(dfp)
        bc_fault_count = len(bc_faults)
        bc_pct = 100.0 * bc_fault_count / num_samples if num_samples > 0 else 0.0

        cd_fault_count = len(cd_faults)
        cd_pct = 100.0 * cd_fault_count / num_samples if num_samples > 0 else 0.0

        print(f"--- Combined Threshold Stats ---")
        print(f"Total samples: {num_samples}")
        print(f"B/C Faults: {bc_fault_count} ({bc_pct:.2f}%)")
        if bc_fault_count > 0:
            print(f"B/C First Fault: {bc_faults.index[0]}, Last Fault: {bc_faults.index[-1]}")
        print(f"C/D Faults: {cd_fault_count} ({cd_pct:.2f}%)")
        if cd_fault_count > 0:
            print(f"C/D First Fault: {cd_faults.index[0]}, Last Fault: {cd_faults.index[-1]}")

        # Print stats
        num_samples = len(dfp)
        bc_fault_count = len(bc_faults)
        bc_pct = 100.0 * bc_fault_count / num_samples if num_samples > 0 else 0.0

        cd_fault_count = len(cd_faults)
        cd_pct = 100.0 * cd_fault_count / num_samples if num_samples > 0 else 0.0

        print(f"--- Combined Threshold Stats ---")
        print(f"Total samples: {num_samples}")
        print(f"B/C Faults: {bc_fault_count} ({bc_pct:.2f}%)")
        if bc_fault_count > 0:
            print(f"B/C First Fault: {bc_faults.index[0]}, Last Fault: {bc_faults.index[-1]}")
        print(f"C/D Faults: {cd_fault_count} ({cd_pct:.2f}%)")
        if cd_fault_count > 0:
            print(f"C/D First Fault: {cd_faults.index[0]}, Last Fault: {cd_faults.index[-1]}")

    # Plot the combined time series
    plot_timeseries_thresholds_combined(df_test_thresholds, bc_threshold, cd_threshold)

    def plot_rms_distribution(df_in):
        dfp = df_in.dropna(subset=["rms"]).copy()
        if len(dfp)==0:
            print("No valid data for RMS distribution.")
            return

        plt.figure(figsize=(8,5))
        sns.histplot(dfp["rms"], kde=True, bins=50, color="steelblue", alpha=0.6)
        plt.axvline(bc_threshold, color="orange", linestyle="--", label=f"B/C = {bc_threshold}")
        plt.axvline(cd_threshold, color="red", linestyle="--", label=f"C/D = {cd_threshold}")
        plt.xlabel("RMS")
        plt.ylabel("Count")
        plt.title("Turbine 1: RMS Histogram with B/C & C/D Thresholds")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    plot_rms_distribution(df_test_thresholds)

print("Done.")
