import os
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import matthews_corrcoef


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,  precision_recall_fscore_support,
)
from sklearn.ensemble import IsolationForest

# 1) Load Pre-Split Data
TRAIN_PATH = "/Users/guillermogildeavallebellido/Desktop/NREL/splits/train_healthy.parquet"
TEST_PATH  = "/Users/guillermogildeavallebellido/Desktop/NREL/splits/test_dataset.parquet"

# Random seeds
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Read train
df_train = pd.read_parquet(TRAIN_PATH)
if "label" in df_train.columns and df_train["label"].dtype == object:
    df_train["label"] = df_train["label"].map({"healthy": 0, "faulty": 1})

drop_cols = ["label", "sensor", "failure_modes", "start_time", "end_time"]
X_train_df = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors="ignore")
y_train_df = df_train["label"] if "label" in df_train.columns else pd.Series(np.zeros(len(df_train)), index=df_train.index)

X_train = X_train_df.values.astype(np.float32)
y_train = y_train_df.values.astype(int)

# Read test
df_test = pd.read_parquet(TEST_PATH)
if "label" in df_test.columns and df_test["label"].dtype == object:
    df_test["label"] = df_test["label"].map({"healthy": 0, "faulty": 1})

X_test_df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors="ignore")
y_test_df = df_test["label"] if "label" in df_test.columns else pd.Series(np.zeros(len(df_test)), index=df_test.index)

X_test = X_test_df.values.astype(np.float32)
y_test = y_test_df.values.astype(int)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 2) Define VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=12, latent_dim=4):
        super(VariationalAutoencoder, self).__init__()
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
    return model

# 3) Run One Experiment (for the charts)
def run_one_experiment(
    X_train, X_test,
    y_train, y_test,
    model_params,
    random_seed=42
):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    latent_dim = model_params.get("latent_dim", 4)
    epochs = model_params.get("epochs", 20)
    lr = model_params.get("lr", 1e-3)
    batch_size = model_params.get("batch_size", 512)
    beta = model_params.get("beta", 1.0)
    n_estimators = model_params.get("n_estimators", 100)
    max_samples = model_params.get("max_samples", "auto")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_torch  = torch.tensor(X_test_scaled,  dtype=torch.float32)

    # Train VAE
    vae_model = VariationalAutoencoder(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim
    )
    train_vae(vae_model, X_train_torch, epochs=epochs, lr=lr, batch_size=batch_size, beta=beta)
    vae_model.eval()

    # Latent embeddings
    with torch.no_grad():
        _, mu_train, _ = vae_model(X_train_torch)
        _, mu_test,  _ = vae_model(X_test_torch)

    mu_train_np = mu_train.numpy()
    mu_test_np  = mu_test.numpy()

    # Isolation Forest
    iso = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_seed
    )
    iso.fit(mu_train_np)

    # Dynamic thresholding
    train_scores = iso.decision_function(mu_train_np)  # Scores for training data
    dynamic_threshold = np.percentile(train_scores, 1)  # Compares scores with the distribution of the healthy data

    # Test predictions based on dynamic threshold
    test_scores = iso.decision_function(mu_test_np)  # Scores for test data
    test_preds_dynamic = np.where(test_scores < dynamic_threshold, 1, 0)  # 1 => faulty, 0 => healthy

    # Metrics
    fpr, tpr, _ = roc_curve(y_test, -test_scores)  # Inverted for anomaly
    prec, rec, _ = precision_recall_curve(y_test, -test_scores)

    mcc = matthews_corrcoef(y_test, test_preds_dynamic)

    test_cm = confusion_matrix(y_test, test_preds_dynamic)
    test_prec, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, test_preds_dynamic, average="macro"
    )
    isolation_score = np.mean(test_scores)

    metrics_dict = {
        "precision_macro": test_prec,
        "recall_macro": test_recall,
        "f1_macro": test_f1,
        "confusion_matrix": test_cm,
        "isolation_score": isolation_score,
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec,
        "rec_curve": rec,
        "anomaly_probs": -test_scores,
        "mu_train": mu_train_np,
        "mu_test": mu_test_np,
        "dynamic_threshold": dynamic_threshold,
        "mcc": mcc
    }
    return metrics_dict

# 4) Run Multiple Experiments
#    Returns the same final metrics: mean ± std, plus confusion matrix sum
def run_multiple_experiments(
    X_train, X_test,
    y_train, y_test,
    model_params,
    n_runs=1
):
    all_metrics = []
    seeds = [42 + i*10 for i in range(n_runs)]  # e.g., 42, 52, 62, ...
    for i in range(n_runs):
        seed = seeds[i]
        result = run_one_experiment(
            X_train, X_test, y_train, y_test,
            model_params=model_params,
            random_seed=seed
        )
        all_metrics.append(result)

    # Collect numeric metrics
    metrics_collector = {}
    skip_keys = {"confusion_matrix", "fpr", "tpr", "prec_curve", "rec_curve", "anomaly_probs"}
    for m in all_metrics[0].keys():
        if m not in skip_keys:
            metrics_collector[m] = [res[m] for res in all_metrics]

    # Compute mean and std
    final_results = {}
    for metric_name, values in metrics_collector.items():
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # sample std
        final_results[metric_name] = {
            "mean": mean_val,
            "std": std_val
        }

    # Sum confusion matrices
    cm_sum = np.sum([m["confusion_matrix"] for m in all_metrics], axis=0)
    final_results["confusion_matrix_sum"] = cm_sum

    # Collect MCC values
    metrics_collector["mcc"] = [res["mcc"] for res in all_metrics]

    # Compute mean and std
    for metric_name, values in metrics_collector.items():
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # sample std
        final_results[metric_name] = {
            "mean": mean_val,
            "std": std_val
        }

    return final_results


# (C) Plot Latent Space (t-SNE on Test Latent Representations)
def plot_latent_space(mu_train, mu_test, y_train, y_test, perplexity=30):
    """
    Visualizes the latent space using t-SNE (2D projection) in a Seaborn style, with custom legend for Healthy/Faulty.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Run t-SNE on combined latent space
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    all_latents = np.concatenate((mu_train, mu_test), axis=0)
    tsne_result = tsne.fit_transform(all_latents)

    # Split results for train and test
    tsne_train = tsne_result[:len(mu_train)]
    tsne_test  = tsne_result[len(mu_train):]

    # Create DataFrame for test latents
    df_test_latent = pd.DataFrame({
        "tsne_0": tsne_test[:, 0],
        "tsne_1": tsne_test[:, 1],
        "label":  y_test
    })

    # Map numeric labels to descriptive labels
    df_test_latent["label_name"] = df_test_latent["label"].map({0: "Healthy", 1: "Faulty"})

    plt.figure(figsize=(8, 6))

    # Create scatter plot and capture the Axes object
    ax = sns.scatterplot(
        data=df_test_latent,
        x="tsne_0",
        y="tsne_1",
        hue="label_name",         # Use the descriptive label column
        palette={"Healthy": "royalblue", "Faulty": "orangered"},
        alpha=0.7,
        legend='full'
    )

    # Change the legend title to "Label"
    ax.legend(title='Label')

    plt.title("t-SNE at p = 0.5; without re-tuning")
    plt.tight_layout()
    plt.show()


# 5) Main
#    (A) First run => produce plots
#    (B) Then run N times => produce mean ± std
if __name__ == "__main__":
    # Example parameters
    model_params = {
        "latent_dim": 4,
        "epochs": 90,
        "lr": 0.0003711474,
        "batch_size": 128,
        "beta": 0.24998934,
        "n_estimators": 250,
        "max_samples": 256
    }
    n_runs = 50

    # (A) First run for plotting
    single_run_result = run_one_experiment(
        X_train, X_test,
        y_train, y_test,
        model_params=model_params,
        random_seed=42
    )

    # Confusion Matrix
    cm = single_run_result["confusion_matrix"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Healthy", "Faulty"], yticklabels=["Healthy", "Faulty"] )
    plt.title("p = 0.5 ; with re-tuning")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Plot anomaly probability histogram with a threshold line
    plt.figure(figsize=(6, 4))
    sns.histplot(single_run_result["anomaly_probs"], bins=50, kde=True, color="orange")
    plt.title("p = 0.01 ; with re-tuning")
    plt.xlabel("Anomaly Distribution")
    plt.ylabel("Count")

    # Add the threshold line
    threshold_value = - single_run_result["dynamic_threshold"]
    plt.axvline(x=threshold_value, color="red", linestyle="--", linewidth=1.5) # label=f"Threshold = {threshold_value:.2f}"


    # Annotate the threshold line
    plt.text(
        threshold_value - 0.04,  # x position
        plt.gca().get_ylim()[1] * 0.9,  # y position, slightly below the top of the plot
        f"Threshold = {threshold_value:.2f}",  # Label text
        color="red", fontsize=10, ha="center"
    )


    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualize the latent space
    plot_latent_space(
        single_run_result["mu_train"],
        single_run_result["mu_test"],
        y_train,
        y_test
    )

    # (B) Multi-run for final average
    results = run_multiple_experiments(
        X_train, X_test,
        y_train, y_test,
        model_params=model_params,
        n_runs=n_runs
    )


    print(f"\n----- Final Results over {n_runs} runs -----")
    for metric_name, metric_vals in results.items():
        if metric_name == "confusion_matrix_sum":
            continue
        mean_val = metric_vals["mean"]
        std_val  = metric_vals["std"]
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")

    print("\nSum of Confusion Matrices across runs:")
    print(results["confusion_matrix_sum"])
