import os
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest

# Fix random seeds
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1) Load and pre-split data
VAL_PATH   = "/home1/s5787084/V3/labeled dataset_NREL/splits/validation_dataset.parquet"
TEST_PATH  = "/home1/s5787084/V3/labeled dataset_NREL/splits/test_dataset.parquet"

def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    # Map "healthy"->0, "faulty"->1 if needed
    if "label" in df.columns and df["label"].dtype == object:
        df["label"] = df["label"].map({"healthy": 0, "faulty": 1})
    # Drop non-feature columns if they exist
    drop_cols = ["label", "sensor", "failure_modes", "start_time", "end_time"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").values.astype(np.float32)
    y = df["label"].values.astype(int) if "label" in df.columns else np.zeros(len(df), dtype=int)
    return X, y

X_train, y_train = load_data(TRAIN_PATH) 
X_val,   y_val   = load_data(VAL_PATH)    
X_test,  y_test  = load_data(TEST_PATH)  

print(f"Training set:   X={X_train.shape}, y={y_train.shape}")
print(f"Validation set: X={X_val.shape},   y={y_val.shape}")
print(f"Scenario set:   X={X_test.shape},  y={y_test.shape}")

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
        total_loss = 0.0
        for batch_x, in dataloader:
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch_x)
            loss = vae_loss_function(reconstructed, batch_x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

from sklearn.metrics import f1_score, classification_report, confusion_matrix

# 3) VAE-IF Process
def evaluate_pipeline(
    X_train, y_train,
    X_val, y_val,
    input_dim,
    latent_dim=4,
    epochs=20,
    lr=0.001,
    batch_size=512,
    beta=1.0,
    n_estimators=100,
    max_samples='auto'
):
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

    # Train VAE
    model = VariationalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    train_vae(model, X_train_tensor, epochs=epochs, lr=lr, batch_size=batch_size, beta=beta)

    # Encode validation data
    model.eval()
    with torch.no_grad():
        _, mu_train, _ = model(X_train_tensor)
        _, mu_val, _ = model(X_val_tensor)

    # Convert latent space to numpy
    mu_train_np = mu_train.numpy()
    mu_val_np = mu_val.numpy()

    # Train Isolation Forest on latent space of training data
    iso = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=42)
    iso.fit(mu_train_np)

    # Predict validation data
    val_preds = iso.predict(mu_val_np)  # +1 (normal), -1 (anomaly)
    val_preds_bin = np.where(val_preds == -1, 1, 0)  # Convert to binary: 1=faulty, 0=healthy

    # Calculate F1-macro
    f1_macro_val = f1_score(y_val, val_preds_bin, average="macro")

    return f1_macro_val

# 4) Optuna Config
def objective(trial):
    latent_dim = trial.suggest_categorical("latent_dim", [2, 4, 8, 16])
    epochs = trial.suggest_int("epochs", 10, 100, step=10)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    beta = trial.suggest_float("beta", 0.1, 10.0, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=20)
    max_samples = trial.suggest_categorical("max_samples", ["auto", 256, 512, 1024])

    f1_macro_val = evaluate_pipeline(
        X_train, y_train,
        X_val, y_val,
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        beta=beta,
        n_estimators=n_estimators,
        max_samples=max_samples
    )

    return f1_macro_val  

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_trial = study.best_trial
  
print("===== BEST TRIAL =====")
print(f"Val F1-macro: {best_trial.value:.4f}")
print("Hyperparams:")
for k, v in best_trial.params.items():
    print(f"  {k}: {v}")

# 5) Retrain on training split and evaluate on testing for final validation of results
def retrain_and_evaluate_on_scenario(best_params):
    # Combine train+val
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    # Scale and tensor conversion
    scaler = StandardScaler()
    X_tv_scaled = scaler.fit_transform(X_tv)
    X_test_scaled = scaler.transform(X_test)

    X_tv_tensor = torch.tensor(X_tv_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Extract best hyperparameters
    latent_dim = best_params["latent_dim"]
    epochs = best_params["epochs"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    beta = best_params["beta"]
    n_estimators = best_params["n_estimators"]
    max_samples = best_params["max_samples"]

    # Retrain VAE on (train+val)
    model = VariationalAutoencoder(input_dim=X_tv.shape[1], latent_dim=latent_dim)
    train_vae(model, X_tv_tensor, epochs=epochs, lr=lr, batch_size=batch_size, beta=beta)
    model.eval()

    # Encode latent space
    with torch.no_grad():
        _, mu_tv, _ = model(X_tv_tensor)
        _, mu_test, _ = model(X_test_tensor)

    mu_tv_np = mu_tv.numpy()
    mu_test_np = mu_test.numpy()

    # Train Isolation Forest on (train+val)
    iso = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=42)
    iso.fit(mu_tv_np)

    # Test predictions
    test_preds = iso.predict(mu_test_np)  # +1 (normal), -1 (anomaly)
    test_preds_bin = np.where(test_preds == -1, 1, 0)  # Convert to binary: 1=faulty, 0=healthy

    # Calculate F1-macro
    f1_macro_test = f1_score(y_test, test_preds_bin, average="macro")

    # Classification report and confusion matrix
    cls_report = classification_report(y_test, test_preds_bin, target_names=["Healthy(0)", "Faulty(1)"])
    cm = confusion_matrix(y_test, test_preds_bin)

    return f1_macro_test, cls_report, cm

print("\RETRAIN on (train+val) & TEST on final_scenario")
f1_macro_test, cls_report, cm = retrain_and_evaluate_on_scenario(best_trial.params)

print(f"Test F1-macro: {f1_macro_test:.4f}\n")
print("Classification Report (Scenario):")
print(cls_report)
print("Confusion Matrix (Scenario):")
print(cm)
