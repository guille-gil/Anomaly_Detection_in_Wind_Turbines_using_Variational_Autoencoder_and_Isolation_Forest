import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, matthews_corrcoef
)
from sklearn.manifold import TSNE

#  1. Load Data
TEST_PATH = "/Users/guillermogildeavallebellido/Desktop/NREL/splits/test_dataset.parquet"
df = pd.read_parquet(TEST_PATH)

# Map labels "healthy" -> 0, "faulty" -> 1
label_map = {"healthy": 0, "faulty": 1}
df["label_binary"] = df["label"].map(label_map)


# 2. Define Thresholds
bc_threshold = 0.765
cd_threshold = 1.223

# 3. Make Predictions
df["bc_pred"] = (df["rms"] > bc_threshold).astype(int)
df["cd_pred"] = (df["rms"] > cd_threshold).astype(int)

y_true = df["label_binary"]

# 4. Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_true, df["bc_pred"], "Confusion Matrix - ISO B/C Threshold ($\mathit{p}$ = 0.01)")
plot_confusion_matrix(y_true, df["cd_pred"], "Confusion Matrix - ISO C/D Threshold ($\mathit{p}$ = 0.01)")

# 6. Metrics for Each Threshold
def calculate_metrics(y_true, y_pred, threshold_name):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Metrics for {threshold_name}:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"MCC: {mcc:.3f}")
    print("-" * 30)

calculate_metrics(y_true, df["bc_pred"], "B/C Threshold")
calculate_metrics(y_true, df["cd_pred"], "C/D Threshold")

# 7. Distribution Plot with Both Thresholds
plt.figure(figsize=(8, 5))
sns.histplot(df["rms"], kde=True, bins=50, color="steelblue", alpha=0.6)
plt.axvline(bc_threshold, color="orange", linestyle="--", label=f"B/C = {bc_threshold}")
plt.axvline(cd_threshold, color="red", linestyle="--", label=f"C/D = {cd_threshold}")
plt.xlabel("RMS")
plt.ylabel("Count")
plt.title("RMS Histogram with ISO B/C and C/D Thresholds ($\mathit{p}$ = 0.01)")
plt.legend()
plt.tight_layout()
plt.show()

# 8. Add Index as Synthetic Time
df = df.reset_index(drop=True)  # Reset index
df["time_index"] = df.index  # Create a synthetic time index based on row order (needed for plotting)


# 9. Time Series Visualization of RMS
def plot_rms_timeseries(df, thresholds=None):
    """
    Time series plot of RMS with thresholds and fault points.
    """
    # Create fault labels based on thresholds
    df = df.copy()
    if thresholds:
        df["bc_fault"] = (df["rms"] > thresholds["B/C Threshold"]["value"]).astype(int)
        df["cd_fault"] = (df["rms"] > thresholds["C/D Threshold"]["value"]).astype(int)

    # Fault categories
    bc_faults = df[df["bc_fault"] == 1]
    cd_faults = df[df["cd_fault"] == 1]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df["time_index"], df["rms"], label="RMS", alpha=0.7, color="blue")

    if thresholds:
        for name, threshold_info in thresholds.items():
            plt.axhline(
                y=threshold_info["value"],
                color=threshold_info["color"],
                linestyle="--",
                label=f"{name} = {threshold_info['value']:.3f}"
            )

    # Add scatter points for faults
    plt.scatter(
        bc_faults["time_index"], bc_faults["rms"],
        color="orange", marker="o", label="Fault B/C", alpha=0.8
    )
    plt.scatter(
        cd_faults["time_index"], cd_faults["rms"],
        color="red", marker="o", label="Fault C/D", alpha=0.8
    )

    # Labels and legend
    plt.xlabel("Sample Index")
    plt.ylabel("RMS Value")
    plt.title("RMS Time series ISO B/C and C/D Thresholds ($\mathit{p}$ = 0.01)")
    plt.legend(loc="lower right")  # Position the legend in the bottom-right
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Define thresholds for the plot
thresholds = {
    "B/C Threshold": {"value": bc_threshold, "color": "orange"},
    "C/D Threshold": {"value": cd_threshold, "color": "red"}
}

# Plot RMS time series
plot_rms_timeseries(df, thresholds=thresholds)
