import os
import logging
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PATHS AND FILENAMES
HEALTHY_DIR = "/home1/s5787084/V3/labeled dataset_NREL/raw/healthy_processed"
DAMAGED_DIR = "/home1/s5787084/V3/labeled dataset_NREL/raw/damaged_processed"
OUTPUT_DIR  = "/home1/s5787084/V3/labeled dataset_NREL/splits"

TRAIN_HEALTHY_PATH = os.path.join(OUTPUT_DIR, "train_healthy.parquet")
VAL_DATA_PATH       = os.path.join(OUTPUT_DIR, "validation_dataset.parquet")
TEST_DATA_PATH      = os.path.join(OUTPUT_DIR, "test_dataset.parquet")

# Train on H5..H10 (healthy)
TRAIN_HEALTHY_FILES = [f"H{i}.parquet" for i in range(5, 11)]
# Validation: H1, H2 (healthy) + D6, D7 (faulty)
VAL_HEALTHY_FILES   = ["H1.parquet", "H2.parquet"]
VAL_FAULTY_FILES    = ["D6.parquet", "D7.parquet"]
# Test: H3, H4 (healthy) + D4, D5 (faulty)
TEST_HEALTHY_FILES  = ["H3.parquet", "H4.parquet"]
TEST_FAULTY_FILES   = ["D4.parquet", "D5.parquet"]

# FILTER AND WINDOW SETTINGS
NREL_SAMPLING_RATE = 40000
BASE_FREQ_HZ = 30.0
FILTER_ORDER = 4
BANDPASS_LOW = 10
BANDPASS_HIGH = 2000

WINDOW_SIZE_SECONDS = 1
OVERLAP = 0.75

# UNDERSAMPLING PARAMETERS
# These ratios generate (not fix) a certain class imbalance.
# For example, healthy_ratio=0.5 => 50% healthy, 50% faulty
VAL_HEALTHY_RATIO = 0.5
TEST_HEALTHY_RATIO = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# UNDERSAMPLING FUNCTION
def undersample_faulty_for_ratio(df_healthy, df_faulty, healthy_ratio=0.5):
    # Creating a desired 'healthy_ratio' by undersampling the faulty data.
    
    if df_healthy.empty or df_faulty.empty:
        return pd.concat([df_healthy, df_faulty], ignore_index=True)
    
    n_healthy = len(df_healthy)
    n_faulty = len(df_faulty)
    
    # We want: ratio = #healthy / (#healthy + #faulty).
    # Solve for #faulty => (#healthy / total) = ratio => total = #healthy / ratio
    # Then #faulty = total - #healthy
    desired_total = int(n_healthy / healthy_ratio)
    desired_faulty = desired_total - n_healthy
    
    # If we actually have more faulty data than desired, undersample
    if n_faulty > desired_faulty:
        df_faulty = df_faulty.sample(n=desired_faulty, random_state=42, replace=False)
    
    return pd.concat([df_healthy, df_faulty], ignore_index=True)

# HELPER FUNCTIONS FOR FEATURE EXTRACTION
def butterworth_bandpass_filter(data, low_cutoff, high_cutoff, sampling_rate, order=4):
    if len(data) < 2:
        return data
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def downsample_signal(data, original_sampling_rate, upper_cutoff, nyquist_factor=2.56):
    # Downsample to 2.56×Nyquist of the upper cutoff frequency.
    
    nyquist_freq = upper_cutoff
    target_sampling_rate = nyquist_freq / nyquist_factor
    downsample_factor = int(original_sampling_rate / target_sampling_rate)
    if downsample_factor < 1:
        return data, original_sampling_rate
    downsampled_data = data[::downsample_factor]
    new_rate = original_sampling_rate / downsample_factor
    return downsampled_data, new_rate

def compute_time_domain_features(data, duration, sampling_rate):
    if len(data) == 0 or duration <= 0 or sampling_rate <= 0:
        return {
            "rms": None, "skewness": None, "kurtosis": None, "peak": None,
            "crest_factor": None, "impulse_factor": None, "shape_factor": None
        }
    mean_data = np.mean(data)
    rms_val = np.sqrt(np.sum(data**2) / (duration * sampling_rate))
    std_data = np.std(data, ddof=0)
    skew_val = skew(data) if std_data > 0 else None
    kurt_val = kurtosis(data) if std_data > 0 else None
    peak_val = np.max(data) - np.min(data)
    crest_factor_val = peak_val / rms_val if rms_val > 0 else None
    impulse_factor_val = peak_val / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else None
    shape_factor_val = rms_val / mean_data if mean_data != 0 else None
    return {
        "rms": rms_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "peak": peak_val,
        "crest_factor": crest_factor_val,
        "impulse_factor": impulse_factor_val,
        "shape_factor": shape_factor_val
    }

def extract_features_from_series(data, sensor_name, sampling_rate, base_freq, window_size=1.0, overlap=0.75):
    # Bandpass filter
    filtered = butterworth_bandpass_filter(data, BANDPASS_LOW, BANDPASS_HIGH, sampling_rate, FILTER_ORDER)
    # Downsample
    downsampled, new_rate = downsample_signal(filtered, sampling_rate, BANDPASS_HIGH, nyquist_factor=2.56)
    window_samples = int(window_size * new_rate)
    step_size = int(window_samples * (1.0 - overlap))
    rows = []
    idx = 0
    
    # Sliding window
    while idx + window_samples <= len(downsampled):
        window = downsampled[idx : idx + window_samples]
        windowed_data = window * np.hanning(len(window))
        t_feats = compute_time_domain_features(windowed_data, window_size, new_rate)
        h_feats = compute_harmonic_features(windowed_data, new_rate, base_freq, num_harmonics=5)
        row_dict = {}
        row_dict.update(t_feats)
        row_dict.update(h_feats)
        row_dict["sensor"] = sensor_name
        rows.append(row_dict)
        idx += step_size
    return pd.DataFrame(rows)

def process_healthy_file(file_path):
    df = pd.read_parquet(file_path)
    if df.shape[1] < 1:
        return pd.DataFrame()
    colname = df.columns[0]
    data_array = df[colname].values.astype(float)
    feats = extract_features_from_series(
        data_array, "AN6", NREL_SAMPLING_RATE, BASE_FREQ_HZ,
        window_size=WINDOW_SIZE_SECONDS, overlap=OVERLAP
    )
    feats["label"] = "healthy"
    feats["failure_modes"] = 0
    return feats

def process_faulty_file(file_path, sensor_name, fail_mode_code):
    df = pd.read_parquet(file_path)
    if df.shape[1] < 1:
        return pd.DataFrame()
    colname = df.columns[0]
    data_array = df[colname].values.astype(float)
    feats = extract_features_from_series(
        data_array, sensor_name, NREL_SAMPLING_RATE, BASE_FREQ_HZ,
        window_size=WINDOW_SIZE_SECONDS, overlap=OVERLAP
    )
    feats["label"] = "faulty"
    feats["failure_modes"] = fail_mode_code
    return feats

def main():
    # Step 1: Identify file paths for train, validation, and test sets
    train_healthy_paths = [os.path.join(HEALTHY_DIR, f) for f in TRAIN_HEALTHY_FILES]
    val_healthy_paths   = [os.path.join(HEALTHY_DIR, f) for f in VAL_HEALTHY_FILES]
    val_faulty_paths    = [os.path.join(DAMAGED_DIR, f) for f in VAL_FAULTY_FILES]
    test_healthy_paths  = [os.path.join(HEALTHY_DIR, f) for f in TEST_HEALTHY_FILES]
    test_faulty_paths   = [os.path.join(DAMAGED_DIR, f) for f in TEST_FAULTY_FILES]

    # Step 2: Process TRAIN data (healthy only)
    df_train_list = []
    for path in train_healthy_paths:
        if os.path.isfile(path):
            df_h = process_healthy_file(path)
            if not df_h.empty:
                df_train_list.append(df_h)
    df_train = pd.concat(df_train_list, ignore_index=True) if df_train_list else pd.DataFrame()

    # Step 3: Process VALIDATION data (healthy + faulty)
    df_val_healthy_list = []
    for path in val_healthy_paths:
        if os.path.isfile(path):
            df_h = process_healthy_file(path)
            if not df_h.empty:
                df_val_healthy_list.append(df_h)
    df_val_healthy = pd.concat(df_val_healthy_list, ignore_index=True) if df_val_healthy_list else pd.DataFrame()

    df_val_faulty_list = []
    for path in val_faulty_paths:
        if os.path.isfile(path):
            if "D6" in path:
                df_f = process_faulty_file(path, "AN6", fail_mode_code=1)
            else:
                df_f = process_faulty_file(path, "AN7", fail_mode_code=2)
            if not df_f.empty:
                df_val_faulty_list.append(df_f)
    df_val_faulty = pd.concat(df_val_faulty_list, ignore_index=True) if df_val_faulty_list else pd.DataFrame()

    # Apply undersampling to create a chosen healthy ratio in validation
    df_val = undersample_faulty_for_ratio(df_val_healthy, df_val_faulty, healthy_ratio=VAL_HEALTHY_RATIO)

    # Step 4: Process TEST data (healthy + faulty)
    df_test_healthy_list = []
    for path in test_healthy_paths:
        if os.path.isfile(path):
            df_h = process_healthy_file(path)
            if not df_h.empty:
                df_test_healthy_list.append(df_h)
    df_test_healthy = pd.concat(df_test_healthy_list, ignore_index=True) if df_test_healthy_list else pd.DataFrame()

    df_test_faulty_list = []
    for path in test_faulty_paths:
        if os.path.isfile(path):
            if "D4" in path:
                df_f = process_faulty_file(path, "AN6", fail_mode_code=1)
            else:
                df_f = process_faulty_file(path, "AN7", fail_mode_code=2)
            if not df_f.empty:
                df_test_faulty_list.append(df_f)
    df_test_faulty = pd.concat(df_test_faulty_list, ignore_index=True) if df_test_faulty_list else pd.DataFrame()

    # Apply undersampling to create a chosen healthy ratio in testing
    df_test = undersample_faulty_for_ratio(df_test_healthy, df_test_faulty, healthy_ratio=TEST_HEALTHY_RATIO)

    # Step 5: Shuffle each dataset to avoid ordering bias
    df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_val   = df_val.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_test  = df_test.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Step 6: Save outputs
    df_train.to_parquet(TRAIN_HEALTHY_PATH, index=False)
    df_val.to_parquet(VAL_DATA_PATH, index=False)
    df_test.to_parquet(TEST_DATA_PATH, index=False)

    # Step 7: Log final counts
    logging.info(f"Train healthy rows: {len(df_train)}")
    logging.info(f"Validation rows (healthy+faulty): {len(df_val)}")
    logging.info(f"Test rows (healthy+faulty): {len(df_test)}")

if __name__ == "__main__":
    main()
