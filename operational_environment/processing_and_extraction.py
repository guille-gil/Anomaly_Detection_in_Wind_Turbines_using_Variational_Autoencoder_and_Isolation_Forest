"""
Script for Processing Wind Turbine Vibration Data
-------------------------------------------------
This script processes raw vibration data and other metrics (e.g., wind, status) from wind turbines.
It extracts features like RMS, peak, kurtosis, skewness, crest factor, and impulse factor
from adaptively sized windows, applies a band-pass filter (serving as anti-aliasing) before downsampling,
and consolidates the results into a single Parquet file for each turbine.
Designed for the Nordex N117/3600 wind turbines.

Naming Convention:
------------------
To ensure proper processing, the raw data directory must follow this folder naming convention:
    <TurbineName>_<Component>_<SensorCode>

Example:
    - BOMMELERWAARD-N-88377_MainBearing_AI1
    - BOMMELERWAARD-N-88378_GeneratorDriveEnd_AI5

Author: Guillermo Gil de Avalle Bellido
Company: Windunie
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import logging
import time

# Define paths
raw_data_dir = "/Users/guillermogildeavallebellido/Desktop/isweargodbethelast/raw"
processed_data_dir = "/Users/guillermogildeavallebellido/Desktop/isweargodbethelast/preprocessed"
os.makedirs(processed_data_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("processing_debug.log")
    ]
)

# Band-pass ranges for each component (Hz)
BANDPASS_RANGES = {
    "MainBearing": (10, 2000),
    "PlanetaryStage1": (10, 2000),
    "PlanetaryStage2": (10, 2000),
    "HighSpeedShaft": (10, 2000),
    "HighSpeedShaftAxial": (10, 2000),
    "GeneratorDriveEnd": (10, 5000),
    "GeneratorNonDriveEnd": (10, 5000),
}

ORIGINAL_SAMPLING_RATE = 15600  # Fallback if not found in metadata


def compute_features(data, duration, sampling_rate):
    """
    Compute vibration features for a window of samples. Time-domain only.

    :param data: array-like, vibration samples
    :param duration: float, window duration in seconds
    :param sampling_rate: float, sample rate in Hz
    :return: dict of time-domain features
    """
    if len(data) == 0 or duration <= 0 or sampling_rate <= 0:
        return {
            "rms": None,
            "skewness": None,
            "kurtosis": None,
            "shape_factor": None,
            "crest_factor": None,
            "peak": None,
            "impulse_factor": None,
        }

    N = len(data)
    mean_data = np.mean(data)

    rms = 50 * np.sqrt(np.mean(np.square(data)))

    std_dev = np.std(data, ddof=0)
    if std_dev > 0:
        skewness_val = np.sum((data - mean_data) ** 3) / (N * (std_dev ** 3))
        kurtosis_val = np.sum((data - mean_data) ** 4) / (N * (std_dev ** 4))
    else:
        skewness_val = None
        kurtosis_val = None

    shape_factor = rms / mean_data if mean_data != 0 else None
    peak = np.max(data) - np.min(data)
    crest_factor = peak / rms if rms > 0 else None
    avg_abs = np.mean(np.abs(data))
    impulse_factor = peak / avg_abs if avg_abs > 0 else None

    return {
        "rms": rms,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "shape_factor": shape_factor,
        "crest_factor": crest_factor,
        "peak": peak,
        "impulse_factor": impulse_factor,
    }


def extract_metadata_and_data(file_path):
    """
    Extract metadata, vibration data, status data, wind data, and sampling info from .txt file.
    Returns:
        metadata (dict),
        vibration_data (list),
        status_data (list),
        wind_data (list),
        original_sampling_rate (float or None)
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Regex to find all [header] sections
        section_pattern = re.compile(r'\[([^\]]+)\](.*?)((?=\n\[[^\]]+\])|$)', re.DOTALL)
        sections = section_pattern.findall(content)

        metadata = {}
        vibration_data = []
        wind_data = []
        status_data = []
        original_sampling_rate = None

        metadata_mapping = {}
        data_mapping = {}

        for section_name, section_content, _ in sections:
            section_name = section_name.strip()
            section_content = section_content.strip()

            # aduchannel: metadata about the channel (incl. iSampleRate)
            if "aduchannel" in section_name.lower():
                for line in section_content.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip().lower()
                        val = value.strip()
                        metadata[key] = val
                        if key == "isamplerate":
                            original_sampling_rate = float(val)

            # adudata: actual vibration samples
            elif "adudata" in section_name.lower():
                for line in section_content.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and re.match(r"^-?\d+(\.\d+)?$", line):
                        vibration_data.append(float(line))

            # prescanopc: store metadata keyed by identifier
            elif re.match(r'^prescanopc\d*:\d+$', section_name, re.IGNORECASE):
                identifier = section_name.split(":")[1].strip()
                section_meta = {}
                for line in section_content.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        section_meta[k.strip()] = v.strip()
                if section_meta:
                    metadata_mapping[identifier] = section_meta

            # prescanopcdata: store numeric data keyed by identifier
            elif re.match(r'^prescanopcdata\d*:\d+$', section_name, re.IGNORECASE):
                identifier = section_name.split(":")[1].strip()
                data_values = []
                for line in section_content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) > 1 and re.match(r"^-?\d+(\.\d+)?$", parts[1]):
                        data_values.append(float(parts[1]))
                if data_values:
                    data_mapping[identifier] = data_values

        # Associate prescanopc with prescanopcdata
        for identifier, meta_info in metadata_mapping.items():
            sz_label = meta_info.get("szLabel", "").lower()
            associated_data = data_mapping.get(identifier, [])
            if "wind" in sz_label:
                wind_data.extend(associated_data)
            elif "status" in sz_label or "particle_lifebit" in sz_label:
                status_data.extend(associated_data)

        # Grab start/end time from metadata
        start_time_key = "starttime"
        end_time_key = "endtime"

        if not metadata.get(start_time_key) or not metadata.get(end_time_key):
            logging.warning(f"Missing start or end time in {file_path}. Returning empty results.")
            return {}, [], [], [], None

        return metadata, vibration_data, status_data, wind_data, original_sampling_rate

    except Exception as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return {}, [], [], [], None


def butterworth_bandpass_filter(data, low_cutoff, high_cutoff, sampling_rate, order=4):
    """
    Band-pass (anti-aliasing) filter to avoid aliasing before downsampling.
    """
    if len(data) == 0 or sampling_rate <= 0:
        return np.array(data)

    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    # Safeguard for edge cases
    if high >= 1:
        high = 0.99
    if low <= 0:
        low = 0.01

    b, a = butter(order, [low, high], btype="band", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def process_file(
        file_path,
        component,
        sensor_code,
        turbine_name,
        window_size_seconds=1,
        overlap_ratio=0.75
):
    """
    Process a single .txt file and compute features, ensuring no data leakage from future data
    prior to filtering/downsampling.
    """
    metadata, vibration_data, status_data, wind_data, orig_rate = extract_metadata_and_data(file_path)
    if not vibration_data:
        logging.warning(f"No vibration data found in {file_path}. Skipping.")
        return []

    # Determine sample rate from metadata or fallback
    sampling_rate = orig_rate if orig_rate else ORIGINAL_SAMPLING_RATE

    # Convert the Unix 'starttime' to UTC
    try:
        unix_starttime = int(metadata.get("starttime", 0))
        utc_starttime = datetime.utcfromtimestamp(unix_starttime)
    except (ValueError, TypeError):
        logging.warning(f"Invalid or missing Unix starttime in {file_path}. Skipping.")
        return []

    # Convert window size (seconds) -> samples
    window_samples = int(window_size_seconds * sampling_rate)
    step_size = int(window_samples * (1 - overlap_ratio))

    # Iterate over raw data in non-overlapping or partially overlapping chunks
    results = []
    for start_idx in range(0, len(vibration_data) - window_samples + 1, step_size):
        raw_window = vibration_data[start_idx: start_idx + window_samples]

        # 1) Band-pass filter (serves as anti-aliasing)
        filtered_window = butterworth_bandpass_filter(
            raw_window,
            *BANDPASS_RANGES.get(component, (10, 2000)),
            sampling_rate
        )

        # 2) Downsample
        high_cut = BANDPASS_RANGES[component][1]
        downsample_factor = max(1, int(sampling_rate / (2.56 * high_cut)))
        downsampled_window = filtered_window[::downsample_factor]
        new_sampling_rate = sampling_rate / downsample_factor

        # 3) Apply Hanning window
        hann_window = np.hanning(len(downsampled_window))
        final_window = downsampled_window * hann_window

        # 4) Compute time-domain features
        time_feats = compute_features(
            final_window,
            duration=window_size_seconds,
            sampling_rate=new_sampling_rate
        )

        # Calculate the approximate UTC time for this window's start
        offset_secs = start_idx / sampling_rate
        window_time_utc = utc_starttime + timedelta(seconds=offset_secs)
        time_feats["utc_datetime"] = window_time_utc.isoformat()

        # Combine feats + metadata
        feats_dict = {**time_feats}
        feats_dict.update({
            "turbine": turbine_name,
            "component": component,
            "sensor_code": sensor_code,
            "original_sampling_rate": sampling_rate,
            "new_sampling_rate": new_sampling_rate,
            "bandpass_range": f"{BANDPASS_RANGES[component][0]}-{BANDPASS_RANGES[component][1]} Hz",
            "start_index": start_idx,
        })
        results.append(feats_dict)

    return results


def process_turbine(
        turbine_name,
        window_size_seconds=5,
        overlap_ratio=0.5,
        selected_sensors=None
):
    """
    Process all .txt files for a given turbine, combining results into a single Parquet file.

    :param turbine_name: e.g. 'BOMMELERWAARD-N-88377'
    :param window_size_seconds: length of each analysis window in seconds
    :param overlap_ratio: fraction of overlap between consecutive windows (0 to <1)
    :param selected_sensors: list of sensor codes (e.g. ["AI3"]) or None to process all
    """
    logging.info(f"Starting processing for turbine: {turbine_name}")
    turbine_root = os.path.join(raw_data_dir, turbine_name)

    if not os.path.isdir(turbine_root):
        logging.warning(f"Turbine folder {turbine_name} does not exist. Skipping.")
        return

    turbine_folders = [
        folder for folder in os.listdir(turbine_root)
        if os.path.isdir(os.path.join(turbine_root, folder))
    ]

    all_features = []

    for folder in turbine_folders:
        folder_parts = folder.split("_")
        if len(folder_parts) < 3:
            logging.warning(f"Skipping folder {folder}. Invalid structure.")
            continue

        component = folder_parts[-2]
        sensor_code = folder_parts[-1]

        # If user specified sensors, skip those not in the list
        if selected_sensors and sensor_code not in selected_sensors:
            logging.info(f"Skipping sensor: {sensor_code} for turbine: {turbine_name}")
            continue

        logging.info(f"Processing component={component}, sensor={sensor_code}, turbine={turbine_name}")
        folder_path = os.path.join(turbine_root, folder)

        files_to_process = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".txt") and not f.lower().startswith("readme")
        ]

        if not files_to_process:
            logging.warning(f"No valid files found in folder: {folder}. Skipping.")
            continue

        for file_path in files_to_process:
            try:
                feats = process_file(
                    file_path,
                    component,
                    sensor_code,
                    turbine_name,
                    window_size_seconds=window_size_seconds,
                    overlap_ratio=overlap_ratio
                )
                all_features.extend(feats)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

    # Save to Parquet
    if all_features:
        df = pd.DataFrame(all_features)
        # Sort the DataFrame by utc_datetime (if the column exists)
        if 'utc_datetime' in df.columns:
            df.sort_values(by='utc_datetime', inplace=True)

        output_path = os.path.join(processed_data_dir, f"{turbine_name}.parquet")
        df.to_parquet(output_path, index=False, compression="snappy")
        logging.info(f"Saved consolidated features for turbine: {turbine_name} to {output_path}")
    else:
        logging.warning(f"No valid data found for turbine: {turbine_name}")


def main():
    start = time.time()

    # e.g. ["AI3"] to only extract sensor AI3, or None to process all sensors
    selected_sensors = ["AI1"]

    # Collect unique turbine folders without splitting names
    turbine_names = [
        folder
        for folder in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, folder))
    ]

    logging.info(f"Found turbine folders: {turbine_names}")
    logging.info(f"Starting processing for {len(turbine_names)} turbines...")

    # Window settings
    window_size_seconds = 1
    overlap_ratio = 0.75

    # Multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(
            process_turbine,
            [
                (turbine, window_size_seconds, overlap_ratio, selected_sensors)
                for turbine in turbine_names
            ]
        )

    elapsed = time.time() - start
    logging.info(f"Processing completed in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
