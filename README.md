# Anomaly Detection in Wind Turbines Using Variational Autoencoders and Isolation Forest

**Authors:** Guillermo Gil de Avalle Bellido, Christos Emmanouilidis

---

This repository contains the implementation of a study that addresses operational challenges in wind turbine fault detection by comparing static thresholds (ISO 10816-21) with a hybrid unsupervised machine learning model using Variational Autoencoder (VAE) and Isolation Forest (IF).

---

## Overview

Wind turbines operate in highly dynamic environments, making static fault detection thresholds prone to inaccuracies. This study addresses these limitations by introducing a fully unsupervised VAE-IF hybrid model capable of adapting to variability in operational conditions and data characteristics, such as imbalance and noise. 

The study involves two main evaluation scenarios:

#### 1. **Controlled Environment**
- NREL Wind Turbine Gearbox Vibration Condition Monitoring Benchmarking Dataset used.
- Dataset Splits:
  - **Training**: Healthy-only samples (around 10,000 observations).  
  - **Validation/Testing**: Balanced datasets (around 6,500 healthy and faulty samples each).  
- Objective: Evaluate model performance under ideal conditions and conduct sensitivity analyses for class imbalance and referene proportions.

#### 2. **Operational Environment**
- Two N131/3.9MW turbines in South Netherlands
- Datasets:
  - **Turbine 1**: A Nordex N131 turbine with a reported gearbox fault (oil leakage and bearing scratches).  
  - **Turbine 2**: Another Nordex N131 turbine confirmed healthy over the same period.  
- Objective: Test model adaptability to real-world complexities, comparing detection results with ISO-based thresholds.

---

### Methodology

1. **Signal Processing and Feature Extraction**:
   - Vibration data is preprocessed following ISO guidelines, including band-pass filtering, downsampling, and windowing.
   - Time-domain statistical features (e.g., RMS, kurtosis, skewness) are extracted as model inputs.

2. **VAE-IF Hybrid Model**:
   - The **VAE** learns a probabilistic latent space from healthy data, compressing inputs into structured representations.
   - The **Isolation Forest** uses the latent representations to isolate anomalies based on sparse data points.

3. **Comparison with ISO Thresholds**:
   - The ISO 10816-21 standard provides static RMS-based vibration thresholds to detect faults.  
   - Both methods are evaluated on their ability to detect faults under controlled and operational conditions.

<img width="906" alt="Captura de pantalla 2025-01-22 a las 15 57 35" src="https://github.com/user-attachments/assets/17e42269-8cf4-490d-ba08-6ba344ad2f8a" />

---

### VAE-IF Hybrid Model

The VAE-IF hybrid model combines a Variational Autoencoder (VAE) and Isolation Forest (IF) for fault detection. The VAE learns a compact latent representation of healthy vibration data, organizing it into a structured probabilistic space that naturally separates anomalies. The IF then identifies outliers in this latent space by evaluating how easily a data point can be isolated. This approach adapts to noise, imbalance, and dynamic conditions, offering a flexible and robust alternative to static fault detection thresholds.

<img width="1029" alt="Captura de pantalla 2025-01-22 a las 15 59 20" src="https://github.com/user-attachments/assets/5f373c14-c8b5-4768-94ab-91433e9d8731" />

---
### Key Contributions

- **Validation of ISO 10816-21**: Examines its effectiveness in controlled and operational wind farm datasets.
- **Hybrid Model Introduction**: Demonstrates the potential of combining VAE and IF for adaptive fault detection.
- **Real-World Applicability**: Provides insights into transitioning from static to flexible thresholds in industrial setups.

---

### Results Summary

- **Controlled Environment**: 
  - ISO thresholds performed well under ideal conditions.
  - The VAE-IF model showed robust performance, especially under imbalanced scenarios.

- **Operational Environment**:
  - VAE-IF detected significantly more faults than ISO thresholds, showing better adaptability to noise and real-world variability.

---

## Dataset Description

### Controlled Environment Datasets

The controlled-environment data is sourced from NREL (Sheng, 2012b). This benchmarking CM dataset features vibration samples collected from two gearboxes installed on a 750 kW three-bladed wind turbine—one healthy and one faulty. The damaged gearbox underwent controlled testing in a specialized facility after field damage, with several sensors mounted on the casings of affected components to detect faults such as scuffing, dents, or corrosion. Subsequently, the healthy gearbox underwent a similar test for comparability. Vibration samples from both gearboxes were collected using piezoelectric accelerometers sampling at 40 kHz.

**Files:**
1. **`train_healthy.parquet`**
   - Contains approximately **10,000 samples**.
   - All samples represent **healthy conditions**.
   - Used for training models in a clean and controlled environment.

2. **`validation_dataset.parquet`**
   - Contains approximately **6,500 samples**.
   - A balanced dataset, with an equal number of samples labeled as **healthy** and **faulty**.
   - Used for hyperparameter tuning and early stopping during training.

3. **`test_dataset.parquet`**
   - Contains approximately **6,500 samples**.
   - A balanced dataset, with an equal number of samples labeled as **healthy** and **faulty**.
   - Used for evaluating model performance in a controlled environment.

### Operational Environment Datasets

The operational dataset features two Nordex N131 turbines located in the south of the Netherlands. These turbines, rated at 3.9 MW, are equipped with sensors that collect short data snapshots at intervals ranging from 6 to 12 hours, which are merged to form a continuous time series. The sensors, which are piezoelectric accelerometers, are distributed throughout the drivetrain and sample data at frequencies between 12.8 kHz and 25.6 kHz, depending on their location.

**Files:**
1. **`Turbine1.parquet`**
   - Data corresponds to a **Nordex N131 turbine**, collected between **25th May 2023** and **31st August 2023**.
   - On **27th August 2023**, this turbine reported a **failure in the gearbox**, specifically:
     - **Scratches in the outer bearing.**
     - **Oil leakage.**
   - This dataset captures a mix of healthy operations and fault progression leading to the gearbox failure.

2. **`Turbine2.parquet`**
   - Data corresponds to another **Nordex N131 turbine**, collected during the **same time period** as Turbine1.
   - This turbine remained **completely healthy** throughout the observation period.
   - Used as a representative dataset for healthy conditions in real-world scenarios and for training models in operational settings.

### Dataset Summary

| Environment       | Dataset File               | Description                                                                                     |
|-------------------|---------------------------|-------------------------------------------------------------------------------------------------|
| **Controlled**    | `train_healthy.parquet`    | 10,000 healthy samples for training.                                                           |
|                   | `validation_dataset.parquet` | 6,500 samples (balanced: healthy and faulty) for validation.                                   |
|                   | `test_dataset.parquet`     | 6,500 samples (balanced: healthy and faulty) for testing.                                      |
| **Operational**   | `Turbine1.parquet`         | Nordex N131 turbine with a gearbox failure reported on 27th August 2023.                       |
|                   | `Turbine2.parquet`         | Nordex N131 turbine (same model, same period) that remained completely healthy.                |

---

## Project Structure

```
.
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore rules (protects data/)
├── data/                          # Dataset files (protected by .gitignore)
│   ├── train_healthy.parquet
│   ├── validation_dataset.parquet
│   ├── test_dataset.parquet
│   ├── Turbine1.parquet
│   └── Turbine2.parquet
└── src/                           # Source code
    ├── controlled_environment/    # Scripts for controlled environment analysis
    │   ├── bayesian_optimization.py      # Hyperparameter optimization
    │   ├── iso_evaluation.py             # ISO 10816-21 threshold evaluation
    │   ├── processing_and_extraction.py  # Data preprocessing and feature extraction
    │   └── vae-if_evaluation.py          # VAE-IF model evaluation
    └── operational_environment/   # Scripts for operational environment analysis
        ├── full_evaluation.py            # Complete evaluation pipeline
        ├── processing_and_extraction.py  # Data preprocessing and feature extraction
        └── unzip_and_organize.py         # Data organization utility
```

---

## Cite This Work

If you use this work in your research, please cite:

```bibtex
@InProceedings{GilDeAvalleBellido2026,
  author    = {Gil de Avalle Bellido, Guillermo and Emmanouilidis, Christos},
  title     = {Anomaly Detection in Wind Turbines Using Variational Autoencoders and Isolation Forest},
  booktitle = {Advances in Production Management Systems. Cyber-Physical-Human Production Systems: Human-AI Collaboration and Beyond},
  year      = {2026},
  series    = {IFIP Advances in Information and Communication Technology},
  volume    = {765},
  publisher = {Springer},
  address   = {Cham},
  doi       = {10.1007/978-3-032-03534-9_35},
  url       = {https://doi.org/10.1007/978-3-032-03534-9_35}
}
```

---
