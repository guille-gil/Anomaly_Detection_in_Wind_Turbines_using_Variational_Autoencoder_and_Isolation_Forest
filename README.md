## Navigating Operational Challenges: Combining Variational Autoencoder and Isolation Forest for Adaptive Fault Detection in Wind Turbines

This repository contains the implementation of a study that addresses operational challenges in wind turbine fault detection by comparing static thresholds (ISO 10816-21) with a hybrid unsupervised machine learning model using Variational Autoencoder (VAE) and Isolation Forest (IF).

---

### Overview

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
