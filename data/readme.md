# Dataset Description for Controlled and Operational Environments

This document outlines the datasets used in the study, categorized into **controlled environments** and **operational environments**.

---

## Controlled Environment Datasets

The controlled environment datasets are synthetic or laboratory-collected data used for initial model development, validation, and testing in a well-defined and noise-free setting. These datasets consist of labeled samples with balanced distributions for robust evaluation.

### Files:
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

---

## Operational Environment Datasets

The operational environment datasets are collected from real-world turbines, introducing realistic complexities such as environmental noise, varying operating conditions, and temporal dependencies. These datasets reflect real-world turbine behavior over an extended period.

### Files:
1. **`Turbine1.parquet`**
   - Data corresponds to a **Nordex N131 turbine** operating between **25th May 2027** and **31st August 2023**.
   - On **27th August 2023**, this turbine reported a **failure in the gearbox**, specifically:
     - **Scratches in the outer bearing.**
     - **Oil leakage.**
   - This dataset captures a mix of healthy operations and fault progression leading to the gearbox failure.

2. **`Turbine2.parquet`**
   - Data corresponds to another **Nordex N131 turbine** operating during the **same time period** as Turbine1.
   - This turbine remained **completely healthy** throughout the observation period.
   - Used as a representative dataset for healthy conditions in real-world scenarios and for training models in operational settings.

---

### Summary Table

| Environment       | Dataset File               | Description                                                                                     |
|-------------------|---------------------------|-------------------------------------------------------------------------------------------------|
| **Controlled**    | `train_healthy.parquet`    | 10,000 healthy samples for training.                                                           |
|                   | `validation_dataset.parquet` | 6,500 samples (balanced: healthy and faulty) for validation.                                   |
|                   | `test_dataset.parquet`     | 6,500 samples (balanced: healthy and faulty) for testing.                                      |
| **Operational**   | `Turbine1.parquet`         | Nordex N131 turbine with a gearbox failure reported on 27th August 2023.                       |
|                   | `Turbine2.parquet`         | Nordex N131 turbine (same model, same period) that remained completely healthy.                |

---

