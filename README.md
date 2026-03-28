# 🏥 Diabetic Patient Readmission — EDA Project

> A complete 8-stage Exploratory Data Analysis pipeline on real-world clinical data from 130 US hospitals, built to understand and prepare diabetic patient data for readmission prediction.

---

## 📌 Overview

This project performs a structured EDA on the **UCI Diabetes 130-US Hospitals dataset (1999–2008)**.  
The goal is to clean, explore, and engineer features from ~101,000 patient records to identify key factors influencing hospital readmission among diabetic patients.

| Property | Details |
|----------|---------|
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) |
| **Records** | 101,766 patient encounters |
| **Features** | 50 clinical & demographic variables |
| **Target** | `readmitted` — No / <30 days / >30 days |
| **Type** | Exploratory Data Analysis + Feature Engineering |

---

## 🗂️ Repository Structure

```
SULProject/
├── SULProject_Final.ipynb     # Complete 8-stage EDA notebook
├── diabetic_data.csv          # Raw dataset
├── EDA_Complete_dataset.csv   # Cleaned & feature-engineered dataset
└── README.md
```

---

## 🔬 8-Stage EDA Pipeline

### Stage 1 — Data Collection & Loading
- Imported all libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `sklearn`)
- Loaded raw CSV and confirmed shape: **101,766 rows × 50 columns**

### Stage 2 — Data Understanding
- Inspected dtypes, non-null counts, and descriptive statistics
- Identified `?` as the placeholder for missing values — replaced with `NaN`

### Stage 3 — Missing Value Analysis & Imputation
- Visualised missingness via heatmap and bar chart (before & after)
- **Dropped** columns with >50% missing: `weight` (97%), `max_glu_serum` (95%), `A1Cresult` (83%)
- **Mode-filled**: `race`, `payer_code`, `medical_specialty`
- **Unknown-filled**: `diag_1`, `diag_2`, `diag_3` (preserves clinical ambiguity)
- ✅ Result: **Zero missing values**

### Stage 4 — Univariate Analysis
- Distribution plots for all numeric features
- Countplots + pie chart for the target variable
- Detected **class imbalance** and **skewed numeric distributions**

### Stage 5 — Bivariate Analysis
- Boxplots: `time_in_hospital` and `num_medications` vs `readmitted`
- Scatter: lab procedures vs medications, coloured by readmission class

### Stage 6 — Multivariate Analysis
- Correlation heatmap (lower triangle masked) across all numeric features
- Pairplot on 4 key features with readmission as hue (sampled 3,000 rows)

### Stage 7 — Outlier Detection & Treatment
- **IQR fencing** (1.5×IQR): outlier count and % per column
- **Z-score cross-check** (|z| > 3): row-level flagging
- **Winsorization**: clipped at 1st–99th percentile (no rows dropped)
- Before & after boxplots + distribution overlays

### Stage 8 — Feature Engineering & Insights
- **Log1p transform** on highly skewed columns (|skew| > 1)
- 4 new features engineered:

| Feature | Description |
|---------|-------------|
| `total_service_utilization` | Sum of all service count columns — patient burden score |
| `med_changed` | Alias for `change` (0/1) — clearer semantic name |
| `diagnosis_count` | Count of non-null diagnoses (0–3) — complexity indicator |
| `stay_category` | Binned `time_in_hospital` → Short / Medium / Long / Extended |

- Variance Threshold (0.01) applied to remove near-zero variance features

---

## 💡 Key Findings

- The dataset is **class-imbalanced** — SMOTE or class weights recommended during modelling
- `time_in_hospital`, `num_medications`, and `num_lab_procedures` are the strongest numeric predictors of readmission
- `total_service_utilization` (engineered) shows a clear increasing trend across readmission classes
- Several medication columns had near-zero variance and were removed before modelling

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/lucifer230407/SULProject.git
cd SULProject

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter

# Launch the notebook
jupyter notebook SULProject_Final.ipynb
```

> **Note:** Update the dataset path in Cell 2 to match your local file location before running.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Citation:** Strack et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates.* BioMed Research International.

---

## 👤 Author

**Himanshu** — [@lucifer230407](https://github.com/lucifer230407)  
B.E. Computer Science (AI/ML) · Chitkara University
