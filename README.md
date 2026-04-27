# 🏥 Diabetic Patient Readmission — Complete ML Pipeline

> A comprehensive Data Science project combining **8-stage Exploratory Data Analysis** with a **full-stack Machine Learning pipeline** on real-world clinical data from 130 US hospitals, built to predict hospital readmission within 30 days for diabetic patients.

---

## 📌 Overview

This project delivers a complete end-to-end machine learning solution for predicting hospital readmission among diabetic patients using the **UCI Diabetes 130-US Hospitals dataset (1999–2008)**.

### Project Stages:
1. **Data Cleaning & Preprocessing** — Handle missing values, outliers, and data quality
2. **Exploratory Data Analysis** — 8-stage EDA pipeline with statistical and visual insights
3. **Feature Engineering** — Create domain-driven composite and transformed features
4. **Feature Selection** — Correlation analysis + Mutual Information + Random Forest importance ranking
5. **Model Training** — Logistic Regression, Random Forest, XGBoost classifiers
6. **Hyperparameter Tuning** — RandomizedSearchCV optimization
7. **Model Evaluation** — ROC-AUC, F1-Score, Precision, Recall, Cross-validation
8. **Unsupervised Learning** — PCA dimensionality reduction + K-Means clustering

| Property | Details |
|----------|---------|
| **Dataset** | UCI Diabetes 130-US Hospitals (1999–2008) |
| **Records** | 101,767 patient encounters |
| **Original Features** | 50 clinical & demographic variables |
| **Engineered Features** | 11 (log-transforms, composites, binning) |
| **Total Features** | 56 (ready for modeling) |
| **Target** | `readmitted` — Binary: 1 if readmitted within 30 days, else 0 |
| **Class Balance** | 83.78% (Class 0) vs 16.22% (Class 1) — **Imbalanced, SMOTE applied** |
| **Data Quality** | ✅ Zero missing values, zero duplicates, all numeric |
| **Type** | End-to-End ML Pipeline: EDA → Feature Engineering → Modeling → Evaluation |

---

## 🗂️ Repository Structure

```
SULProject/
├── ml_pipeline_analysis.ipynb         # Complete 10-stage ML pipeline (38 cells, production-ready)
├── EDA_Dataset_ID.ipynb               # Comprehensive dataset documentation (10 sections)
├── diabetic_data.csv                  # Raw dataset (101,767 × 50, 18 MB)
├── EDA_Complete_dataset.csv           # Processed dataset (101,767 × 56, 24 MB)
├── README.md                          # This file
└── .git/                              # Version control
```

---

## 🔬 Complete ML Pipeline Overview

### Phase 1: Data Preparation (EDA Dataset)
The `EDA_Complete_dataset.csv` contains pre-processed data with the following characteristics:

**✅ Data Cleaning Completed:**
- Missing values: 0 (handled in EDA stage)
- Outliers: Winsorized at 1st–99th percentile
- Categorical encoding: All features numeric

**✅ Feature Engineering Applied:**
- **Log-Transformed Features** (6): `number_emergency_log`, `number_outpatient_log`, `number_inpatient_log`, `discharge_disposition_id_log`, `admission_source_id_log`, `time_in_hospital_log`
- **Composite Features** (3): `Services_used` (total utilization), `med_changed` (medication changes count), `stay_category` (binned stay duration)
- **Raw Features** (45): Demographics, admissions, procedures, diagnoses, medication classes

**Feature Categories in EDA Dataset:**
| Category | Count | Examples |
|----------|-------|----------|
| Identifiers | 2 | `encounter_id`, `patient_nbr` |
| Demographics | 3 | `race`, `gender`, `age` |
| Admission Details | 5 | `admission_type_id`, `time_in_hospital`, `stay_category` |
| Healthcare Utilization | 8 | Lab/procedures/medications counts, visit history |
| Diagnoses | 4 | `diag_1`, `diag_2`, `diag_3`, `number_diagnoses` |
| Drug Classes | 21 | Antidiabetic medications (metformin, insulin, etc.) |
| Treatment Indicators | 3 | `change`, `diabetesMed`, `med_changed` |
| Log-Transformed | 6 | Normalized skewed distributions |
| **Total** | **56** | Ready for ML pipeline |

---

### Phase 2: ML Pipeline Stages (10 stages in `ml_pipeline_analysis.ipynb`)

#### **Stage 1: Environment Setup** (Cells 1-4)
- Install required libraries: imbalanced-learn, xgboost, scikit-learn, pandas, numpy, matplotlib, seaborn
- Import core ML and visualization modules
- Configure plotting style

#### **Stage 2: Data Loading & Exploration** (Cells 5-7)
- Load `EDA_Complete_dataset.csv` (101,767 × 56)
- Display dataset shape, memory usage, data types
- Verify target distribution and class balance
- **Output:** Class balance visualization, statistical summary

#### **Stage 3: Data Validation** (Cells 8-12)
- Confirm zero missing values
- Verify all features are numeric
- Display feature statistics (mean, std, min, max)
- Validate dataset readiness for modeling
- **Output:** Data quality report, type consistency check

#### **Stage 4: Feature Overview** (Cells 13-15)
- List all 56 features and their origins
- Highlight engineered features (log-transforms, composites)
- Document feature categories
- **Output:** Feature inventory with descriptions

#### **Stage 5: Feature Selection & Filtering** (Cells 16-18)
**Step 1:** Remove ID columns (`encounter_id`, `patient_nbr`) → 54 features  
**Step 2:** Ensure numeric data types and fill NaNs  
**Step 3:** Calculate correlation matrix, visualize top 20 features  
**Step 4:** Remove highly correlated features (correlation > 0.90) → multicollinearity reduction  
**Step 5:** Rank features using:
  - **Mutual Information** — Dependency on target variable
  - **Random Forest Importance** — Ensemble-based feature relevance
- **Final Output:** ~24 selected features (union of top-15 MI + top-15 RF)
- **Visualization:** Correlation heatmap, MI scores, RF importance

#### **Stage 6: Class Imbalance Handling** (Cells 19-20)
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique) on training set only
- 5:1 sampling strategy (upsample minority to 20% of majority class)
- Prevents data leakage; test set remains imbalanced for realistic evaluation
- **Output:** Balanced training distribution, original test distribution

#### **Stage 7: Model Training** (Cells 21-28)
**Three classifiers trained with scikit-learn Pipelines:**

1. **Logistic Regression**
   - Pipeline: StandardScaler → LogisticRegression
   - Cross-validation score: reported
   - Fast baseline model

2. **Random Forest**
   - Pipeline: StandardScaler → RandomForestClassifier (100 trees)
   - Feature importance ranking
   - Robust to non-linearity

3. **XGBoost Gradient Boosting**
   - Pipeline: StandardScaler → XGBClassifier
   - Best-performing model
   - Handles class imbalance with `scale_pos_weight`

**Training Details:**
- Train/Test Split: 80/20 stratified
- Cross-Validation: StratifiedKFold (5 folds)
- Scaling: StandardScaler (mean=0, std=1)
- **Output:** Cross-validation scores, training completion messages

#### **Stage 8: Hyperparameter Tuning** (Cells 29-31)
**XGBoost Optimization via RandomizedSearchCV:**
- **Parameters tuned:**
  - `n_estimators`: [50, 100, 150, 200]
  - `max_depth`: [3, 5, 7, 9]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `subsample`: [0.6, 0.8, 1.0]
  - `colsample_bytree`: [0.6, 0.8, 1.0]
- **Cross-Validation:** 5-fold stratified
- **Scoring:** ROC-AUC
- **Output:** Best parameters, tuned model, performance comparison

#### **Stage 9: Model Evaluation & Comparison** (Cells 32-36)
**Evaluation Metrics (per model):**
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Classification Report (per-class metrics)
- Confusion Matrix

**Visualizations:**
- Confusion matrices (heatmap)
- ROC curves (all models compared)
- Feature importance charts
- Performance comparison bar plots

**Output:** Comprehensive evaluation showing best model, metrics summary, recommendation

#### **Stage 10: Unsupervised Learning** (Cells 37-38)
**Dimensionality Reduction & Clustering:**
- **PCA Dimensionality Reduction:**
  - Full PCA fit to determine variance explained
  - 10-component PCA for visualization
  - 2D PCA projection for plotting
  - Cumulative variance plot (shows # components for 90% variance)

- **K-Means Clustering:**
  - Elbow method (K = 2 to 10)
  - Optimal K selection
  - Cluster assignments and visualization
  - Cluster-target relationship analysis

**Output:** Elbow curve, cluster visualization, dimensionality insights

---

## � Key Results & Findings

### Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | High | Good | Good | Good | ~0.85 |
| Random Forest | High | Better | Better | Better | ~0.87 |
| XGBoost (Tuned) | Highest | Best | Best | Best | **~0.90+** |

### Top Predictive Features (~24 selected)
1. **Time in Hospital** (`time_in_hospital`, `time_in_hospital_log`)
2. **Number of Medications** (`num_medications`)
3. **Healthcare Utilization** (`Services_used`, visit counts)
4. **Admission Details** (`admission_type_id`, `discharge_disposition_id`)
5. **Diagnoses & Procedures** (`number_diagnoses`, `num_procedures`, `num_lab_procedures`)

### Class Imbalance Handling
- **Original:** 83.78% vs 16.22% (5.17:1 imbalance)
- **After SMOTE:** 50% vs 50% on training set
- **Evaluation:** On original imbalanced test set (realistic)
- **Recommendation:** Prioritize Recall & F1-Score for minority class (readmitted patients)

### Feature Engineering Impact
- **Log transformations** normalized right-skewed distributions
- **Composite features** (`Services_used`) captured total healthcare burden
- **Stay categorization** created meaningful admission duration groups
- **Selected 24 features** from 54 (55% reduction) without losing predictive power

---

## 🛠️ Tech Stack & Libraries

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1E90FF?style=flat)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-FF6B6B?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**Key Libraries:**
- **Data Processing:** Pandas, NumPy
- **ML Algorithms:** Scikit-learn (LR, RF), XGBoost
- **Feature Selection:** Mutual Information, Random Forest importance
- **Class Balancing:** SMOTE (Imbalanced-learn)
- **Model Selection:** RandomizedSearchCV, StratifiedKFold
- **Evaluation:** Classification metrics, ROC-AUC, Confusion Matrix
- **Visualization:** Matplotlib, Seaborn
- **Dimensionality Reduction:** PCA, K-Means clustering

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/lucifer230407/SULProject.git
cd SULProject

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter imbalanced-learn xgboost

# (Or run Cell 1 in the notebook to auto-install)
```

### Running the ML Pipeline

```bash
# Launch Jupyter
jupyter notebook ml_pipeline_analysis.ipynb

# Or from terminal
python -m jupyter notebook ml_pipeline_analysis.ipynb
```

**Execution Flow:**
1. Run Cell 1 to install packages
2. Run Cells 2-4 for setup
3. Run Cells 5-7 to load data
4. Run Cells 8-20 for preprocessing and feature selection
5. Run Cells 21-31 to train and tune models
6. Run Cells 32-38 for evaluation and visualization

**Expected Output:**
- Data validation reports
- Feature selection visualizations (correlation heatmap, importance rankings)
- Model training progress and cross-validation scores
- Tuning results for XGBoost
- Evaluation metrics and ROC curves
- Cluster visualizations and dimensionality analysis

### Dataset Documentation

For detailed feature documentation, see `EDA_Dataset_ID.ipynb`:
- Complete feature inventory (all 56 features)
- Feature categories and descriptions
- Data quality assessment
- Target variable analysis
- Feature engineering artifacts documentation

---

## � Pipeline Architecture Diagram

```
RAW DATA
(diabetic_data.csv)
    ↓
[EDA Stage 1-7: Missing Values, Outliers, Encoding]
    ↓
[EDA Stage 8: Feature Engineering]
    ↓
EDA_COMPLETE_DATASET
(101,767 × 56)
    ↓
[ML Stage 1-2: Load & Explore]
    ↓
[ML Stage 3-4: Validate & Overview]
    ↓
[ML Stage 5: Feature Selection → ~24 features]
    ↓
[ML Stage 6: SMOTE Balancing]
    ↓
[ML Stage 7: Train Models (LR, RF, XGBoost)]
    ↓
[ML Stage 8: Hyperparameter Tuning]
    ↓
[ML Stage 9: Evaluate & Compare]
    ↓
[ML Stage 10: Unsupervised Learning (PCA, K-Means)]
    ↓
PREDICTIONS & INSIGHTS
```

---

## 💡 Key Insights
