# TPdsm Comprehensive Usage Tutorial

Welcome to TPdsm! This tutorial will guide you through every step of using TPdsm for synonymous mutation pathogenicity prediction, from environment setup to result interpretation.

***

## Table of Contents

1.  [Getting Started - Environment Setup](#1-getting-started---environment-setup)

2.  [Preparing Input Data](#2-preparing-input-data)

3.  [Running Feature Selection and Model Training](#3-running-feature-selection-and-model-training)

4.  [Using Pre-trained Model for Prediction](#4-using-pre-trained-model-for-prediction)

5.  [Comparing with Other Tools](#5-comparing-with-other-tools)

6.  [Result Interpretation](#6-result-interpretation)

***

## 1. Getting Started - Environment Setup

### Prerequisites

*   Python 3.10 or higher

*   conda package manager

*   Git (for cloning the repository)

### Installation with conda

```bash
# 1. Clone the repository
git clone https://github.com/Project4bz2023/TPdsm.git
cd TPdsm

# 2. Create and activate conda environment
conda env create -f conda.yaml
conda activate tpdsm
```

***

## 2. Preparing Input Data

### Data Format

TPdsm accepts **Excel files (.xlsx)** as input. The project includes example datasets in the `data/` directory.

### Required Columns

Your data file must include:

1.  **All candidate features** (see `tabpfn4syntool_withGA.py` for the full list of features)

2.  **Label column**: Must be named `Otherinfo1`, where:

    *   `1` \= Pathogenic

    *   `0` \= Benign

### Example Data Structure

Here's a simplified example of what your input file should look like:

    Chr    Start    End    Ref    Alt    CADD_PHRED    DANN    Otherinfo1
    chr1   123456   123457   A      G      23.5          0.82    1
    chr2   789012   789013   C      T      10.2          0.35    0
    ...

### Data Preprocessing

TPdsm automatically handles:

*   **Missing values**: Imputed with column means

*   **Constant features**: Filtered out (features with zero variance)

You don't need to preprocess your data manually - the scripts handle this for you!

### Placing Your Data

Put your training and test datasets in the `data/` directory:

    TPdsm-main/
    ├── data/
    │   ├── Trainning dataset.xlsx
    │   ├── Testing dataset 1.xlsx
    │   ├── Testing dataset 2.xlsx
    │   ├── Testing dataset 3.xlsx
    │   └── Testing dataset 4.xlsx

***

## 3. Running Feature Selection and Model Training

TPdsm uses **TabPFN** (a Transformer-based pre-trained model for small tabular data) combined with a **Genetic Algorithm (GA)** for optimal feature selection.

### Step 1: Configure the Script

Open `script/tabpfn4syntool_withGA.py` and update the following parameters:

#### Key Parameters

| Parameter         | Description                                                                  | Default Value    |
| ----------------- | ---------------------------------------------------------------------------- | ---------------- |
| `best_score`      | Minimum AUC threshold (all test sets must meet this for a model to be saved) | 0.80             |
| `population_size` | Number of feature subsets per GA generation                                  | 100              |
| `generations`     | Number of GA generations to run                                              | 100              |
| `file_dir`        | Path to your data directory                                                  | Must be updated! |

#### Example Configuration

```python
# Set AUC threshold
best_score = 0.80

# Update data directory path
file_dir = "d:/AI_code/project/syn/TPdsm-main/TPdsm-main/data/"

# (Optional) Modify candidate features if needed
```

### Step 2: Run the Training Script

```bash
# Navigate to the script directory
cd script

# Run the feature selection and training
python tabpfn4syntool_withGA.py
```

### What Happens During Training

1.  **Data Loading & Preprocessing**: Loads and preprocesses all datasets

2.  **Constant Feature Removal**: Filters out features with no variation

3.  **Genetic Algorithm**:

    *   Initializes random feature subsets

    *   Evaluates each subset's performance

    *   Uses crossover and mutation to evolve better feature combinations

    *   Saves models that meet the AUC threshold

4.  **Results Generation**: Creates a CSV with performance metrics

### Output Files

After training completes, you'll find these files in the `script/` directory:

| File                     | Description                           |
| ------------------------ | ------------------------------------- |
| `feature_{N}.json`       | Selected features for the Nth model   |
| `bestmodel_{N}.pkl`      | Saved TabPFN model (joblib format)    |
| `result_metric_list.csv` | Performance metrics for all test sets |
| `processed.*.txt`        | Preprocessed data files               |

***

## 4. Using Pre-trained Model for Prediction

Once you have a trained model, you can use it to make predictions on new data.

### Step 1: Load the Model and Features

Here's a code example to load and use a pre-trained model:

```python
import joblib
import json
import pandas as pd
from pathlib import Path

# Set paths
model_dir = Path("/TPdsm-main/model/")
data_dir = Path("/TPdsm-main/data/")

# Load selected features
with open(model_dir / "feature.json", "r") as f:
    selected_features = json.load(f)

# Load the trained model
model = joblib.load(model_dir / "TPdsm.pkl")

# Load new data
new_data = pd.read_excel(data_dir / "your_new_data.xlsx")

# Preprocess data (impute missing values)
def preprocess_data(data, features):
    df = data.copy()
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean_val = df[col].mean(skipna=True)
        df[col].fillna(mean_val, inplace=True)
        df[col] = df[col].astype(float)
    return df

preprocessed_data = preprocess_data(new_data, selected_features)

# Make predictions
X = preprocessed_data[selected_features]
predictions_proba = model.predict_proba(X)[:, 1]  # Probability of pathogenicity
predictions = model.predict(X)  # Binary predictions (0/1)

# Add predictions to the dataframe
new_data["TPdsm_prediction_probability"] = predictions_proba
new_data["TPdsm_prediction"] = predictions

# Save results
new_data.to_excel("predicted_results.xlsx", index=False)

print("Predictions saved to predicted_results.xlsx")
```

### Understanding Predictions

*   `TPdsm_prediction_probability`: Value between 0 and 1

    *   Higher values → Higher likelihood of pathogenicity

    *   Typical threshold: > 0.5 suggests pathogenic

*   `TPdsm_prediction`: Binary prediction (0 \= benign, 1 \= pathogenic)

***

## 5. Comparing with Other Tools

TPdsm includes a script to compare its performance against 14 existing synonymous mutation prediction tools.

### Tools Compared

| Tool Name           | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| CADD                | Combined Annotation Dependent Depletion                           |
| DANN                | Deleterious Annotation of genetic variants using Neural Networks  |
| DDIG                | DNA Damage Interaction Graph                                      |
| Eigen               | Evolutionary and Functional Impact of Non-coding Genomic Variants |
| EnDSM               | Ensembl-based Deleterious Synonymous Mutations                    |
| Fathmm\_MKL\_coding | Functional Analysis through Hidden Markov Models with MKL         |
| Fathmm\_XF\_coding  | FATHMM with extended features                                     |
| frDSM               | Fungal Deleterious Synonymous Mutations                           |
| PhD\_SNPg           | Predictor of Human Deleterious SNPs                               |
| PrDSM               | Plant Deleterious Synonymous Mutations                            |
| SilVA               | Silent Variant Analyzer                                           |
| Syntool             | Synonymous mutation prediction tool                               |
| usDSM               | Unsupervised Deleterious Synonymous Mutations                     |

### Step 1: Configure the Comparison Script

Open `script/Compare_Synmethod_TPdsm_metrics.py` and update these paths:

```python
# Update model directory
model_dir = Path("/TPdsm-main/model/")

# Update feature file name if needed
with open(model_dir / "feature.json", "r", encoding="utf-8") as f:
    trainFeatures = json.load(f)

# Update model file name if needed
model = load(model_dir / "TPdsm.pkl")

# Update data directory
file_dir = "/TPdsm-main/data/"
```

### Step 2: Run the Comparison

```bash
cd script
python Compare_Synmethod_TPdsm_metrics.py
```

### Output of Comparison

The script creates a directory `SYNmethod_compare_TPdsm/` with:

| File                                       | Description               |
| ------------------------------------------ | ------------------------- |
| `SYNmethod_compare_TPdsm_ROCintest1.pdf`   | ROC curves for test set 1 |
| `SYNmethod_compare_TPdsm_PRCintest1.pdf`   | PRC curves for test set 1 |
| `SYNmethod_compare_TPdsm_ROCintest2.pdf`   | ROC curves for test set 2 |
| `SYNmethod_compare_TPdsm_PRCintest2.pdf`   | PRC curves for test set 2 |
| ...and similar files for test sets 3 and 4 |                           |

***

## 6. Result Interpretation

### Performance Metrics

The `result_metric_list.csv` file contains these metrics for each test set:

| Metric             | Full Name                             | Interpretation                                            |
| ------------------ | ------------------------------------- | --------------------------------------------------------- |
| AUC                | Area Under the ROC Curve              | Overall model performance (0.5 \= random, 1.0 \= perfect) |
| AUPRC              | Area Under the Precision-Recall Curve | Performance on imbalanced datasets                        |
| F1-score           | F1 Score                              | Harmonic mean of precision and recall                     |
| MCC                | Matthews Correlation Coefficient      | Balanced measure even for imbalanced classes              |
| Accuracy           | Accuracy                              | Overall correct prediction rate                           |
| Precision          | Precision                             | True positive rate among predicted positives              |
| Recall/Sensitivity | Recall                                | True positive rate among actual positives                 |

### Understanding the Plots

#### ROC Curves (Receiver Operating Characteristic)

*   **X-axis**: False Positive Rate (FPR)

*   **Y-axis**: True Positive Rate (TPR, Sensitivity)

*   The diagonal line represents random guessing (AUC \= 0.5)

*   Curves closer to the top-left corner indicate better performance

*   AUC values are shown in the legend for each tool

#### PRC Curves (Precision-Recall)

*   **X-axis**: Recall (Sensitivity)

*   **Y-axis**: Precision

*   Particularly useful for imbalanced datasets (common in variant prediction)

*   Curves closer to the top-right corner indicate better performance

*   AUPRC values are shown in the legend

### Example Results

Here's how to interpret typical results:

    AUC = 0.88 → Excellent performance (well above random)
    AUPRC = 0.85 → Good performance on imbalanced data
    F1-score = 0.82 → Good balance between precision and recall

### Troubleshooting

**Q: No models were saved?**
A: Try lowering the `best_score` threshold or increasing `population_size`/`generations`.

**Q: The script is taking too long?**
A: Reduce `population_size` or `generations` (but this may affect performance).

**Q: I'm getting errors about missing columns?**
A: Make sure your input data includes all required features and the `Otherinfo1` label column.

***

## Next Steps

*   Explore the example datasets in `data/`

*   Try modifying GA parameters to find optimal feature combinations

*   Use the comparison script to benchmark TPdsm against other tools

*   Check out the README.md for additional information

***

Happy predicting! If you have questions, please refer to the README or open an issue on GitHub.
