# TPdsm

## About

TPdsm is a synonymous mutation pathogenicity prediction tool that combines TabPFN (a Transformer-based Pre-trained model for small tabular data) with Genetic Algorithm (GA) feature selection for enhanced prediction performance.

## Directory Structure

    TPdsm-main/
    ├── data/                                    # Training and test datasets
    │   ├── Trainning dataset.xlsx              # Training dataset
    │   ├── Testing dataset 1.xlsx              # Testing dataset 1
    │   ├── Testing dataset 2.xlsx              # Testing dataset 2
    │   ├── Testing dataset 3.xlsx              # Testing dataset 3
    │   └── Testing dataset 4.xlsx  # Testing dataset 4
    ├── docs/                                    # Documentation
    │   └── tutorial.md                          # Usage tutorial
    ├── model/                                   # Pre-trained model and features
    │   ├── TPdsm.pkl                           # Trained TabPFN model
    │   └── feature.json                        # Selected features
    ├── script/                                  # Core scripts
    │   ├── tabpfn4syntool_withGA.py            # GA + TabPFN for feature selection and training
    │   └── Compare_Synmethod_TPdsm_metrics.py  # Compare TPdsm with other tools and calculate metrics
    ├── LICENSE                                  # License file
    ├── README.md                                # This file
    └── conda.yaml                               # Conda environment configuration

## Installation

### Prerequisites

*   Python 3.10 or higher

*   conda package manager

### Using conda with conda.yaml

```bash
# Clone the repository
git clone https://github.com/Project4bz2023/TPdsm.git
cd TPdsm

# Create and activate conda environment
conda env create -f conda.yaml
conda activate tpdsm
```

## Quick Start

### 1. Data Preparation

Place your annotated variant files in the `data/` directory. The files should be in Excel format (.xlsx) with the same structure as the provided training and test datasets.

### 2. Feature Selection and Model Training

```bash
cd script
python tabpfn4syntool_withGA.py
```

Note: Before running, modify the `file_dir` variable in the script to point to your data directory.

### 3. Compare with Other Tools

```bash
cd script
python Compare_Synmethod_TPdsm_metrics.py
```

Note: Before running, modify `model_dir` and `file_dir` variables in the script.

## Script Details

### tabpfn4syntool\_withGA.py

This script performs:

*   **Data preprocessing**: Handles missing values by imputing with column means

*   **Constant feature removal**: Filters out features with zero variance

*   **Genetic Algorithm (GA) feature selection**:

    *   Initializes a population of random feature subsets

    *   Evaluates fitness using TabPFN classifier performance

    *   Uses crossover and mutation to evolve feature combinations

    *   Saves models that meet predefined AUC thresholds

*   **Results generation**: Saves performance metrics for all test sets

Key parameters:

*   `best_score`: Minimum AUC threshold for models to be saved (default: 0.80)

*   `population_size`: Size of GA population (default: 100)

*   `generations`: Number of GA generations (default: 100)

Outputs:

*   `feature_{N}.json`: Selected features for the Nth model

*   `bestmodel_{N}.pkl`: Saved TabPFN model

*   `result_metric_list.csv`: Performance metrics across all test sets

### Compare\_Synmethod\_TPdsm\_metrics.py

This script:

*   Loads a pre-trained TPdsm model and its selected features

*   Compares TPdsm performance against 13 existing synonymous mutation prediction tools:

    *   CADD

    *   DANN

    *   DDIG

    *   Eigen

    *   EnDSM

    *   Fathmm\_MKL\_coding

    *   Fathmm\_XF\_coding

    *   frDSM

    *   PhD\_SNPg

    *   PrDSM

    *   SilVA

    *   Syntool

    *   usDSM

*   Generates ROC and PRC curves in PDF format for all test sets

*   Calculates comprehensive metrics (AUC, AUPRC, F1, MCC, Accuracy, Precision, Sensitivity, Specificity)

Outputs:

*   `SYNmethod_compare_TPdsm/` directory with ROC and PRC plots

*   `SYNmethod_compare_TPdsm_metrics_*.csv` files with detailed metrics for each test set

## Pre-trained Model

The `model/` directory contains a pre-trained model ready for use:

*   `TPdsm.pkl`: Trained TabPFN model

*   `feature.json`: Selected features used by the model

### Using the Pre-trained Model

```python
import joblib
import json
import pandas as pd

# Load model and features
model = joblib.load("model/TPdsm.pkl")
with open("model/feature.json", "r") as f:
    features = json.load(f)

# Load your data
data = pd.read_excel("your_data.xlsx")

# Preprocess and predict
X = data[features]
predictions = model.predict_proba(X)[:, 1]
```

## Citation

If you use TPdsm in your research, please cite:

    [Placeholder for citation information]
    Authors. (Year). TPdsm: a method based on TabPFN for prediction of deleterious synonymous mutations. Journal Name, Volume(Issue), Pages.

## FAQ

### Q: What input format is required?

A: TPdsm accepts txt files with annotated variant data. The label column should be named `Otherinfo1` where 1 indicates pathogenic and 0 indicates benign.

### Q: How long does feature selection take?

A: The runtime depends on your hardware, population size, and number of generations. With default settings (pop\=100, gens\=100), it typically takes several hours on a modern CPU.

### Q: Can I use TPdsm on my own dataset?

A: Yes! Simply place your annotated data in the `data/` directory and update the file paths in the scripts.

### Q: How do I interpret the prediction scores?

A: TPdsm outputs probabilities between 0 and 1. Higher scores indicate higher likelihood of pathogenicity.  We used 0.09 in Compare\_Synmethod\_TPdsm\_metrics.py.  You may optimal thresholds according to your dataset.

## Result Interpretation

### Performance Metrics

The `result_metric_list.csv` file contains the following metrics for each test set:

*   **AUC**: Area Under the Receiver Operating Characteristic Curve

*   **AUPRC**: Area Under the Precision-Recall Curve

*   **F1-score**: Harmonic mean of precision and recall

*   **MCC**: Matthews Correlation Coefficient

*   **Accuracy**: Overall prediction accuracy

*   **Precision**: Positive predictive value

*   **Recall/Sensitivity**: True positive rate

### ROC and PRC Curves

*   **ROC Curves**: Show trade-off between true positive rate and false positive rate at various thresholds

*   **PRC Curves**: Show trade-off between precision and recall, particularly useful for imbalanced datasets

Both plots include AUC/AUPRC values in the legend for easy comparison between tools.
