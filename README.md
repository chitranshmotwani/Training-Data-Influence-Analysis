# Training Data Influence Analysis

This project focuses on understanding the influence of training data on model outputs. The goal is to compute and analyze the influence of individual data points, groups of data points, and Shapley values for a binary classification task using the Adult Income dataset.

## Project Structure
```
adult_income_project/
├── data/                               # Folder for datasets
│   ├── adult.data                      # Original training data from UCI
│   ├── adult.test                      # Original test data from UCI
│   ├── adult.names                     # Dataset description from UCI
│   ├── X_train.csv                     # Preprocessed training features
│   ├── y_train.csv                     # Preprocessed training labels
│   ├── X_test.csv                      # Preprocessed test features
│   └── y_test.csv                      # Preprocessed test labels
├── models/                             # Folder for saved models
│   └── baseline_model.pkl              # Trained baseline model
├── scripts/                            # Folder for Python scripts
│   ├── preprocess.py                   # Script for data preprocessing
│   ├── train.py                        # Script for training the baseline model
│   └── influence.py                    # Script for LOO, group, and Shapley influence
├── reports/                            # Folder for project report
│   ├── roc_curve.png                   # ROC Curve
│   ├── groupsize_vs_influence.png      # Group Size vs Influence graph
│   ├── shapley_distribution.png        # Shapley Distribution graph
│   └── report.pdf                      # Preprocessed test labels
├── README.md                           # Project README file
```

## Dataset

The dataset used is the **Adult Income Dataset** from the UCI Machine Learning Repository. It contains information about individuals (e.g., age, education, occupation) and is used to predict whether an individual earns more than $50K/year.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Task**: Binary classification (income > $50K or <= $50K).

## Scripts

### 1. `preprocess.py`
- Loads the dataset.
- Handles missing values, encodes categorical variables, and normalizes numerical features.
- Splits the data into training and test sets.
- Saves the preprocessed data to `data/`.

**Run the script**:
```bash
python scripts/preprocess.py
```

### 2. `train.py`
- Loads the preprocessed data.
- Trains a logistic regression model as the baseline classifier.
- Evaluates the model and saves it to `models/baseline_model.pkl`.

**Run the script**:
```bash
python scripts/train.py
```

### 3. `influence.py`
- Computes Leave-One-Out (LOO) influence for 10 randomly selected training points.
- Computes group-level influence for 10 groups of different sizes.
- Computes Shapley values for a subsample of the training data using Truncated Monte Carlo Shapley Value Estimation.
- Outputs influence scores and plots.

**Run the script**:
```bash
python scripts/influence.py
```

## Requirements

To run the scripts, ensure you have the following Python packages installed:

- numpy
- pandas
- scikit-learn
- joblib
- matplotlib

You can install the dependencies using:
```bash
pip install numpy pandas scikit-learn joblib matplotlib
```

## Report

The final report includes:

- **Dataset and Preprocessing**: Description of the dataset and preprocessing steps.
- **Baseline Model**: Performance of the logistic regression model (accuracy, confusion matrix, ROC curve).
- **LOO Influence**: Influence scores for 10 training points and analysis.
- **Group-Level Influence**: Influence scores for 10 groups and a plot of group size vs. influence.
- **Shapley Values**: Distribution of Shapley values and analysis.

---

