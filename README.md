# Week 5 — Deep Learning Application in Data Science

## Project Title

**Deep Learning-Based Student Performance Prediction**

This project uses a Feed-Forward Neural Network (MLP) to predict the final Mathematics grade (`G3`) from the UCI Student Performance dataset.

---

## 1. Dataset

**Dataset:** UCI Student Performance — Mathematics

| Property | Value |
|---|---|
| Records | 395 students |
| Attributes | 33 |
| Target | G3 |
| Target scale | 0–20 |
| Task | Regression |
| File | `data/student-mat.csv` |
| CSV separator | `;` |

The dataset contains demographic, social, school-related, study-behaviour and academic-history information.

### Dataset download

The project includes a dedicated downloader:

```bash
python download_dataset.py
```

It automatically downloads the official UCI archive and extracts:

```text
data/student-mat.csv
```

Official UCI page:

https://archive.ics.uci.edu/dataset/320/student+performance

Official archive:

https://archive.ics.uci.edu/static/public/320/student.zip

---

## 2. Project Structure

```text
Week5_Deep_Learning_Student_Performance/
│
├── data/
│   ├── student-mat.csv
│   └── README.md
│
├── figures/
│   ├── network_architecture.png
│   ├── rmse_comparison.png
│   └── r2_comparison.png
│
├── models/
│   └── Generated after training
│
├── outputs/
│   └── Generated after training
│
├── Week5_Deep_Learning_Student_Performance_Report.docx
├── week5_deep_learning_student_performance.py
├── download_dataset.py
├── requirements.txt
└── README.md
```

`models/` and `outputs/` are generated automatically when the training script is executed.

---

## 3. Problem Statement

The objective is to predict the final Mathematics grade (`G3`) of a student using information such as:

- age
- parental education
- study time
- previous failures
- absences
- family relationship
- free time
- social activity
- health
- school information
- family information
- educational support
- internet access
- and other available attributes

The project treats the problem as **supervised regression**.

---

## 4. Important Feature Decision: G1 and G2

The dataset contains:

- `G1` — first-period grade
- `G2` — second-period grade
- `G3` — final grade

G1 and G2 have very strong relationships with G3 because they are earlier grades from the same course.

Therefore, the project evaluates two scenarios:

### Experiment A — Full Model

Includes:

```text
G1 + G2 + other features
```

This represents prediction when earlier grades are already available.

### Experiment B — Early-Intervention Model

Excludes:

```text
G1 and G2
```

This represents a more difficult scenario where the model attempts to identify performance risk before those earlier grades are available.

This distinction is important because selecting G1/G2 only because they have high correlation with G3 would otherwise make the model's practical interpretation unclear.

---

## 5. Preprocessing

The pipeline performs:

1. Train/validation/test split.
2. Numerical feature standardization using `StandardScaler`.
3. Categorical feature encoding using `OneHotEncoder`.
4. Encoder and scaler are fitted only on the training data.
5. Validation and test data are transformed using the training-fitted preprocessing pipeline.

This prevents preprocessing leakage.

---

## 6. Neural Network Architecture

```text
Input
  ↓
Dense(32) + ReLU
  ↓
Batch Normalization
  ↓
Dropout(0.20)
  ↓
Dense(16) + ReLU
  ↓
Dropout(0.10)
  ↓
Dense(1) + Linear
  ↓
Predicted G3
```

The network is intentionally small because the dataset contains only 395 observations.

---

## 7. Training Configuration

| Parameter | Setting |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean Squared Error |
| Main evaluation metrics | MAE, RMSE, R² |
| Batch size | 16 |
| Maximum epochs | 150 |
| Early stopping | Patience 15 |
| Reduce LR | Factor 0.5 |
| Dropout | 0.20 and 0.10 |
| Random seed | 42 |

---

## 8. Evaluation

The project evaluates:

### MAE

Mean Absolute Error measures the average absolute grade prediction error.

### RMSE

Root Mean Squared Error gives greater weight to larger prediction errors.

### R²

R² measures how much variance in the target is explained relative to a mean-prediction baseline.

---

## 9. Reference Benchmark

A deterministic reference experiment on the 395-record dataset produced:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 4.110 | 5.239 | -0.042 |
| Full MLP — G1/G2 included | 2.453 | 3.056 | 0.645 |
| Early MLP — G1/G2 excluded | 4.321 | 5.162 | -0.012 |

These values are included as a reference benchmark. The TensorFlow/Keras implementation may produce slightly different values because TensorFlow training, initialization and callback behaviour can vary.

---

## 10. Installation

Recommended Python version:

```text
Python 3.10+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 11. Download Dataset

Run:

```bash
python download_dataset.py
```

After successful download:

```text
data/student-mat.csv
```

should exist.

You can verify it with:

```python
import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=";")

print(df.shape)
print(df.head())
```

Expected:

```text
(395, 33)
```

---

## 12. Run the Complete Project

From the project root:

```bash
python download_dataset.py
python week5_deep_learning_student_performance.py
```

The training script will:

- load the dataset
- validate the data
- create train/validation/test partitions
- preprocess numerical and categorical features
- train the full model
- train the early-intervention model
- evaluate both models
- generate training curves
- generate actual-vs-predicted plots
- save model files
- save preprocessing objects
- save metrics

---

## 13. Generated Outputs

After execution:

```text
figures/
├── full_mlp_training_curve.png
├── full_mlp_actual_vs_predicted.png
├── early_mlp_training_curve.png
└── early_mlp_actual_vs_predicted.png

models/
├── full_mlp.keras
├── full_mlp_preprocessor.pkl
├── early_mlp.keras
└── early_mlp_preprocessor.pkl

outputs/
├── full_mlp_metrics.json
├── early_mlp_metrics.json
└── model_comparison.csv
```

---

## 14. Why Regularization Is Used

The dataset is small relative to the number of encoded features, so overfitting is a significant risk.

The project uses:

- small network architecture
- Dropout
- Batch Normalization
- Early Stopping
- ReduceLROnPlateau
- validation monitoring

These techniques help reduce unnecessary memorization of the training data.

---

## 15. Limitations

The dataset has only 395 observations and comes from two Portuguese secondary schools.

Therefore:

- results may not generalize to all schools
- a single split can be unstable
- neural networks are not automatically superior to classical ML models
- G1/G2 make the prediction problem substantially easier
- correlation does not imply causation

For a stronger research-grade evaluation, repeated k-fold cross-validation should be added.

---

## 16. Future Improvements

Possible next steps:

1. Compare MLP with Linear Regression.
2. Compare with Random Forest.
3. Compare with Gradient Boosting / XGBoost.
4. Use repeated k-fold cross-validation.
5. Perform hyperparameter tuning.
6. Add SHAP/permutation importance.
7. Build an at-risk classification model.
8. Test the model on a larger external dataset.
9. Develop a real-time student risk prediction dashboard.

---

## 17. Author

**Aman Kumar Yadav**

Week 5 — Deep Learning Application in Data Science

---

## 18. License / Dataset Note

The dataset belongs to the UCI Machine Learning Repository and is subject to its dataset terms.

The project code is provided for educational and academic use.
