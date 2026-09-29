"""
Week 5 - Deep Learning Application in Data Science
Project: Deep Learning-Based Student Performance Prediction

Dataset:
    UCI Student Performance - Mathematics (student-mat.csv)

Task:
    Regression - predict final Mathematics grade G3 (0-20).

The script:
1. Loads the local UCI dataset or downloads the official UCI archive.
2. Performs data-quality checks.
3. Creates a leakage-aware train/validation/test split.
4. Encodes categorical variables and standardizes numerical variables.
5. Trains two neural-network experiments:
      A. Full model: includes G1 and G2.
      B. Early-intervention model: excludes G1 and G2.
6. Uses Batch Normalization, Dropout, EarlyStopping and ReduceLROnPlateau.
7. Evaluates using MAE, RMSE and R2.
8. Saves figures, metrics, model and preprocessing objects.

Run:
    pip install -r requirements.txt
    python week5_deep_learning_student_performance.py
"""

from pathlib import Path
import io
import zipfile
import urllib.request
import random
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# Configuration
# ============================================================

SEED = 42
DATA_DIR = Path("data")
FIG_DIR = Path("figures")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

for folder in [DATA_DIR, FIG_DIR, OUTPUT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "student-mat.csv"

# Official UCI archive.
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student.zip"


def set_seeds(seed: int = SEED) -> None:
    """Make the experiment as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def download_dataset() -> None:
    """Download and extract student-mat.csv when it is not present."""
    if CSV_PATH.exists():
        return

    print("Downloading official UCI Student Performance dataset...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member = next(
            name for name in zf.namelist()
            if name.endswith("student-mat.csv")
        )
        CSV_PATH.write_bytes(zf.read(member))

    print(f"Saved dataset to: {CSV_PATH}")


def load_data() -> pd.DataFrame:
    download_dataset()

    # The original UCI file is semicolon-separated.
    df = pd.read_csv(CSV_PATH, sep=";")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run basic quality checks before modeling."""
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    assert df.shape[0] == 395, "Unexpected number of Mathematics records."
    assert "G3" in df.columns
    assert df["G3"].between(0, 20).all(), "G3 contains values outside 0-20."


def make_preprocessor(
    X_train: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
):
    """
    Fit preprocessing ONLY on the training data.

    This prevents test-set statistics from leaking into model training.
    """
    # sklearn renamed sparse=False -> sparse_output=False.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", encoder, categorical_features),
        ],
        remainder="drop",
    )

    preprocessor.fit(X_train)
    return preprocessor


def build_model(input_dim: int) -> keras.Model:
    """
    Small MLP designed for a 395-row tabular dataset.

    Architecture:
        Input
        Dense(32, ReLU)
        BatchNorm
        Dropout(0.20)
        Dense(16, ReLU)
        Dropout(0.10)
        Dense(1, Linear)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.20),

        layers.Dense(16, activation="relu"),
        layers.Dropout(0.10),

        layers.Dense(1, activation="linear"),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return model


def evaluate_regression(y_true, y_pred) -> dict:
    """Return standard regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_experiment(
    df: pd.DataFrame,
    include_g1_g2: bool,
    experiment_name: str,
):
    """
    Run one complete experiment.

    G1/G2 are optionally excluded to create an early-intervention
    scenario that predicts G3 before those grades are available.
    """

    target = "G3"

    all_numerical = [
        "age", "Medu", "Fedu", "traveltime", "studytime",
        "failures", "famrel", "freetime", "goout", "Dalc",
        "Walc", "health", "absences"
    ]

    if include_g1_g2:
        numerical_features = all_numerical + ["G1", "G2"]
    else:
        numerical_features = all_numerical

    categorical_features = [
        "school", "sex", "address", "famsize", "Pstatus",
        "Mjob", "Fjob", "reason", "guardian", "schoolsup",
        "famsup", "paid", "activities", "nursery", "higher",
        "internet", "romantic"
    ]

    X = df[numerical_features + categorical_features].copy()
    y = df[target].astype("float32").copy()

    # First split: 70% train, 30% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=SEED,
    )

    # Second split: half validation and half test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=SEED,
    )

    preprocessor = make_preprocessor(
        X_train,
        numerical_features,
        categorical_features,
    )

    X_train_t = preprocessor.transform(X_train).astype("float32")
    X_val_t = preprocessor.transform(X_val).astype("float32")
    X_test_t = preprocessor.transform(X_test).astype("float32")

    model = build_model(X_train_t.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train_t,
        y_train,
        validation_data=(X_val_t, y_val),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    predictions = model.predict(X_test_t, verbose=0).reshape(-1)

    metrics = evaluate_regression(y_test, predictions)

    # Mean baseline for contextual comparison.
    baseline_prediction = np.full(
        len(y_test),
        float(y_train.mean()),
        dtype="float32",
    )
    baseline_metrics = evaluate_regression(
        y_test,
        baseline_prediction,
    )

    # -------------------------
    # Save metrics
    # -------------------------
    metrics_out = {
        "experiment": experiment_name,
        "include_G1_G2": include_g1_g2,
        "input_features_after_encoding": int(X_train_t.shape[1]),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "test_MAE": metrics["MAE"],
        "test_RMSE": metrics["RMSE"],
        "test_R2": metrics["R2"],
        "baseline_MAE": baseline_metrics["MAE"],
        "baseline_RMSE": baseline_metrics["RMSE"],
        "baseline_R2": baseline_metrics["R2"],
    }

    with open(
        OUTPUT_DIR / f"{experiment_name}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_out, f, indent=2)

    # -------------------------
    # Training curves
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training vs Validation Loss - {experiment_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{experiment_name}_training_curve.png",
        dpi=180,
    )
    plt.close()

    # -------------------------
    # Actual vs predicted
    # -------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.75)
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Actual vs Predicted - {experiment_name}")
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{experiment_name}_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close()

    # Save model and preprocessing.
    model.save(MODEL_DIR / f"{experiment_name}.keras")
    with open(
        MODEL_DIR / f"{experiment_name}_preprocessor.pkl",
        "wb",
    ) as f:
        pickle.dump(preprocessor, f)

    return metrics_out


def main():
    set_seeds()

    df = load_data()
    validate_data(df)

    print("\nDataset summary:")
    print(df[["G1", "G2", "G3", "studytime", "failures", "absences"]].describe())

    print("\nRunning full-feature experiment...")
    full_metrics = run_experiment(
        df,
        include_g1_g2=True,
        experiment_name="full_mlp",
    )

    print("\nRunning early-intervention experiment...")
    early_metrics = run_experiment(
        df,
        include_g1_g2=False,
        experiment_name="early_mlp",
    )

    comparison = pd.DataFrame([full_metrics, early_metrics])
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nFinal comparison:")
    print(
        comparison[
            [
                "experiment",
                "test_MAE",
                "test_RMSE",
                "test_R2",
            ]
        ]
    )

    print("\nProject completed.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Metrics saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
