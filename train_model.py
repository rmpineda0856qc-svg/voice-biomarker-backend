"""Train a Random Forest Classifier for respiratory risk screening.

This script:
  1. Generates a synthetic training dataset based on published voice biomarker
     thresholds and reasonable air quality / demographic distributions.
  2. Trains a RandomForestClassifier with hyperparameter search.
  3. Evaluates on a held-out test set.
  4. Saves the model to model/risk_classifier.joblib so the backend can load it.

IMPORTANT: The synthetic data is an educated placeholder for the starter project.
Replace with real collected data (voice recordings + self-reported symptoms +
paired WAQI readings) before your final capstone submission.

Usage:
    cd backend
    python train_model.py

Output:
    model/risk_classifier.joblib    <- the trained model bundle
    model/training_report.txt       <- metrics and feature importances
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

np.random.seed(42)

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "f0_hz", "jitter_pct", "shimmer_pct", "hnr_db",
    "delta_f0", "delta_jitter", "delta_shimmer", "delta_hnr",
    "pm25", "no2", "o3",
    "age", "gender_male", "gender_female", "smoker", "has_asthma",
]


def generate_synthetic_sample(risk_class: str) -> dict:
    """Generate a single synthetic training row for a given risk class.

    Ranges are rough approximations derived from voice biomarker literature
    (healthy adult jitter <1.04%, shimmer <3.81%, HNR >20dB) and WAQI pollutant
    thresholds. Replace this with real collected data for your final model.
    """
    # Demographic base
    age = int(np.random.normal(35, 15))
    age = max(15, min(75, age))
    is_male = np.random.random() < 0.5
    smoker = np.random.random() < 0.15
    has_asthma = np.random.random() < 0.10

    # Baseline pitch varies by gender
    baseline_f0 = np.random.normal(115 if is_male else 210, 15)

    if risk_class == "Low":
        f0 = baseline_f0 + np.random.normal(0, 3)
        jitter = max(0.1, np.random.normal(0.6, 0.2))
        shimmer = max(0.5, np.random.normal(2.5, 0.7))
        hnr = np.random.normal(24, 2)
        pm25 = np.random.uniform(0, 60)
        no2 = np.random.uniform(0, 40)
        o3 = np.random.uniform(0, 40)
    elif risk_class == "Moderate":
        f0 = baseline_f0 + np.random.normal(0, 6)
        jitter = max(0.3, np.random.normal(1.1, 0.3))
        shimmer = max(1.0, np.random.normal(4.0, 1.0))
        hnr = np.random.normal(19, 2.5)
        pm25 = np.random.uniform(40, 130)
        no2 = np.random.uniform(30, 90)
        o3 = np.random.uniform(20, 90)
    else:  # High
        f0 = baseline_f0 + np.random.normal(0, 10)
        jitter = max(0.8, np.random.normal(1.9, 0.5))
        shimmer = max(2.5, np.random.normal(6.5, 1.5))
        hnr = np.random.normal(14, 3)
        pm25 = np.random.uniform(100, 250)
        no2 = np.random.uniform(60, 160)
        o3 = np.random.uniform(50, 150)

    # Deltas from personal baseline
    delta_f0 = f0 - baseline_f0
    # Baseline jitter/shimmer/hnr drawn from normal healthy range
    baseline_jitter = max(0.1, np.random.normal(0.6, 0.15))
    baseline_shimmer = max(0.5, np.random.normal(2.5, 0.5))
    baseline_hnr = np.random.normal(24, 1.5)

    return {
        "f0_hz": round(f0, 2),
        "jitter_pct": round(jitter, 3),
        "shimmer_pct": round(shimmer, 3),
        "hnr_db": round(hnr, 2),
        "delta_f0": round(delta_f0, 2),
        "delta_jitter": round(jitter - baseline_jitter, 3),
        "delta_shimmer": round(shimmer - baseline_shimmer, 3),
        "delta_hnr": round(hnr - baseline_hnr, 2),
        "pm25": round(pm25, 1),
        "no2": round(no2, 1),
        "o3": round(o3, 1),
        "age": age,
        "gender_male": 1 if is_male else 0,
        "gender_female": 0 if is_male else 1,
        "smoker": 1 if smoker else 0,
        "has_asthma": 1 if has_asthma else 0,
        "risk_level": risk_class,
    }


def generate_dataset(n_per_class: int = 500) -> pd.DataFrame:
    rows = []
    for risk in ["Low", "Moderate", "High"]:
        for _ in range(n_per_class):
            rows.append(generate_synthetic_sample(risk))
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Voice Biomarker Risk Classifier — Training")
    print("=" * 60)

    print("\n[1/4] Generating synthetic dataset...")
    df = generate_dataset(n_per_class=500)
    print(f"      Total samples: {len(df)}")
    print(f"      Class distribution:\n{df['risk_level'].value_counts().to_string()}")

    X = df[FEATURE_NAMES]
    y = df["risk_level"]

    print("\n[2/4] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n[3/4] Hyperparameter search (GridSearchCV, 5-fold)...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"      Best params: {grid.best_params_}")
    print(f"      Best CV F1 score: {grid.best_score_:.4f}")

    print("\n[4/4] Evaluating on test set...")
    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"])
    print(report)
    print("Confusion matrix (rows=true, cols=pred, order=Low/Moderate/High):")
    print(cm)

    # Feature importances
    importances = sorted(
        zip(FEATURE_NAMES, best.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop feature importances:")
    for name, imp in importances[:10]:
        print(f"  {name:25s} {imp:.4f}")

    # Save model bundle
    bundle = {
        "model": best,
        "feature_names": FEATURE_NAMES,
        "best_params": grid.best_params_,
    }
    out_path = MODEL_DIR / "risk_classifier.joblib"
    joblib.dump(bundle, out_path)
    print(f"\nModel saved to: {out_path}")

    # Save training report
    report_path = MODEL_DIR / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("Voice Biomarker Risk Classifier — Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Best params: {grid.best_params_}\n")
        f.write(f"Best CV F1: {grid.best_score_:.4f}\n\n")
        f.write("Classification report (test set):\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nFeature importances:\n")
        for name, imp in importances:
            f.write(f"  {name:25s} {imp:.4f}\n")
    print(f"Report saved to: {report_path}")

    print("\nDone. Restart the backend to load the new model.")


if __name__ == "__main__":
    main()
