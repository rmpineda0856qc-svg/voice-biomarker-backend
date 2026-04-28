"""Risk classification service.

Loads a trained Random Forest model if available. Otherwise, falls back to a
deterministic rule-based classifier so the system is fully functional from day 1
while you collect data and train the real model.

This is intentional — during your capstone timeline you need a working end-to-end
demo long before you have labeled training data. Swap in the real model by:
  1. Training a RandomForestClassifier in train_model.py
  2. Saving it to model/risk_classifier.joblib
  3. The service auto-detects and uses it on next startup.
"""
import os
import joblib
from typing import Dict, List, Tuple
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "risk_classifier.joblib"
_model = None
_feature_names: List[str] = []


def _try_load_model():
    """Attempt to load the trained model. Silently fails if not found."""
    global _model, _feature_names
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            _model = bundle["model"]
            _feature_names = bundle["feature_names"]
            print(f"[classifier] Loaded trained model from {MODEL_PATH}")
        except Exception as e:
            print(f"[classifier] Could not load model: {e}")
            _model = None


_try_load_model()


def _rule_based_classify(
    biomarkers: Dict[str, float],
    deltas: Dict[str, float],
    air_quality: Dict,
    demographics: Dict,
) -> Tuple[str, float, List[str]]:
    """Deterministic fallback classifier used before the ML model is trained.

    Scores risk based on:
      - Voice feature deviations from baseline (jitter, shimmer, HNR)
      - Air quality levels (PM2.5, NO2, O3)
      - User risk factors (asthma, smoking)

    Returns (risk_level, confidence, top_factors).
    """
    score = 0.0
    factors: List[Tuple[str, float]] = []  # (factor_name, contribution)

    # --- Voice biomarker deviations ---
    # Reference thresholds based on Praat/clinical voice literature:
    #   Healthy jitter < 1.04%, healthy shimmer < 3.81%, healthy HNR > 20 dB
    jitter = biomarkers.get("jitter_pct", 0)
    shimmer = biomarkers.get("shimmer_pct", 0)
    hnr = biomarkers.get("hnr_db", 25)

    if jitter > 1.04:
        contrib = min((jitter - 1.04) * 1.0, 2.0)
        score += contrib
        factors.append(("jitter", contrib))
    if shimmer > 3.81:
        contrib = min((shimmer - 3.81) * 0.3, 2.0)
        score += contrib
        factors.append(("shimmer", contrib))
    if hnr < 20:
        contrib = min((20 - hnr) * 0.15, 2.0)
        score += contrib
        factors.append(("hnr", contrib))

    # Additional penalty for deviation from user's personal baseline
    if abs(deltas.get("delta_jitter", 0)) > 0.3:
        score += 0.5
        factors.append(("baseline_jitter_shift", 0.5))
    if abs(deltas.get("delta_hnr", 0)) > 3:
        score += 0.5
        factors.append(("baseline_hnr_drop", 0.5))

    # --- Air quality ---
    pm25 = air_quality.get("pm25")
    no2 = air_quality.get("no2")
    o3 = air_quality.get("o3")

    # WAQI iaqi values are AQI sub-index values; 50 = moderate, 100 = unhealthy for sensitive
    if pm25 is not None:
        if pm25 > 150:
            score += 2.5
            factors.append(("pm25", 2.5))
        elif pm25 > 100:
            score += 1.5
            factors.append(("pm25", 1.5))
        elif pm25 > 50:
            score += 0.7
            factors.append(("pm25", 0.7))
    if no2 is not None and no2 > 100:
        score += 1.0
        factors.append(("no2", 1.0))
    if o3 is not None and o3 > 100:
        score += 1.0
        factors.append(("o3", 1.0))

    # --- Demographic modifiers ---
    if demographics.get("has_asthma"):
        score *= 1.3
        factors.append(("asthma_history", 0.5))
    if demographics.get("smoker"):
        score *= 1.2
        factors.append(("smoker", 0.4))
    age = demographics.get("age", 30)
    if age >= 60 or age <= 15:
        score *= 1.15

    # --- Risk level thresholds ---
    if score >= 5.0:
        risk = "High"
        confidence = min(0.65 + score * 0.03, 0.92)
    elif score >= 2.5:
        risk = "Moderate"
        confidence = 0.70
    else:
        risk = "Low"
        confidence = 0.80

    # Top 3 factors by contribution
    factors.sort(key=lambda x: x[1], reverse=True)
    top = [f[0] for f in factors[:3]] if factors else ["baseline_stable", "good_air_quality"]

    return risk, round(confidence, 2), top


def _ml_classify(
    biomarkers: Dict[str, float],
    deltas: Dict[str, float],
    air_quality: Dict,
    demographics: Dict,
) -> Tuple[str, float, List[str]]:
    """Use the trained RandomForestClassifier."""
    import numpy as np
    # Build feature vector in the order expected by the model
    feature_map = {
        "f0_hz": biomarkers.get("f0_hz", 0),
        "jitter_pct": biomarkers.get("jitter_pct", 0),
        "shimmer_pct": biomarkers.get("shimmer_pct", 0),
        "hnr_db": biomarkers.get("hnr_db", 0),
        "delta_f0": deltas.get("delta_f0", 0),
        "delta_jitter": deltas.get("delta_jitter", 0),
        "delta_shimmer": deltas.get("delta_shimmer", 0),
        "delta_hnr": deltas.get("delta_hnr", 0),
        "pm25": air_quality.get("pm25") or 0,
        "no2": air_quality.get("no2") or 0,
        "o3": air_quality.get("o3") or 0,
        "age": demographics.get("age", 30),
        "gender_male": 1 if demographics.get("gender") == "male" else 0,
        "gender_female": 1 if demographics.get("gender") == "female" else 0,
        "smoker": 1 if demographics.get("smoker") else 0,
        "has_asthma": 1 if demographics.get("has_asthma") else 0,
    }
    X = np.array([[feature_map[name] for name in _feature_names]])
    pred = _model.predict(X)[0]
    proba = _model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    # Feature importances give us the top factors
    importances = _model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    top = [_feature_names[i] for i in top_idx]

    return str(pred), round(confidence, 2), top


def classify_risk(
    biomarkers: Dict[str, float],
    deltas: Dict[str, float],
    air_quality: Dict,
    demographics: Dict,
) -> Tuple[str, float, List[str]]:
    """Classify respiratory risk. Uses ML model if available, otherwise rule-based."""
    if _model is not None:
        try:
            return _ml_classify(biomarkers, deltas, air_quality, demographics)
        except Exception as e:
            print(f"[classifier] ML classify failed, falling back to rules: {e}")
    return _rule_based_classify(biomarkers, deltas, air_quality, demographics)


def build_recommendation(risk: str, air_quality: Dict, top_factors: List[str]) -> str:
    """Generate actionable health advice based on the risk level and context."""
    pm25 = air_quality.get("pm25") or 0
    high_pm25 = pm25 > 100

    if risk == "Low":
        return (
            "Your voice and local air quality readings look good. "
            "Maintain hydration, continue normal activities, and check in again tomorrow."
        )

    if risk == "Moderate":
        parts = ["We detected mild signs of respiratory strain."]
        if high_pm25:
            parts.append(
                "Air quality in your area is elevated. "
                "Consider wearing an N95 mask when outdoors and limit strenuous activity."
            )
        else:
            parts.append(
                "Try resting your voice, drinking warm water, "
                "and re-testing in a few hours."
            )
        return " ".join(parts)

    # High risk
    parts = ["Several indicators suggest significant respiratory strain."]
    if high_pm25:
        parts.append(
            "Air quality is poor. Stay indoors if possible, "
            "use an air purifier, and wear an N95 mask when going out."
        )
    parts.append(
        "If you experience persistent coughing, shortness of breath, "
        "or chest tightness, please consult a healthcare provider. "
        "This tool is a screening aid, not a medical diagnosis."
    )
    return " ".join(parts)
