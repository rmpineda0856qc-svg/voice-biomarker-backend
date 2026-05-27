"""Database layer using Supabase (PostgreSQL).

Replaces the in-memory store with persistent Supabase database.
Data survives server restarts, Render sleep cycles, and redeployments.
"""
from typing import Optional, List
from datetime import datetime, timezone
import uuid

from app.config import get_settings

# ── Supabase client (lazy-loaded) ─────────────────────────────────────────────
_supabase = None

def _get_client():
    """Get or create Supabase client."""
    global _supabase
    if _supabase is None:
        from supabase import create_client
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_service_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
            )
        _supabase = create_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
    return _supabase


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(email: str, password_hash: str, age: int, gender: str) -> str:
    """Create a new user. Raises ValueError if email already exists."""
    db = _get_client()

    # Check if email already exists
    existing = db.table("users").select("id").eq("email", email).execute()
    if existing.data:
        raise ValueError("Email already registered")

    user_id = str(uuid.uuid4())
    db.table("users").insert({
        "id": user_id,
        "email": email,
        "password_hash": password_hash,
        "age": age,
        "gender": gender,
    }).execute()
    return user_id


def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email address."""
    db = _get_client()
    result = db.table("users").select("*").eq("email", email).execute()
    if not result.data:
        return None
    user = result.data[0]
    user["id"] = user["id"]
    return user


def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID."""
    db = _get_client()
    result = db.table("users").select("*").eq("id", user_id).execute()
    if not result.data:
        return None
    return result.data[0]


# ── Baselines ─────────────────────────────────────────────────────────────────

def save_baseline(user_id: str, biomarkers: dict) -> str:
    """Save or update baseline for a user."""
    db = _get_client()
    baseline_id = str(uuid.uuid4())

    # Delete existing baseline if any (one baseline per user)
    db.table("baselines").delete().eq("user_id", user_id).execute()

    # Insert new baseline
    db.table("baselines").insert({
        "id": baseline_id,
        "user_id": user_id,
        "f0_hz": biomarkers.get("f0_hz"),
        "jitter_pct": biomarkers.get("jitter_pct"),
        "shimmer_pct": biomarkers.get("shimmer_pct"),
        "hnr_db": biomarkers.get("hnr_db"),
        "duration_s": biomarkers.get("duration_s"),
        "rms_energy": biomarkers.get("rms_energy"),
        "voiced_ratio": biomarkers.get("voiced_ratio"),
    }).execute()

    return baseline_id


def get_baseline(user_id: str) -> Optional[dict]:
    """Get baseline for a user."""
    db = _get_client()
    result = db.table("baselines").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return None
    return result.data[0]


# ── Assessments ───────────────────────────────────────────────────────────────

def save_assessment(user_id: str, data: dict) -> str:
    """Save an assessment result."""
    db = _get_client()
    assessment_id = str(uuid.uuid4())

    biomarkers = data.get("biomarkers", {})
    air_quality = data.get("air_quality", {})
    comparison = data.get("biomarker_comparison")

    # Convert comparison to dict if it's a Pydantic model
    if comparison and hasattr(comparison, "model_dump"):
        comparison = comparison.model_dump()

    db.table("assessments").insert({
        "id": assessment_id,
        "user_id": user_id,
        "risk_level": data.get("risk_level"),
        "confidence": data.get("confidence"),
        # Voice biomarkers
        "f0_hz": biomarkers.get("f0_hz"),
        "jitter_pct": biomarkers.get("jitter_pct"),
        "shimmer_pct": biomarkers.get("shimmer_pct"),
        "hnr_db": biomarkers.get("hnr_db"),
        "duration_s": biomarkers.get("duration_s"),
        "rms_energy": biomarkers.get("rms_energy"),
        "voiced_ratio": biomarkers.get("voiced_ratio"),
        # Air quality
        "pm25": air_quality.get("pm25"),
        "no2": air_quality.get("no2"),
        "o3": air_quality.get("o3"),
        # Location
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        # Classification
        "top_factors": data.get("top_factors", []),
        "recommendation": data.get("recommendation"),
        "biomarker_comparison": comparison,
    }).execute()

    return assessment_id


def get_user_history(user_id: str, limit: int = 50) -> List[dict]:
    """Get assessment history for a user (newest first)."""
    db = _get_client()
    result = (
        db.table("assessments")
        .select("*")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    history = []
    for a in result.data:
        history.append({
            "id": a["id"],
            "user_id": a["user_id"],
            "timestamp": a["timestamp"],
            "risk_level": a["risk_level"],
            "confidence": a["confidence"],
            "biomarkers": {
                "f0_hz": a.get("f0_hz", 0),
                "jitter_pct": a.get("jitter_pct", 0),
                "shimmer_pct": a.get("shimmer_pct", 0),
                "hnr_db": a.get("hnr_db", 0),
                "duration_s": a.get("duration_s"),
                "rms_energy": a.get("rms_energy"),
                "voiced_ratio": a.get("voiced_ratio"),
            },
            "air_quality": {
                "pm25": a.get("pm25"),
                "no2": a.get("no2"),
                "o3": a.get("o3"),
            },
            "latitude": a.get("latitude"),
            "longitude": a.get("longitude"),
            "top_factors": a.get("top_factors", []),
            "recommendation": a.get("recommendation"),
            "biomarker_comparison": a.get("biomarker_comparison"),
        })
    return history


def get_assessment(assessment_id: str) -> Optional[dict]:
    """Get a specific assessment by ID."""
    db = _get_client()
    result = (
        db.table("assessments")
        .select("*")
        .eq("id", assessment_id)
        .execute()
    )
    if not result.data:
        return None
    a = result.data[0]
    return {
        "id": a["id"],
        "user_id": a["user_id"],
        "timestamp": a["timestamp"],
        "risk_level": a["risk_level"],
        "confidence": a["confidence"],
        "biomarkers": {
            "f0_hz": a.get("f0_hz", 0),
            "jitter_pct": a.get("jitter_pct", 0),
            "shimmer_pct": a.get("shimmer_pct", 0),
            "hnr_db": a.get("hnr_db", 0),
            "duration_s": a.get("duration_s"),
            "rms_energy": a.get("rms_energy"),
            "voiced_ratio": a.get("voiced_ratio"),
        },
        "air_quality": {
            "pm25": a.get("pm25"),
            "no2": a.get("no2"),
            "o3": a.get("o3"),
        },
        "latitude": a.get("latitude"),
        "longitude": a.get("longitude"),
        "top_factors": a.get("top_factors", []),
        "recommendation": a.get("recommendation"),
        "biomarker_comparison": a.get("biomarker_comparison"),
    }