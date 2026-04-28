"""Simple in-memory data store for the starter project.

NOTE: For production, replace with a proper database (PostgreSQL via SQLAlchemy,
or SQLite for a lighter option). This in-memory store is intentionally minimal
so you can get the app running quickly and then migrate when ready.
"""
from typing import Dict, List, Optional
from datetime import datetime
import uuid


# User table: {user_id: {email, password_hash, age, gender, smoker, has_asthma}}
_users: Dict[str, dict] = {}
# Email index for login lookup: {email: user_id}
_email_to_id: Dict[str, str] = {}
# Baselines: {user_id: {f0, jitter, shimmer, hnr, created_at}}
_baselines: Dict[str, dict] = {}
# Assessments: {assessment_id: {...}}
_assessments: Dict[str, dict] = {}
# User -> list of assessment_ids (newest first)
_user_assessments: Dict[str, List[str]] = {}


def create_user(email: str, password_hash: str, age: int, gender: str,
                smoker: bool, has_asthma: bool) -> str:
    if email in _email_to_id:
        raise ValueError("Email already registered")
    user_id = str(uuid.uuid4())
    _users[user_id] = {
        "email": email,
        "password_hash": password_hash,
        "age": age,
        "gender": gender,
        "smoker": smoker,
        "has_asthma": has_asthma,
        "created_at": datetime.utcnow(),
    }
    _email_to_id[email] = user_id
    _user_assessments[user_id] = []
    return user_id


def get_user_by_email(email: str) -> Optional[dict]:
    user_id = _email_to_id.get(email)
    if not user_id:
        return None
    user = _users[user_id].copy()
    user["id"] = user_id
    return user


def get_user_by_id(user_id: str) -> Optional[dict]:
    user = _users.get(user_id)
    if not user:
        return None
    result = user.copy()
    result["id"] = user_id
    return result


def save_baseline(user_id: str, biomarkers: dict) -> str:
    baseline_id = str(uuid.uuid4())
    _baselines[user_id] = {
        "id": baseline_id,
        **biomarkers,
        "created_at": datetime.utcnow(),
    }
    return baseline_id


def get_baseline(user_id: str) -> Optional[dict]:
    return _baselines.get(user_id)


def save_assessment(user_id: str, data: dict) -> str:
    assessment_id = str(uuid.uuid4())
    _assessments[assessment_id] = {
        "id": assessment_id,
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        **data,
    }
    if user_id not in _user_assessments:
        _user_assessments[user_id] = []
    _user_assessments[user_id].insert(0, assessment_id)
    return assessment_id


def get_user_history(user_id: str, limit: int = 50) -> List[dict]:
    ids = _user_assessments.get(user_id, [])[:limit]
    return [_assessments[aid] for aid in ids if aid in _assessments]


def get_assessment(assessment_id: str) -> Optional[dict]:
    return _assessments.get(assessment_id)
