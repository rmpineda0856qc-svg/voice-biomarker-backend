"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UserRegister(BaseModel):
    email: str
    password: str
    age: int
    gender: str
    smoker: bool
    has_asthma: bool


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    
Token = TokenResponse


class Biomarkers(BaseModel):
    f0_hz: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    duration_s: Optional[float] = None
    rms_energy: Optional[float] = None
    voiced_ratio: Optional[float] = None


class BaselineBiomarkers(BaseModel):
    f0_hz: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float


class BiomarkerDeltas(BaseModel):
    """Absolute and percentage changes from baseline."""
    # Absolute deltas
    delta_f0: float
    delta_jitter: float
    delta_shimmer: float
    delta_hnr: float
    # Percentage changes
    delta_f0_pct: float
    delta_jitter_pct: float
    delta_shimmer_pct: float
    delta_hnr_pct: float


class BiomarkerComparison(BaseModel):
    """
    Side-by-side comparison of baseline vs current biomarkers.
    Sent to mobile app for display in result screen.
    """
    # Baseline values
    baseline_f0_hz: Optional[float] = None
    baseline_jitter_pct: Optional[float] = None
    baseline_shimmer_pct: Optional[float] = None
    baseline_hnr_db: Optional[float] = None

    # Current values
    current_f0_hz: float
    current_jitter_pct: float
    current_shimmer_pct: float
    current_hnr_db: float

    # Deltas (None if no baseline exists)
    delta_f0: Optional[float] = None
    delta_jitter: Optional[float] = None
    delta_shimmer: Optional[float] = None
    delta_hnr: Optional[float] = None

    # Percentage changes
    delta_f0_pct: Optional[float] = None
    delta_jitter_pct: Optional[float] = None
    delta_shimmer_pct: Optional[float] = None
    delta_hnr_pct: Optional[float] = None

    # Whether baseline exists
    has_baseline: bool = False


class AirQuality(BaseModel):
    pm25: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    station: Optional[str] = None
    aqi: Optional[float] = None


class AssessmentResponse(BaseModel):
    assessment_id: str
    risk_level: str
    confidence: float
    biomarkers: Biomarkers
    air_quality: AirQuality
    top_factors: List[str]
    recommendation: str
    timestamp: datetime

    # NEW — baseline comparison for result screen display
    biomarker_comparison: Optional[BiomarkerComparison] = None
