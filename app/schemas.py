"""Pydantic models for request and response bodies."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    age: int = Field(..., ge=10, le=100)
    gender: str = Field(..., pattern="^(male|female|other)$")
    smoker: bool = False
    has_asthma: bool = False


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


class Biomarkers(BaseModel):
    f0_hz: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float


class AirQuality(BaseModel):
    pm25: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    station_name: Optional[str] = None
    aqi: Optional[int] = None


class AssessmentResponse(BaseModel):
    assessment_id: str
    risk_level: str  # Low, Moderate, High
    confidence: float
    biomarkers: Biomarkers
    air_quality: AirQuality
    top_factors: List[str]
    recommendation: str
    timestamp: datetime


class BaselineResponse(BaseModel):
    baseline_id: str
    biomarkers: Biomarkers
    created_at: datetime


class HistoryItem(BaseModel):
    assessment_id: str
    risk_level: str
    pm25: Optional[float]
    timestamp: datetime
