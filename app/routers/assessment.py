"""Voice assessment endpoints with describe/predict capabilities.

All timestamps are returned in ISO 8601 with explicit UTC offset ("+00:00")
so the Flutter client can correctly convert them to the user's local time.
"""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from datetime import datetime, timezone
import asyncio
import httpx

from app import database, schemas
from app.config import get_settings
from app.services.auth import get_current_user
from app.services.voice_analysis import extract_biomarkers, compute_deltas
from app.services.air_quality import fetch_air_quality
from app.services.classifier import classify_risk, build_recommendation

router = APIRouter(tags=["assessment"])


def _utc_now() -> datetime:
    """Current UTC time with explicit timezone marker."""
    return datetime.now(timezone.utc)


def _to_utc_iso(dt) -> str:
    """Convert a datetime (naive or aware) to ISO 8601 string with UTC offset.

    Flutter's DateTime.parse() needs a timezone indicator ("Z" or "+00:00") to
    correctly identify a string as UTC; otherwise it treats it as local time.
    """
    if dt is None:
        return ""
    if hasattr(dt, "tzinfo"):
        if dt.tzinfo is None:
            # Naive datetime — assume it was UTC
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    return str(dt)


PH_MAJOR_CITIES = [
    # NCR
    {"name": "Manila",        "lat": 14.5995, "lng": 120.9842, "region": "NCR"},
    {"name": "Quezon City",   "lat": 14.6760, "lng": 121.0437, "region": "NCR"},
    {"name": "Caloocan",      "lat": 14.6488, "lng": 120.9669, "region": "NCR"},
    {"name": "Makati",        "lat": 14.5547, "lng": 121.0244, "region": "NCR"},
    {"name": "Pasig",         "lat": 14.5764, "lng": 121.0851, "region": "NCR"},
    # Central Luzon
    {"name": "Angeles City",  "lat": 15.1450, "lng": 120.5887, "region": "Central Luzon"},
    {"name": "San Jose del Monte", "lat": 14.8136, "lng": 121.0453, "region": "Central Luzon"},
    {"name": "Olongapo",      "lat": 14.8294, "lng": 120.2826, "region": "Central Luzon"},
    {"name": "Cabanatuan",    "lat": 15.4917, "lng": 120.9700, "region": "Central Luzon"},
    {"name": "Tarlac",        "lat": 15.4865, "lng": 120.5916, "region": "Central Luzon"},
    # Northern Luzon
    {"name": "Baguio",        "lat": 16.4023, "lng": 120.5960, "region": "CAR"},
    {"name": "Dagupan",       "lat": 16.0438, "lng": 120.3331, "region": "Ilocos"},
    {"name": "Vigan",         "lat": 17.5747, "lng": 120.3870, "region": "Ilocos"},
    {"name": "Tuguegarao",    "lat": 17.6132, "lng": 121.7270, "region": "Cagayan Valley"},
    # CALABARZON + Bicol
    {"name": "Batangas City", "lat": 13.7565, "lng": 121.0583, "region": "CALABARZON"},
    {"name": "Lucena",        "lat": 13.9311, "lng": 121.6170, "region": "CALABARZON"},
    {"name": "Naga",          "lat": 13.6218, "lng": 123.1948, "region": "Bicol"},
    {"name": "Legazpi",       "lat": 13.1391, "lng": 123.7438, "region": "Bicol"},
    # Visayas
    {"name": "Iloilo City",   "lat": 10.7202, "lng": 122.5621, "region": "Western Visayas"},
    {"name": "Bacolod",       "lat": 10.6713, "lng": 122.9511, "region": "Western Visayas"},
    {"name": "Cebu City",     "lat": 10.3157, "lng": 123.8854, "region": "Central Visayas"},
    {"name": "Tacloban",      "lat": 11.2444, "lng": 125.0029, "region": "Eastern Visayas"},
    {"name": "Tagbilaran",    "lat":  9.6417, "lng": 123.8563, "region": "Central Visayas"},
    # Mindanao
    {"name": "Cagayan de Oro","lat":  8.4542, "lng": 124.6319, "region": "Northern Mindanao"},
    {"name": "Davao City",    "lat":  7.1907, "lng": 125.4553, "region": "Davao"},
    {"name": "General Santos","lat":  6.1164, "lng": 125.1716, "region": "SOCCSKSARGEN"},
    {"name": "Zamboanga",     "lat":  6.9214, "lng": 122.0790, "region": "Zamboanga Peninsula"},
    {"name": "Butuan",        "lat":  8.9475, "lng": 125.5406, "region": "Caraga"},
]


@router.post("/assess", response_model=schemas.AssessmentResponse)
async def create_assessment(
    audio: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    user: dict = Depends(get_current_user),
):
    audio_bytes = await audio.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio file too small")

    try:
        biomarkers = extract_biomarkers(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")

    baseline = database.get_baseline(user["id"])
    if baseline:
        baseline_biomarkers = {
            "f0_hz": baseline["f0_hz"],
            "jitter_pct": baseline["jitter_pct"],
            "shimmer_pct": baseline["shimmer_pct"],
            "hnr_db": baseline["hnr_db"],
        }
        deltas = compute_deltas(biomarkers, baseline_biomarkers)
    else:
        deltas = {"delta_f0": 0, "delta_jitter": 0, "delta_shimmer": 0, "delta_hnr": 0}

    air_quality = await fetch_air_quality(latitude, longitude)

    demographics = {
        "age": user["age"],
        "gender": user["gender"],
        "smoker": user["smoker"],
        "has_asthma": user["has_asthma"],
    }
    risk, confidence, top_factors = classify_risk(
        biomarkers, deltas, air_quality, demographics
    )

    recommendation = build_recommendation(risk, air_quality, top_factors)

    data = {
        "biomarkers": biomarkers,
        "air_quality": air_quality,
        "latitude": latitude,
        "longitude": longitude,
        "risk_level": risk,
        "confidence": confidence,
        "top_factors": top_factors,
        "recommendation": recommendation,
    }
    assessment_id = database.save_assessment(user["id"], data)

    return schemas.AssessmentResponse(
        assessment_id=assessment_id,
        risk_level=risk,
        confidence=confidence,
        biomarkers=schemas.Biomarkers(**biomarkers),
        air_quality=schemas.AirQuality(**air_quality),
        top_factors=top_factors,
        recommendation=recommendation,
        timestamp=_utc_now(),
    )


@router.get("/history")
def get_history(limit: int = 30, user: dict = Depends(get_current_user)):
    history = database.get_user_history(user["id"], limit=limit)
    return [
        {
            "assessment_id": a["id"],
            "risk_level": a["risk_level"],
            "pm25": a["air_quality"].get("pm25"),
            "no2": a["air_quality"].get("no2"),
            "o3": a["air_quality"].get("o3"),
            "biomarkers": a["biomarkers"],
            # Always include explicit UTC offset so Flutter knows to convert
            "timestamp": _to_utc_iso(a["timestamp"]),
        }
        for a in history
    ]


@router.get("/assessments/{assessment_id}", response_model=schemas.AssessmentResponse)
def get_assessment_detail(
    assessment_id: str,
    user: dict = Depends(get_current_user),
):
    a = database.get_assessment(assessment_id)
    if not a or a.get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail="Assessment not found")
    ts = a["timestamp"]
    # Ensure timezone-aware UTC
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return schemas.AssessmentResponse(
        assessment_id=a["id"],
        risk_level=a["risk_level"],
        confidence=a["confidence"],
        biomarkers=schemas.Biomarkers(**a["biomarkers"]),
        air_quality=schemas.AirQuality(**a["air_quality"]),
        top_factors=a["top_factors"],
        recommendation=a["recommendation"],
        timestamp=ts,
    )


def _classify_aqi_level(pm25) -> str:
    if pm25 is None:
        return "Unknown"
    p = float(pm25)
    if p <= 50:
        return "Good"
    if p <= 100:
        return "Moderate"
    if p <= 150:
        return "Unhealthy for Sensitive"
    if p <= 200:
        return "Unhealthy"
    if p <= 300:
        return "Very Unhealthy"
    return "Hazardous"


@router.get("/air-quality/map")
async def get_air_quality_map(user: dict = Depends(get_current_user)):
    tasks = [
        fetch_air_quality(city["lat"], city["lng"])
        for city in PH_MAJOR_CITIES
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    locations = []
    for city, aq in zip(PH_MAJOR_CITIES, results):
        if isinstance(aq, Exception):
            aq = {"pm25": None, "no2": None, "o3": None}

        pm25 = aq.get("pm25")
        category = _classify_aqi_level(pm25)

        locations.append({
            "name": city["name"],
            "region": city["region"],
            "latitude": city["lat"],
            "longitude": city["lng"],
            "pm25": pm25,
            "no2": aq.get("no2"),
            "o3": aq.get("o3"),
            "category": category,
        })

    return {"locations": locations, "timestamp": _to_utc_iso(_utc_now())}


def _pm25_ugm3_to_aqi(c: float) -> float:
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.0, 301, 500),
    ]
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= c <= c_hi:
            return round(((a_hi - a_lo) / (c_hi - c_lo)) * (c - c_lo) + a_lo, 1)
    c_lo, c_hi, a_lo, a_hi = breakpoints[-1]
    return round(((a_hi - a_lo) / (c_hi - c_lo)) * (c - c_lo) + a_lo, 1)


@router.get("/air-quality/forecast")
async def get_air_quality_forecast(
    latitude: float,
    longitude: float,
    user: dict = Depends(get_current_user),
):
    """Hourly air quality forecast for the next ~48 hours.

    All timestamps returned with explicit UTC offset so Flutter can
    convert them to local time correctly.
    """
    settings = get_settings()
    token = getattr(settings, "owm_api_token", None) or ""
    if not token:
        raise HTTPException(status_code=500, detail="Forecast service not configured")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
                params={"lat": latitude, "lon": longitude, "appid": token},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Forecast service error: {e}")

    items = data.get("list") or []

    hourly = []
    for item in items[:48]:
        components = item.get("components", {}) or {}
        pm25_ugm3 = components.get("pm2_5")
        pm25_aqi = _pm25_ugm3_to_aqi(float(pm25_ugm3)) if pm25_ugm3 is not None else None
        ts = item.get("dt", 0)
        if ts:
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            iso = dt_utc.isoformat()
        else:
            iso = None
        hourly.append({
            "timestamp": iso,
            "pm25": pm25_aqi,
            "category": _classify_aqi_level(pm25_aqi),
        })

    daily_summary = []
    if hourly:
        from collections import defaultdict
        days = defaultdict(list)
        for h in hourly:
            if h["timestamp"] and h["pm25"] is not None:
                # Convert to PHT for day grouping (UTC+8)
                # Parse the UTC ISO and add 8 hours to get PH date
                try:
                    dt_utc = datetime.fromisoformat(h["timestamp"])
                    # Naive shift to PH local for grouping by date
                    ph_dt = dt_utc.astimezone(timezone.utc)
                    ph_date = (ph_dt.timestamp() + 8 * 3600)
                    day_key = datetime.utcfromtimestamp(ph_date).date().isoformat()
                except Exception:
                    day_key = h["timestamp"][:10]
                days[day_key].append(h["pm25"])
        for date_key, values in sorted(days.items())[:3]:
            avg = sum(values) / len(values)
            mx = max(values)
            daily_summary.append({
                "date": date_key,
                "avg_pm25": round(avg, 1),
                "max_pm25": round(mx, 1),
                "category": _classify_aqi_level(avg),
            })

    return {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": hourly,
        "daily": daily_summary,
        "generated_at": _to_utc_iso(_utc_now()),
    }