"""
Assessment endpoint — key fix:
- Builds BiomarkerComparison object so mobile can show baseline vs current
- Silence detection already handled in voice_analysis.py
- Indoor/outdoor adjustment using WHO infiltration factor (0.5)
- Reverse geocoding for city name in history
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

# WHO Indoor Air Quality Guidelines infiltration factor
# Indoor PM2.5 ≈ 50% of outdoor in well-ventilated buildings
INDOOR_INFILTRATION_FACTOR = 0.5


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc_iso(dt) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "tzinfo"):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    return str(dt)


def _adjust_for_indoor(air_quality: dict) -> dict:
    """Apply WHO infiltration factor for indoor recordings."""
    adjusted = air_quality.copy()
    if adjusted.get("pm25") is not None:
        adjusted["pm25"] = round(adjusted["pm25"] * INDOOR_INFILTRATION_FACTOR, 2)
    if adjusted.get("no2") is not None:
        adjusted["no2"] = round(adjusted["no2"] * INDOOR_INFILTRATION_FACTOR, 2)
    if adjusted.get("o3") is not None:
        adjusted["o3"] = round(adjusted["o3"] * INDOOR_INFILTRATION_FACTOR, 2)
    adjusted["location_type"] = "Indoor (adjusted)"
    return adjusted


async def _get_city_name(latitude: float, longitude: float) -> str:
    """Reverse geocode coordinates to city name using OWM API."""
    settings = get_settings()
    token = getattr(settings, "owm_api_token", None) or ""
    if not token:
        return "Unknown Location"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "http://api.openweathermap.org/geo/1.0/reverse",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "limit": 1,
                    "appid": token,
                },
            )
            response.raise_for_status()
            data = response.json()
            if data:
                name = data[0].get("name", "")
                state = data[0].get("state", "")
                if name and state:
                    return f"{name}, {state}"
                return name or "Unknown Location"
    except Exception:
        pass
    return "Unknown Location"


def _build_comparison(
    current: dict,
    baseline: dict | None,
    deltas: dict | None
) -> schemas.BiomarkerComparison:
    """Build BiomarkerComparison for result screen display."""
    if baseline and deltas:
        return schemas.BiomarkerComparison(
            baseline_f0_hz=baseline["f0_hz"],
            baseline_jitter_pct=baseline["jitter_pct"],
            baseline_shimmer_pct=baseline["shimmer_pct"],
            baseline_hnr_db=baseline["hnr_db"],
            current_f0_hz=current["f0_hz"],
            current_jitter_pct=current["jitter_pct"],
            current_shimmer_pct=current["shimmer_pct"],
            current_hnr_db=current["hnr_db"],
            delta_f0=deltas["delta_f0"],
            delta_jitter=deltas["delta_jitter"],
            delta_shimmer=deltas["delta_shimmer"],
            delta_hnr=deltas["delta_hnr"],
            delta_f0_pct=deltas["delta_f0_pct"],
            delta_jitter_pct=deltas["delta_jitter_pct"],
            delta_shimmer_pct=deltas["delta_shimmer_pct"],
            delta_hnr_pct=deltas["delta_hnr_pct"],
            has_baseline=True,
        )
    else:
        return schemas.BiomarkerComparison(
            current_f0_hz=current["f0_hz"],
            current_jitter_pct=current["jitter_pct"],
            current_shimmer_pct=current["shimmer_pct"],
            current_hnr_db=current["hnr_db"],
            has_baseline=False,
        )


# ─── Philippines cities for map ───────────────────────────────────────────────

PH_MAJOR_CITIES = [
    {"name": "Manila",         "lat": 14.5995, "lng": 120.9842, "region": "NCR"},
    {"name": "Quezon City",    "lat": 14.6760, "lng": 121.0437, "region": "NCR"},
    {"name": "Caloocan",       "lat": 14.6488, "lng": 120.9669, "region": "NCR"},
    {"name": "Makati",         "lat": 14.5547, "lng": 121.0244, "region": "NCR"},
    {"name": "Pasig",          "lat": 14.5764, "lng": 121.0851, "region": "NCR"},
    {"name": "Angeles City",   "lat": 15.1450, "lng": 120.5887, "region": "Central Luzon"},
    {"name": "San Jose del Monte","lat": 14.8136,"lng": 121.0453,"region": "Central Luzon"},
    {"name": "Olongapo",       "lat": 14.8294, "lng": 120.2826, "region": "Central Luzon"},
    {"name": "Cabanatuan",     "lat": 15.4917, "lng": 120.9700, "region": "Central Luzon"},
    {"name": "Tarlac",         "lat": 15.4865, "lng": 120.5916, "region": "Central Luzon"},
    {"name": "Baguio",         "lat": 16.4023, "lng": 120.5960, "region": "CAR"},
    {"name": "Dagupan",        "lat": 16.0438, "lng": 120.3331, "region": "Ilocos"},
    {"name": "Vigan",          "lat": 17.5747, "lng": 120.3870, "region": "Ilocos"},
    {"name": "Tuguegarao",     "lat": 17.6132, "lng": 121.7270, "region": "Cagayan Valley"},
    {"name": "Batangas City",  "lat": 13.7565, "lng": 121.0583, "region": "CALABARZON"},
    {"name": "Lucena",         "lat": 13.9311, "lng": 121.6170, "region": "CALABARZON"},
    {"name": "Naga",           "lat": 13.6218, "lng": 123.1948, "region": "Bicol"},
    {"name": "Legazpi",        "lat": 13.1391, "lng": 123.7438, "region": "Bicol"},
    {"name": "Iloilo City",    "lat": 10.7202, "lng": 122.5621, "region": "Western Visayas"},
    {"name": "Bacolod",        "lat": 10.6713, "lng": 122.9511, "region": "Western Visayas"},
    {"name": "Cebu City",      "lat": 10.3157, "lng": 123.8854, "region": "Central Visayas"},
    {"name": "Tacloban",       "lat": 11.2444, "lng": 125.0029, "region": "Eastern Visayas"},
    {"name": "Tagbilaran",     "lat":  9.6417, "lng": 123.8563, "region": "Central Visayas"},
    {"name": "Cagayan de Oro", "lat":  8.4542, "lng": 124.6319, "region": "Northern Mindanao"},
    {"name": "Davao City",     "lat":  7.1907, "lng": 125.4553, "region": "Davao"},
    {"name": "General Santos", "lat":  6.1164, "lng": 125.1716, "region": "SOCCSKSARGEN"},
    {"name": "Zamboanga",      "lat":  6.9214, "lng": 122.0790, "region": "Zamboanga Peninsula"},
    {"name": "Butuan",         "lat":  8.9475, "lng": 125.5406, "region": "Caraga"},
]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/assess", response_model=schemas.AssessmentResponse)
async def create_assessment(
    audio: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    is_indoor: bool = Form(False),
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
        baseline_biomarkers = None
        deltas = {"delta_f0": 0, "delta_jitter": 0,
                  "delta_shimmer": 0, "delta_hnr": 0,
                  "delta_f0_pct": 0, "delta_jitter_pct": 0,
                  "delta_shimmer_pct": 0, "delta_hnr_pct": 0}

    air_quality = await fetch_air_quality(latitude, longitude)
    if is_indoor:
        air_quality = _adjust_for_indoor(air_quality)

    demographics = {
        "age": user.get("age", 30),
        "gender": user.get("gender", "other"),
    }

    risk, confidence, top_factors = classify_risk(
        biomarkers, deltas, air_quality, demographics
    )
    recommendation = build_recommendation(risk, air_quality, top_factors)
    comparison = _build_comparison(
        biomarkers, baseline_biomarkers, deltas if baseline else None
    )

    data = {
        "biomarkers": biomarkers,
        "air_quality": air_quality,
        "latitude": latitude,
        "longitude": longitude,
        "risk_level": risk,
        "confidence": confidence,
        "top_factors": top_factors,
        "recommendation": recommendation,
        "biomarker_comparison": comparison.model_dump(),
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
        biomarker_comparison=comparison,
    )


@router.get("/history")
async def get_history(
    limit: int = 30,
    user: dict = Depends(get_current_user)
):
    history = database.get_user_history(user["id"], limit=limit)

    async def _enrich(a):
        lat = a.get("latitude")
        lng = a.get("longitude")
        city = "Unknown Location"
        if lat is not None and lng is not None:
            city = await _get_city_name(lat, lng)
        return {
            "assessment_id": a["id"],
            "risk_level": a["risk_level"],
            "pm25": a["air_quality"].get("pm25"),
            "no2": a["air_quality"].get("no2"),
            "o3": a["air_quality"].get("o3"),
            "biomarkers": a["biomarkers"],
            "timestamp": _to_utc_iso(a["timestamp"]),
            "latitude": lat,
            "longitude": lng,
            "city": city,
        }

    results = await asyncio.gather(*[_enrich(a) for a in history])
    return list(results)


@router.get("/assessments/{assessment_id}",
            response_model=schemas.AssessmentResponse)
def get_assessment_detail(
    assessment_id: str,
    user: dict = Depends(get_current_user),
):
    a = database.get_assessment(assessment_id)
    if not a or a.get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail="Assessment not found")
    ts = a["timestamp"]
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    comparison_data = a.get("biomarker_comparison")
    comparison = schemas.BiomarkerComparison(**comparison_data) \
        if comparison_data else None

    return schemas.AssessmentResponse(
        assessment_id=a["id"],
        risk_level=a["risk_level"],
        confidence=a["confidence"],
        biomarkers=schemas.Biomarkers(**a["biomarkers"]),
        air_quality=schemas.AirQuality(**a["air_quality"]),
        top_factors=a["top_factors"],
        recommendation=a["recommendation"],
        timestamp=ts,
        biomarker_comparison=comparison,
    )


def _classify_aqi_level(pm25) -> str:
    if pm25 is None:
        return "Unknown"
    p = float(pm25)
    if p <= 50:
        return "Low"
    if p <= 100:
        return "Moderate"
    return "High"


@router.get("/air-quality/map")
async def get_air_quality_map(user: dict = Depends(get_current_user)):
    tasks = [fetch_air_quality(c["lat"], c["lng"]) for c in PH_MAJOR_CITIES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    locations = []
    for city, aq in zip(PH_MAJOR_CITIES, results):
        if isinstance(aq, Exception):
            aq = {"pm25": None, "no2": None, "o3": None}
        pm25 = aq.get("pm25")
        locations.append({
            "name": city["name"],
            "region": city["region"],
            "latitude": city["lat"],
            "longitude": city["lng"],
            "pm25": pm25,
            "no2": aq.get("no2"),
            "o3": aq.get("o3"),
            "category": _classify_aqi_level(pm25),
        })
    return {"locations": locations, "timestamp": _to_utc_iso(_utc_now())}


def _pm25_to_aqi(c: float) -> float:
    bp = [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300), (225.5, 500.0, 301, 500),
    ]
    for c_lo, c_hi, a_lo, a_hi in bp:
        if c_lo <= c <= c_hi:
            return round(((a_hi - a_lo) / (c_hi - c_lo)) * (c - c_lo) + a_lo, 1)
    return 500.0


@router.get("/air-quality/forecast")
async def get_air_quality_forecast(
    latitude: float,
    longitude: float,
    user: dict = Depends(get_current_user),
):
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
        raise HTTPException(status_code=502, detail=f"Forecast error: {e}")

    items = data.get("list") or []
    hourly = []
    for item in items[:48]:
        components = item.get("components", {}) or {}
        pm25_ugm3 = components.get("pm2_5")
        pm25_aqi = _pm25_to_aqi(float(pm25_ugm3)) if pm25_ugm3 is not None else None
        ts = item.get("dt", 0)
        if ts:
            from datetime import datetime as dt_
            iso = dt_.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            iso = None
        hourly.append({
            "timestamp": iso,
            "pm25": pm25_aqi,
            "category": _classify_aqi_level(pm25_aqi),
        })

    from collections import defaultdict
    days = defaultdict(list)
    for h in hourly:
        if h["timestamp"] and h["pm25"] is not None:
            try:
                from datetime import datetime as dt_
                ph_ts = dt_.fromisoformat(h["timestamp"]).timestamp() + 8 * 3600
                day_key = dt_.utcfromtimestamp(ph_ts).date().isoformat()
            except Exception:
                day_key = h["timestamp"][:10]
            days[day_key].append(h["pm25"])

    daily = []
    for date_key, values in sorted(days.items())[:3]:
        avg = sum(values) / len(values)
        daily.append({
            "date": date_key,
            "avg_pm25": round(avg, 1),
            "max_pm25": round(max(values), 1),
            "category": _classify_aqi_level(avg),
        })

    return {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": hourly,
        "daily": daily,
        "generated_at": _to_utc_iso(_utc_now()),
    }