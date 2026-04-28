"""Air quality client using OpenWeatherMap (CAMS atmospheric model).

Data source: https://openweathermap.org/api/air-pollution

OpenWeatherMap serves data from the Copernicus Atmosphere Monitoring Service
(CAMS), the European Union's official atmospheric monitoring program. CAMS
combines satellite observations (Sentinel-5P, TROPOMI), global emission
inventories, and weather data into a physics-based atmospheric model that
estimates pollutant concentrations anywhere on Earth at ~40 km spatial
resolution, updated every 3 hours.

This is preferred over single-station data sources because it provides:
- Complete coverage of the Philippines (not just locations with WAQI stations)
- Consistent, freshly-updated readings (3-hourly)
- All major pollutants (PM2.5, NO2, O3) from the same model run

Concentration values from OpenWeatherMap (in µg/m³) are converted to US-EPA
AQI sub-index values for clinical interpretability and consistency with
international air quality reporting standards.
"""
import httpx
from typing import Dict, Optional, List, Tuple
from app.config import get_settings


OWM_AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


def _empty_result() -> Dict[str, Optional[float]]:
    return {
        "pm25": None,
        "no2": None,
        "o3": None,
        "station_name": None,
        "aqi": None,
    }


def _linear_aqi(c: float, breakpoints: List[Tuple[float, float, int, int]]) -> float:
    """Convert raw concentration to AQI sub-index via linear interpolation.

    breakpoints: list of (conc_lo, conc_hi, aqi_lo, aqi_hi) tuples.
    """
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= c <= c_hi:
            return round(((a_hi - a_lo) / (c_hi - c_lo)) * (c - c_lo) + a_lo, 1)
    # Above all breakpoints — extrapolate using the last one
    c_lo, c_hi, a_lo, a_hi = breakpoints[-1]
    return round(((a_hi - a_lo) / (c_hi - c_lo)) * (c - c_lo) + a_lo, 1)


def _pm25_ugm3_to_aqi(c: float) -> float:
    """US-EPA PM2.5 24h -> AQI breakpoints (2024 update)."""
    return _linear_aqi(c, [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 500.0, 301, 500),
    ])


def _no2_ugm3_to_aqi(c_ugm3: float) -> float:
    """NO2 1h concentration (µg/m³) -> AQI sub-index.

    US-EPA uses ppb; convert µg/m³ to ppb using factor 0.532 for NO2.
    """
    c_ppb = c_ugm3 * 0.532
    return _linear_aqi(c_ppb, [
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
        (1250.0, 2049.0, 301, 500),
    ])


def _o3_ugm3_to_aqi(c_ugm3: float) -> float:
    """O3 8h concentration (µg/m³) -> AQI sub-index.

    Convert µg/m³ to ppb using factor 0.509 for O3.
    """
    c_ppb = c_ugm3 * 0.509
    return _linear_aqi(c_ppb, [
        (0.0, 54.0, 0, 50),
        (55.0, 70.0, 51, 100),
        (71.0, 85.0, 101, 150),
        (86.0, 105.0, 151, 200),
        (106.0, 200.0, 201, 300),
    ])


async def fetch_air_quality(latitude: float, longitude: float) -> Dict[str, Optional[float]]:
    """Fetch air quality from OpenWeatherMap (CAMS atmospheric model).

    Returns PM2.5, NO2, O3 as US-EPA AQI sub-index values.
    Returns all-None if the API is unavailable.
    """
    settings = get_settings()
    token = getattr(settings, "owm_api_token", None) or ""
    if not token:
        return _empty_result()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                OWM_AIR_URL,
                params={"lat": latitude, "lon": longitude, "appid": token},
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("list") or []
            if not items:
                return _empty_result()
            entry = items[0]

            components = entry.get("components", {}) or {}
            owm_aqi = (entry.get("main") or {}).get("aqi")  # 1-5 scale

            result = _empty_result()
            if "pm2_5" in components:
                try:
                    result["pm25"] = _pm25_ugm3_to_aqi(float(components["pm2_5"]))
                except (TypeError, ValueError):
                    pass
            if "no2" in components:
                try:
                    result["no2"] = _no2_ugm3_to_aqi(float(components["no2"]))
                except (TypeError, ValueError):
                    pass
            if "o3" in components:
                try:
                    result["o3"] = _o3_ugm3_to_aqi(float(components["o3"]))
                except (TypeError, ValueError):
                    pass

            result["station_name"] = "OpenWeatherMap (CAMS)"
            result["aqi"] = owm_aqi

            return result

    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return _empty_result()