"""Diagnostic: show what pollutants each PH station has."""
import asyncio
import sys
sys.path.insert(0, ".")

from app.services.air_quality import (
    _search_ph_stations,
    _fetch_station_detail,
    _haversine_km,
)
from app.config import get_settings
import httpx


async def main():
    token = get_settings().waqi_api_token
    user_lat, user_lng = 15.15, 120.59

    async with httpx.AsyncClient(timeout=20.0) as client:
        print("Searching PH stations...")
        stations = await _search_ph_stations(client, token)
        print(f"Found {len(stations)} stations.\n")

        # Sort by distance
        stations.sort(key=lambda s: _haversine_km(user_lat, user_lng, s["lat"], s["lng"]))

        for station in stations:
            dist = _haversine_km(user_lat, user_lng, station["lat"], station["lng"])
            detail = await _fetch_station_detail(client, int(station["uid"]), token)
            if detail is None:
                print(f"  [DEAD] UID={station['uid']} {station['name']} ({dist:.1f} km)")
                continue
            iaqi = detail.get("iaqi", {})
            pollutants = [k for k in ["pm25", "no2", "o3", "pm10", "so2", "co"] if k in iaqi]
            time_iso = detail.get("time", {}).get("iso", "?")
            print(f"  [OK]   UID={station['uid']} {station['name']} ({dist:.1f} km)")
            print(f"         Pollutants: {pollutants}")
            print(f"         Last reading: {time_iso}")
            for key in ["pm25", "no2", "o3"]:
                if key in iaqi:
                    print(f"         {key}: {iaqi[key].get('v')}")


if __name__ == "__main__":
    asyncio.run(main())