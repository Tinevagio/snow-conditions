"""
openmeteo.py
------------
Fetcher Open-Meteo : historique 5 jours + forecast 2 jours.
Retourne une liste de HourlyWeather prete pour snow_model.py.
API gratuite, sans cle. Doc : https://open-meteo.com/en/docs
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import List

try:
    from snow_model import HourlyWeather
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class HourlyWeather:
        hour: int
        temperature_2m: float
        reference_elevation: float
        wind_speed: float
        shortwave_radiation: float
        snowfall_last_24h: float
        snowfall_last_72h: float
        hours_above_zero_last_48h: int
        hours_below_minus2_last_12h: int

OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m", "windspeed_10m", "shortwave_radiation", "snowfall"]
PAST_DAYS = 25
FORECAST_DAYS = 2


def fetch_raw(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(VARIABLES),
        "windspeed_unit": "kmh",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "UTC",
    }
    url = OPENMETEO_BASE_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"Open-Meteo inaccessible : {e}") from e
    if "error" in data:
        raise RuntimeError(f"Open-Meteo erreur : {data.get('reason')}")
    return data


def _safe(series, idx, default=0.0):
    if idx < 0 or idx >= len(series): return default
    val = series[idx]
    return float(val) if val is not None else default


def _rolling_stats(temp_series, idx):
    w48 = temp_series[max(0, idx-48):idx]
    w12 = temp_series[max(0, idx-12):idx]
    return (
        sum(1 for t in w48 if t is not None and t > 0),
        sum(1 for t in w12 if t is not None and t < -2)
    )


def _snow_cumuls(snow_series, idx):
    s24 = sum(s for s in snow_series[max(0, idx-24):idx] if s is not None)
    s72 = sum(s for s in snow_series[max(0, idx-72):idx] if s is not None)
    return round(s24, 1), round(s72, 1)


def get_hourly_weather(lat, lon, target_date=None) -> List[HourlyWeather]:
    """
    Point d'entree principal.

    Usage:
        today   = datetime.now(timezone.utc)
        weather = get_hourly_weather(45.92, 6.87, target_date=today)
        # passe weather a compute_snow_conditions(grid, weather, month, day)

        tomorrow = today + timedelta(days=1)
        weather  = get_hourly_weather(45.92, 6.87, target_date=tomorrow)
    """
    raw    = fetch_raw(lat, lon)
    hourly = raw["hourly"]
    times  = hourly["time"]
    temps  = hourly["temperature_2m"]
    snow   = hourly["snowfall"]
    elev   = raw.get("elevation", 0.0)
    results = []

    for idx, time_str in enumerate(times):
        dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
        if target_date is not None and dt.date() != target_date:
            continue
        above_zero, below_minus2 = _rolling_stats(temps, idx)
        s24, s72 = _snow_cumuls(snow, idx)
        results.append(HourlyWeather(
            hour                        = dt.hour,
            temperature_2m              = _safe(temps, idx),
            reference_elevation         = elev,
            wind_speed                  = _safe(hourly["windspeed_10m"], idx),
            shortwave_radiation         = _safe(hourly["shortwave_radiation"], idx),
            snowfall_last_24h           = s24,
            snowfall_last_72h           = s72,
            hours_above_zero_last_48h   = above_zero,
            hours_below_minus2_last_12h = below_minus2,
        ))
    return results


if __name__ == "__main__":
    today = datetime.now(timezone.utc)
    try:
        series = get_hourly_weather(45.92, 6.87, target_date=today)
        print(f"{len(series)} heures pour {today.strftime('%d/%m/%Y')}")
        print(f"{'H':>3} | {'Temp':>6} | {'Vent':>6} | {'Soleil':>8} | {'S24':>5} | {'S72':>5} | {'>0/48h':>6} | {'<-2/12h':>7}")
        print("-" * 60)
        for w in series:
            print(f"{w.hour:02d}h | {w.temperature_2m:+5.1f}C | "
                  f"{w.wind_speed:4.0f}kmh | {w.shortwave_radiation:6.0f}W/m2 | "
                  f"{w.snowfall_last_24h:3.1f}cm | {w.snowfall_last_72h:3.1f}cm | "
                  f"{w.hours_above_zero_last_48h:5d}h | {w.hours_below_minus2_last_12h:6d}h")
    except RuntimeError as e:
        print(f"Erreur : {e}")