"""
api/main.py
-----------
API REST FastAPI exposant le moteur de conditions de neige.

Endpoints :
  GET  /health                        → statut de l'API
  GET  /conditions?bbox=...&date=...  → conditions par versant, heure par heure
  GET  /conditions/point?lat=&lon=&date= → conditions pour un point unique
  GET  /best-window?bbox=...&date=... → meilleure fenêtre de poudre

Lancer en local :
  uvicorn api.main:app --reload

Ou depuis la racine du projet :
  uvicorn main:app --reload --app-dir api/
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, date
from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Imports internes — chemins relatifs au projet
# ---------------------------------------------------------------------------


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.snow_model import (
    GridPoint,
    SnowResult,
    compute_snow_conditions,
    compute_surface_temperature,
    classify_snow_condition
)
from core.solar_radiation import best_powder_window
from data.fetchers.openmeteo import get_hourly_weather
from core.terrain import get_terrain_grid, get_terrain_data, TerrainPoint

from datetime import date

def parse_date(date_str: str) -> date:
    """Parse une date au format YYYY-MM-DD"""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Date invalide : '{date_str}'. Format attendu : YYYY-MM-DD")
        
# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Snow Conditions API",
    description="Prédiction des conditions de neige par versant, heure par heure.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # à restreindre en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schémas de réponse
# ---------------------------------------------------------------------------

class SnowConditionEnum(str, Enum):
    POWDER_COLD   = "POWDER_COLD"
    POWDER_WARM   = "POWDER_WARM"
    SPRING_SNOW   = "SPRING_SNOW"
    CRUST_MORNING = "CRUST_MORNING"
    WET_HEAVY     = "WET_HEAVY"
    WIND_AFFECTED = "WIND_AFFECTED"
    OLD_PACKED    = "OLD_PACKED"
    UNDEFINED     = "UNDEFINED"

CONDITION_META = {
    "POWDER_COLD":   {"label": "Poudre froide",      "color": "#4A90D9"},
    "POWDER_WARM":   {"label": "Poudre réchauffée",  "color": "#7FB3E8"},
    "SPRING_SNOW":   {"label": "Neige de printemps", "color": "#F5A623"},
    "CRUST_MORNING": {"label": "Croûte de regel",    "color": "#D0021B"},
    "WET_HEAVY":     {"label": "Neige humide lourde","color": "#8B572A"},
    "WIND_AFFECTED": {"label": "Neige soufflée",     "color": "#9B9B9B"},
    "OLD_PACKED":    {"label": "Neige ancienne",     "color": "#B8D4F0"},
    "UNDEFINED":     {"label": "Indéterminé",        "color": "#EEEEEE"},
}

class HourlyCondition(BaseModel):
    hour: int                  = Field(..., description="Heure UTC (0-23)")
    condition: str             = Field(..., description="Code condition")
    label: str                 = Field(..., description="Libellé lisible")
    color: str                 = Field(..., description="Couleur hex pour la carte")
    temp_surface: float        = Field(..., description="Température de surface estimée (°C)")
    wind_speed: float          = Field(..., description="Vitesse du vent (km/h)")

class PointConditions(BaseModel):
    lat: float
    lon: float
    elevation_m: float
    aspect_deg: float
    aspect_label: str
    slope_deg: float
    hours: List[HourlyCondition]

class ConditionsResponse(BaseModel):
    date: str
    bbox: List[float]          = Field(..., description="[lat_min, lon_min, lat_max, lon_max]")
    resolution_m: float
    generated_at: str
    points: List[PointConditions]

class PowderWindow(BaseModel):
    lat: float
    lon: float
    aspect_label: str
    elevation_m: float
    powder_until_hour: Optional[int]
    message: str

class BestWindowResponse(BaseModel):
    date: str
    bbox: List[float]
    best_north_facing: List[PowderWindow]

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: str) -> tuple:
    """
    Parse une bbox 'lat_min,lon_min,lat_max,lon_max' en tuple de floats.
    Lève HTTPException si invalide.
    """
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        lat_min, lon_min, lat_max, lon_max = parts
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError("lat_min doit être < lat_max, idem pour lon")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError("Latitudes hors plage [-90, 90]")
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError("Longitudes hors plage [-180, 180]")
        return lat_min, lon_min, lat_max, lon_max
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"bbox invalide (attendu 'lat_min,lon_min,lat_max,lon_max') : {e}"
        )

def _result_to_hourly(result: SnowResult) -> HourlyCondition:
    condition_key = result.condition.name if hasattr(result.condition, "name") else str(result.condition)
    meta = CONDITION_META.get(condition_key, CONDITION_META["UNDEFINED"])
    return HourlyCondition(
        hour=result.hour,
        condition=condition_key,
        label=meta["label"],
        color=meta["color"],
        temp_surface=round(result.temp_surface, 1),
        wind_speed=round(result.wind_speed, 1),
    )

def _group_results_by_point(
    grid: List[GridPoint],
    terrain_points: List[TerrainPoint],
    results: List[SnowResult],
) -> List[PointConditions]:
    """
    Regroupe les résultats heure par heure par point de grille.
    """
    # Index résultats par (lat, lon)
    from collections import defaultdict
    by_point: dict = defaultdict(list)
    for r in results:
        by_point[(round(r.lat, 6), round(r.lon, 6))].append(r)

    output = []
    terrain_map = {(round(t.lat, 6), round(t.lon, 6)): t for t in terrain_points}

    for gp in grid:
        key = (round(gp.lat, 6), round(gp.lon, 6))
        terrain = terrain_map.get(key)
        hours_data = sorted(by_point.get(key, []), key=lambda r: r.hour)

        output.append(PointConditions(
            lat=gp.lat,
            lon=gp.lon,
            elevation_m=terrain.elevation_m if terrain else gp.elevation,
            aspect_deg=terrain.aspect_deg if terrain else gp.aspect,
            aspect_label=terrain.aspect_label() if terrain else "?",
            slope_deg=terrain.slope_deg if terrain else gp.slope,
            hours=[_result_to_hourly(r) for r in hours_data],
        ))

    return output

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Système"])
def health():
    """Statut de l'API."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/conditions", response_model=ConditionsResponse, tags=["Conditions"])
def get_conditions(
    bbox: str = Query(
        ...,
        description="Zone géographique : lat_min,lon_min,lat_max,lon_max",
        example="45.90,6.85,45.95,6.95",
    ),
    date: Optional[str] = Query(
        None,
        description="Date au format YYYY-MM-DD (défaut : aujourd'hui UTC)",
        example="2024-02-15",
    ),
    resolution_m: float = Query(
        500,
        ge=100,
        le=2000,
        description="Résolution de la grille en mètres (100–2000)",
    ),
    tiff_path: Optional[str] = Query(
        None,
        description="Chemin vers un GeoTIFF IGN local (optionnel)",
    ),
):
    """
    Retourne les conditions de neige heure par heure pour chaque point
    de la grille couvrant la bbox.

    **Exemple d'appel :**
    ```
    GET /conditions?bbox=45.90,6.85,45.95,6.95&date=2024-02-15&resolution_m=500
    ```
    """
    lat_min, lon_min, lat_max, lon_max = _parse_bbox(bbox)

    # Date cible
    if date:
        try:
            target_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="date invalide, format attendu : YYYY-MM-DD")
    else:
        target_dt = datetime.now(timezone.utc)

    # 1. Grille terrain
    try:
        terrain_points = get_terrain_grid(
            lat_min, lon_min, lat_max, lon_max,
            resolution_m=resolution_m,
            tiff_path=tiff_path,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur terrain : {e}")

    if not terrain_points:
        raise HTTPException(status_code=422, detail="Aucun point dans la bbox — vérifier les coordonnées.")

    # 2. GridPoints pour snow_model
    grid = [
        GridPoint(
            lat=p.lat,
            lon=p.lon,
            elevation=p.elevation_m,
            aspect=p.aspect_deg,
            slope=p.slope_deg,
        )
        for p in terrain_points
    ]

    # 3. Météo temps réel (centre de la bbox)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    try:
        weather = get_hourly_weather(center_lat, center_lon, target_date=target_dt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo inaccessible : {e}")

    if not weather:
        raise HTTPException(status_code=502, detail="Aucune donnée météo disponible pour cette date.")

    # 4. Calcul des conditions
    results = compute_snow_conditions(grid, weather, target_dt.month, target_dt.day)

    # 5. Formatage de la réponse
    points_out = _group_results_by_point(grid, terrain_points, results)

    return ConditionsResponse(
        date=target_dt.strftime("%Y-%m-%d"),
        bbox=[lat_min, lon_min, lat_max, lon_max],
        resolution_m=resolution_m,
        generated_at=datetime.now(timezone.utc).isoformat(),
        points=points_out,
    )


@app.get("/conditions/point", response_model=PointConditions, tags=["Conditions"])
def get_conditions_point(
    lat: float = Query(..., ge=-90, le=90, description="Latitude", example=45.9237),
    lon: float = Query(..., ge=-180, le=180, description="Longitude", example=6.8694),
    date: Optional[str] = Query(None, description="Date YYYY-MM-DD (défaut : aujourd'hui)"),
    tiff_path: Optional[str] = Query(None, description="GeoTIFF IGN local (optionnel)"),
):
    """
    Conditions de neige heure par heure pour un point GPS unique.
    Plus rapide que /conditions pour un point précis.
    """
    if date:
        try:
            target_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="date invalide, format attendu : YYYY-MM-DD")
    else:
        target_dt = datetime.now(timezone.utc)

    terrain = get_terrain_data(lat, lon, tiff_path=tiff_path)

    gp = GridPoint(
        lat=lat, lon=lon,
        elevation=terrain.elevation_m,
        aspect=terrain.aspect_deg,
        slope=terrain.slope_deg,
    )

    try:
        weather = get_hourly_weather(lat, lon, target_date=target_dt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo inaccessible : {e}")

    results = compute_snow_conditions([gp], weather, target_dt.month, target_dt.day)
    hours_out = [_result_to_hourly(r) for r in sorted(results, key=lambda r: r.hour)]

    return PointConditions(
        lat=lat,
        lon=lon,
        elevation_m=terrain.elevation_m,
        aspect_deg=terrain.aspect_deg,
        aspect_label=terrain.aspect_label(),
        slope_deg=terrain.slope_deg,
        hours=hours_out,
    )


@app.get("/best-window", response_model=BestWindowResponse, tags=["Conditions"])
def get_best_window(
    bbox: str = Query(
        ...,
        description="Zone géographique : lat_min,lon_min,lat_max,lon_max",
        example="45.90,6.85,45.95,6.95",
    ),
    date: Optional[str] = Query(None, description="Date YYYY-MM-DD (défaut : demain)"),
    resolution_m: float = Query(500, ge=100, le=2000),
):
    """
    Retourne la fenêtre de poudre optimale pour les versants nord de la zone.
    Killer feature pour planifier le départ la veille au soir.

    Répond à la question : **"Ma poudre tient jusqu'à quelle heure ?"**
    """
    lat_min, lon_min, lat_max, lon_max = _parse_bbox(bbox)

    if date:
        try:
            target_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="date invalide")
    else:
        from datetime import timedelta
        target_dt = datetime.now(timezone.utc) + timedelta(days=1)

    terrain_points = get_terrain_grid(lat_min, lon_min, lat_max, lon_max, resolution_m=resolution_m)

    # On filtre sur les versants nord (aspect 315°–360° ou 0°–45°)
    north_points = [p for p in terrain_points if p.is_north_facing()]

    if not north_points:
        raise HTTPException(status_code=404, detail="Aucun versant nord dans cette zone.")

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    try:
        weather = get_hourly_weather(center_lat, center_lon, target_date=target_dt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo inaccessible : {e}")

    windows = []
    for p in north_points:
        # best_powder_window retourne l'heure max où la poudre est encore bonne
        window_hour = best_powder_window(
            lat=p.lat,
            lon=p.lon,
            month=target_dt.month,
            day=target_dt.day,
            aspect=p.aspect_deg,
            slope=p.slope_deg,
        )

        if window_hour is not None:
            msg = f"Poudre optimale jusqu'à {window_hour:02d}h UTC sur ce versant."
        else:
            msg = "Pas de fenêtre poudre identifiée (manteau trop vieux ou conditions défavorables)."

        windows.append(PowderWindow(
            lat=p.lat,
            lon=p.lon,
            aspect_label=p.aspect_label(),
            elevation_m=p.elevation_m,
            powder_until_hour=window_hour,
            message=msg,
        ))

    # Trier par heure de fermeture décroissante (meilleures fenêtres en premier)
    windows.sort(key=lambda w: w.powder_until_hour or -1, reverse=True)

    return BestWindowResponse(
        date=target_dt.strftime("%Y-%m-%d"),
        bbox=[lat_min, lon_min, lat_max, lon_max],
        best_north_facing=windows[:10],  # top 10
    )
    
@app.get("/debug/point")


def debug_point(lat: float, lon: float, date: str = None,
                aspect: float = 0, elevation: float = 1500, slope: float = 15):
    target_date = parse_date(date)
    weather_series = get_hourly_weather(lat, lon, target_date)
    
    result = []
    for w in weather_series:
        point = GridPoint(lat=lat, lon=lon, elevation=elevation, aspect=aspect, slope=slope)
        temp_surface = compute_surface_temperature(point, w, target_date.month, target_date.day)
        condition, _ = classify_snow_condition(point, w, target_date.month, target_date.day)
        
        from core.solar_radiation import effective_radiation
        
        rad = effective_radiation(
        hour_utc=w.hour, lat=lat, lon=lon,
        month=target_date.month, day=target_date.day,
        aspect=aspect, slope=slope, altitude_m=elevation
        )
        
        result.append({
            "hour": w.hour,
            "temp_2m": w.temperature_2m,
            "temp_surface": round(temp_surface, 2),
            "wind_speed": w.wind_speed,
            "snowfall_24h": w.snowfall_last_24h,
            "snowfall_72h": w.snowfall_last_72h,
            "hours_above_zero_48h": w.hours_above_zero_last_48h,
            "hours_below_minus2_12h": w.hours_below_minus2_last_12h,
            "solar_correction": round(temp_surface - w.temperature_2m, 2),
            "condition": condition.name,
            "solar_bonus": round(rad.temperature_correction, 2),  # ← bonus solaire pur
            "altitude_correction": round(-((elevation - w.reference_elevation) * 0.006), 2),
            "solar_correction": round(temp_surface - w.temperature_2m, 2),  # total des deux
        })
    
    return result


@app.get("/debug/compare")
def debug_compare(lat: float, lon: float, date: str = None,
                  elevation: float = 1500, slope: float = 15):
    target_date = parse_date(date)
    weather_series = get_hourly_weather(lat, lon, target_date)

    result = []
    for w in weather_series:
        point_nord = GridPoint(lat=lat, lon=lon, elevation=elevation, aspect=0, slope=slope)
        point_sud  = GridPoint(lat=lat, lon=lon, elevation=elevation, aspect=180, slope=slope)

        temp_nord = compute_surface_temperature(point_nord, w, target_date.month, target_date.day)
        temp_sud  = compute_surface_temperature(point_sud,  w, target_date.month, target_date.day)

        cond_nord, _ = classify_snow_condition(point_nord, w, target_date.month, target_date.day)
        cond_sud,  _ = classify_snow_condition(point_sud,  w, target_date.month, target_date.day)

        result.append({
            "hour": w.hour,
            "temp_2m": w.temperature_2m,
            "nord": {
                "temp_surface": round(temp_nord, 2),
                "condition": cond_nord.name,
            },
            "sud": {
                "temp_surface": round(temp_sud, 2),
                "condition": cond_sud.name,
            },
            "delta_temp": round(temp_sud - temp_nord, 2),
        })

    return result
    
@app.get("/debug/terrain")
def debug_terrain(lat: float, lon: float):
    from data.fetchers.terrain import (
        _fetch_from_ign_wcs,
        _estimate_terrain_from_neighbors,
        _static_estimate,
        get_terrain_data
    )

    return {
        "ign_wcs": str(_fetch_from_ign_wcs(lat, lon)),
        "open_elevation": str(_estimate_terrain_from_neighbors(lat, lon)),
        "final": str(get_terrain_data(lat, lon))
    }
    