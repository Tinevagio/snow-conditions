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
import logging
from datetime import datetime, timezone, date, timedelta
from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Imports internes — chemins relatifs au projet
# ---------------------------------------------------------------------------

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.snow_model import (
    GridPoint,
    SnowResult,
    compute_snow_conditions,
    compute_surface_temperature,
    classify_snow_condition,
)
from core.solar_radiation import best_powder_window
from data.fetchers.openmeteo import get_hourly_weather
from core.terrain import get_terrain_grid, get_terrain_data, TerrainPoint

from datetime import date

logger = logging.getLogger(__name__)

def parse_date(date_str: str) -> date:
    """Parse une date au format YYYY-MM-DD"""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Date invalide : '{date_str}'. Format attendu : YYYY-MM-DD",
        )

# ---------------------------------------------------------------------------
# BERA Corrector — instancié une seule fois au démarrage
# Les fichiers sont lus depuis Ski-touring-live via GitHub raw si absents en local.
# ---------------------------------------------------------------------------

try:
    from bera_corrector import BeraCorrector
    _bera_corrector = BeraCorrector(
        bera_json_path="data/bera_enneigement.json",
        polygons_path="data/massif_polygons.json",
        use_github=True,   # fallback GitHub si fichiers locaux absents
        alpha_max=0.8,
    )
    logger.info("✅ BeraCorrector initialisé")
except Exception as e:
    _bera_corrector = None
    print(f"❌ BeraCorrector ERREUR: {e}")
    logger.warning(f"⚠️  BeraCorrector indisponible, correction BERA désactivée : {e}")


def _apply_bera(weather: list, terrain: TerrainPoint) -> list:
    """
    Applique la correction BERA sur une liste de HourlyWeather pour un point terrain.
    Si le corrector est indisponible, retourne la liste inchangée.
    """
    if _bera_corrector is None:
        return weather
    try:
        return _bera_corrector.correct(
            hourly_list=weather,
            lat=terrain.lat,
            lon=terrain.lon,
            elevation=terrain.elevation_m,
            aspect_deg=terrain.aspect_deg,
        )
    except Exception as e:
        logger.warning(f"BeraCorrector.correct() error ({terrain.lat},{terrain.lon}): {e}")
        return weather


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
    allow_origins=["*"],
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
    "POWDER_COLD":   {"label": "Poudre froide",       "color": "#4A90D9"},
    "POWDER_WARM":   {"label": "Poudre réchauffée",   "color": "#7FB3E8"},
    "SPRING_SNOW":   {"label": "Neige de printemps",  "color": "#F5A623"},
    "CRUST_MORNING": {"label": "Croûte de regel",     "color": "#D0021B"},
    "WET_HEAVY":     {"label": "Neige humide lourde", "color": "#8B572A"},
    "WIND_AFFECTED": {"label": "Neige soufflée",      "color": "#9B9B9B"},
    "OLD_PACKED":    {"label": "Neige ancienne",      "color": "#B8D4F0"},
    "UNDEFINED":     {"label": "Indéterminé",         "color": "#EEEEEE"},
}

class HourlyCondition(BaseModel):
    hour:         int   = Field(..., description="Heure UTC (0-23)")
    condition:    str   = Field(..., description="Code condition")
    label:        str   = Field(..., description="Libellé lisible")
    color:        str   = Field(..., description="Couleur hex pour la carte")
    temp_surface: float = Field(..., description="Température de surface estimée (°C)")
    wind_speed:   float = Field(..., description="Vitesse du vent (km/h)")

class PointConditions(BaseModel):
    lat:          float
    lon:          float
    elevation_m:  float
    aspect_deg:   float
    aspect_label: str
    slope_deg:    float
    bera:         Optional[BeraInfo] = None 
    hours:        List[HourlyCondition]

class ConditionsResponse(BaseModel):
    date:         str
    bbox:         List[float] = Field(..., description="[lat_min, lon_min, lat_max, lon_max]")
    resolution_m: float
    generated_at: str
    points:       List[PointConditions]

class WindowPoint(BaseModel):
    lat:                 float
    lon:                 float
    elevation_m:         float
    aspect_deg:          float
    aspect_label:        str
    slope_deg:           float
    powder_until_hour:   Optional[int]
    spring_optimal_hour: Optional[int]

class BestWindowResponse(BaseModel):
    date:   str
    bbox:   List[float]
    points: List[WindowPoint]

class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: str

class BeraSnowLevel(BaseModel):
    alti:  int
    N_cm:  Optional[int]
    S_cm:  Optional[int]

class BeraInfo(BaseModel):
    massif_name:       Optional[str]
    bera_date:         Optional[str]
    limite_nord_m:     Optional[int]
    limite_sud_m:      Optional[int]
    bera_72h_cm:       Optional[float]
    bera_24h_cm:       Optional[float]
    enneigement_niveaux: Optional[List[BeraSnowLevel]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: str) -> tuple:
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
            detail=f"bbox invalide (attendu 'lat_min,lon_min,lat_max,lon_max') : {e}",
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
    from collections import defaultdict
    by_point: dict = defaultdict(list)
    for r in results:
        by_point[(round(r.lat, 6), round(r.lon, 6))].append(r)

    terrain_map = {(round(t.lat, 6), round(t.lon, 6)): t for t in terrain_points}
    output = []

    for gp in grid:
        key     = (round(gp.lat, 6), round(gp.lon, 6))
        terrain = terrain_map.get(key)
        hours_data = sorted(by_point.get(key, []), key=lambda r: r.hour)

        # ── Infos BERA pour ce point ──────────────────────────────────────────
        bera_info = None
        if _bera_corrector is not None and terrain is not None:
            try:
                raw = _bera_corrector.get_massif_info(terrain.lat, terrain.lon)
                if raw:
                    niveaux = [
                        BeraSnowLevel(alti=n["alti"], N_cm=n.get("N_cm"), S_cm=n.get("S_cm"))
                        for n in raw.get("enneigement_niveaux", [])
                    ]
                    bera_info = BeraInfo(
                        massif_name=raw.get("massif_name"),
                        bera_date=raw.get("bera_date"),
                        limite_nord_m=raw.get("limite_nord_m"),
                        limite_sud_m=raw.get("limite_sud_m"),
                        bera_72h_cm=raw.get("bera_72h_cm"),
                        bera_24h_cm=raw.get("bera_24h_cm"),
                        enneigement_niveaux=niveaux,
                    )
            except Exception as e:
                logger.warning(f"BeraInfo error ({gp.lat},{gp.lon}): {e}")

        output.append(PointConditions(
            lat=gp.lat,
            lon=gp.lon,
            elevation_m=terrain.elevation_m if terrain else gp.elevation,
            aspect_deg=terrain.aspect_deg   if terrain else gp.aspect,
            aspect_label=terrain.aspect_label() if terrain else "?",
            slope_deg=terrain.slope_deg     if terrain else gp.slope,
            bera=bera_info,
            hours=[_result_to_hourly(r) for r in hours_data],
        ))

    return output

def _compute_windows(point: TerrainPoint, weather_series: list, month: int, day: int) -> dict:
    gp = GridPoint(
        lat=point.lat, lon=point.lon,
        elevation=point.elevation_m,
        aspect=point.aspect_deg,
        slope=point.slope_deg,
    )

    powder_until = None
    for w in sorted(weather_series, key=lambda x: x.hour):
        cond, _ = classify_snow_condition(gp, w, month, day)
        if cond.name in ("POWDER_COLD", "POWDER_WARM"):
            powder_until = w.hour
        else:
            break

    spring_hours = []
    for w in sorted(weather_series, key=lambda x: x.hour):
        cond, _ = classify_snow_condition(gp, w, month, day)
        if cond.name == "SPRING_SNOW" and w.hour <= 14:  # pas après 14h
            spring_hours.append(w.hour)

    spring_optimal = None
    if spring_hours:
        best_start, best_len = spring_hours[0], 1
        cur_start,  cur_len  = spring_hours[0], 1
        for i in range(1, len(spring_hours)):
            if spring_hours[i] == spring_hours[i - 1] + 1:
                cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_start, cur_len = spring_hours[i], 1
        if best_len >= 1:
            spring_optimal = best_start + (best_len // 2)  # milieu de la meilleure fenêtre

    return {"powder_until_hour": powder_until, "spring_optimal_hour": spring_optimal}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Système"])
def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/conditions", response_model=ConditionsResponse, tags=["Conditions"])
def get_conditions(
    bbox: str = Query(..., description="Zone géographique : lat_min,lon_min,lat_max,lon_max", example="45.90,6.85,45.95,6.95"),
    date: Optional[str] = Query(None, description="Date au format YYYY-MM-DD (défaut : aujourd'hui UTC)", example="2024-02-15"),
    resolution_m: float = Query(500, ge=100, le=2000, description="Résolution de la grille en mètres (100–2000)"),
    tiff_path: Optional[str] = Query(None, description="Chemin vers un GeoTIFF IGN local (optionnel)"),
):
    lat_min, lon_min, lat_max, lon_max = _parse_bbox(bbox)

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
            padding_m=1000,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur terrain : {e}")

    if not terrain_points:
        raise HTTPException(status_code=422, detail="Aucun point dans la bbox.")

    # 2. Météo (centre bbox)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    try:
        weather = get_hourly_weather(center_lat, center_lon, target_date=target_dt.date())
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo inaccessible : {e}")

    if not weather:
        raise HTTPException(status_code=502, detail="Aucune donnée météo disponible.")

    # 3. GridPoints + correction BERA par point
    grid = []
    for p in terrain_points:
        # ── Correction BERA : recalibre snowfall_last_72h/24h selon massif réel ──
        corrected_weather = _apply_bera(weather, p)
        grid.append(GridPoint(
            lat=p.lat,
            lon=p.lon,
            elevation=p.elevation_m,
            aspect=p.aspect_deg,
            slope=p.slope_deg,
        ))

    # 4. Calcul des conditions
    # Note : on utilise la météo corrigée du centre bbox pour tous les points.
    # Pour une correction point-par-point fine, compute_snow_conditions devrait
    # accepter une weather list par point — évolution future possible.
    corrected_weather_center = _apply_bera(weather, terrain_points[len(terrain_points)//2])
    results = compute_snow_conditions(grid, corrected_weather_center, target_dt.month, target_dt.day)

    # 5. Formatage
    points_out = _group_results_by_point(
        [p for p in terrain_points if not p.is_padding],
        terrain_points,
        results,
    )

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
        weather = get_hourly_weather(lat, lon, target_date=target_dt.date())
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo inaccessible : {e}")

    # ── Correction BERA point unique ──────────────────────────────────────────
    weather = _apply_bera(weather, terrain)

    results   = compute_snow_conditions([gp], weather, target_dt.month, target_dt.day)
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
    bbox: str = Query(..., example="45.90,6.85,45.95,6.95"),
    date: Optional[str] = Query(None, description="Date YYYY-MM-DD (défaut : demain)"),
    resolution_m: float = Query(500, ge=100, le=2000),
):
    lat_min, lon_min, lat_max, lon_max = _parse_bbox(bbox)

    if date:
        try:
            target_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="date invalide")
    else:
        target_dt = datetime.now(timezone.utc) + timedelta(days=1)

    terrain_points = get_terrain_grid(
        lat_min, lon_min, lat_max, lon_max,
        resolution_m=resolution_m,
        padding_m=1000,
    )

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    try:
        weather = get_hourly_weather(center_lat, center_lon, target_date=target_dt.date())
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Open-Meteo inaccessible : {e}")

    points_out = []
    for p in [pt for pt in terrain_points if not pt.is_padding]:
        # ── Correction BERA par point ─────────────────────────────────────────
        corrected = _apply_bera(weather, p)
        w = _compute_windows(p, corrected, target_dt.month, target_dt.day)
        points_out.append(WindowPoint(
            lat=p.lat,
            lon=p.lon,
            elevation_m=p.elevation_m,
            aspect_deg=p.aspect_deg,
            aspect_label=p.aspect_label(),
            slope_deg=p.slope_deg,
            powder_until_hour=w["powder_until_hour"],
            spring_optimal_hour=w["spring_optimal_hour"],
        ))

    return BestWindowResponse(
        date=target_dt.strftime("%Y-%m-%d"),
        bbox=[lat_min, lon_min, lat_max, lon_max],
        points=points_out,
    )


# ---------------------------------------------------------------------------
# Endpoints debug (inchangés)
# ---------------------------------------------------------------------------

@app.get("/debug/point")
def debug_point(lat: float, lon: float, date: str = None,
                aspect: float = 0, elevation: float = 1500, slope: float = 15):
    target_date  = parse_date(date)
    weather_series = get_hourly_weather(lat, lon, target_date)

    result = []
    for w in weather_series:
        point        = GridPoint(lat=lat, lon=lon, elevation=elevation, aspect=aspect, slope=slope)
        temp_surface = compute_surface_temperature(point, w, target_date.month, target_date.day)
        condition, _ = classify_snow_condition(point, w, target_date.month, target_date.day)

        from core.solar_radiation import effective_radiation
        rad = effective_radiation(
            hour_utc=w.hour, lat=lat, lon=lon,
            month=target_date.month, day=target_date.day,
            aspect=aspect, slope=slope, altitude_m=elevation,
        )

        result.append({
            "hour":                   w.hour,
            "temp_2m":                w.temperature_2m,
            "temp_surface":           round(temp_surface, 2),
            "wind_speed":             w.wind_speed,
            "snowfall_24h":           w.snowfall_last_24h,
            "snowfall_72h":           w.snowfall_last_72h,
            "hours_above_zero_48h":   w.hours_above_zero_last_48h,
            "hours_below_minus2_12h": w.hours_below_minus2_last_12h,
            "condition":              condition.name,
            "solar_bonus":            round(rad.temperature_correction, 2),
            "altitude_correction":    round(-((elevation - w.reference_elevation) * 0.006), 2),
            "solar_correction":       round(temp_surface - w.temperature_2m, 2),
        })

    return result


@app.get("/debug/compare")
def debug_compare(lat: float, lon: float, date: str = None,
                  elevation: float = 1500, slope: float = 15):
    target_date    = parse_date(date)
    weather_series = get_hourly_weather(lat, lon, target_date)

    result = []
    for w in weather_series:
        point_nord = GridPoint(lat=lat, lon=lon, elevation=elevation, aspect=0,   slope=slope)
        point_sud  = GridPoint(lat=lat, lon=lon, elevation=elevation, aspect=180, slope=slope)
        temp_nord  = compute_surface_temperature(point_nord, w, target_date.month, target_date.day)
        temp_sud   = compute_surface_temperature(point_sud,  w, target_date.month, target_date.day)
        cond_nord, _ = classify_snow_condition(point_nord, w, target_date.month, target_date.day)
        cond_sud,  _ = classify_snow_condition(point_sud,  w, target_date.month, target_date.day)

        result.append({
            "hour":    w.hour,
            "temp_2m": w.temperature_2m,
            "nord":    {"temp_surface": round(temp_nord, 2), "condition": cond_nord.name},
            "sud":     {"temp_surface": round(temp_sud, 2),  "condition": cond_sud.name},
            "delta_temp": round(temp_sud - temp_nord, 2),
        })

    return result


@app.get("/debug/terrain")
def debug_terrain(lat: float, lon: float, tiff_path: str = None):
    from core.terrain import (
        _fetch_from_ign_wcs,
        _estimate_terrain_from_neighbors,
        _extract_from_geotiff,
        get_terrain_data,
        RASTERIO_AVAILABLE,
    )
    return {
        "rasterio_available": RASTERIO_AVAILABLE,
        "tiff_path_recu":     tiff_path,
        "geotiff":            str(_extract_from_geotiff(tiff_path, lat, lon) if tiff_path else "non fourni"),
        "ign_wcs":            str(_fetch_from_ign_wcs(lat, lon)),
        "open_elevation":     str(_estimate_terrain_from_neighbors(lat, lon)),
        "final":              str(get_terrain_data(lat, lon, tiff_path=tiff_path)),
    }


@app.get("/debug/bera")
def debug_bera(lat: float, lon: float):
    """Retourne les infos BERA brutes pour un point GPS (debug correction)."""
    if _bera_corrector is None:
        return {"error": "BeraCorrector non initialisé"}
    return _bera_corrector.get_massif_info(lat, lon)