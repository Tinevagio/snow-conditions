"""
core/avalanche_model.py
-----------------------
Modèle de simulation des zones d'avalanche.

Étapes :
  1. Chargement grille pentes/exposition (.npz précalculé)
  2. Filtrage cellules de départ selon BERA (pente, exposition, altitude)
  3. Propagation des cônes d'impact à la volée
  4. Export GeoJSON

Paramètres BERA → simulation :
  Risque  Pente départ  Longueur cône  Angle ouverture
    1       >35°          120m            18°
    2       >32°          220m            22°
    3       >29°          400m            28°
    4       >25°          650m            34°
    5       >20°          900m            42°

Comportement riskOverride :
  Si riskOverride > risque BERA réel, le filtre exposition est désactivé :
  on suppose que le danger s'est étendu à tous les versants. Seuls les
  filtres pente et altitude restent actifs.
  Si riskOverride <= risque réel, le filtre exposition BERA est conservé.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR         = Path(__file__).parent.parent / "data"
SLOPE_GRIDS_DIR  = DATA_DIR / "slope_grids"
BERA_JSON_PATH   = DATA_DIR / "bera_enneigement.json"

BERA_PARAMS = {
    1: {"slope_min": 35, "cone_length_m": 120, "cone_angle_deg": 18},
    2: {"slope_min": 32, "cone_length_m": 220, "cone_angle_deg": 22},
    3: {"slope_min": 29, "cone_length_m": 400, "cone_angle_deg": 28},
    4: {"slope_min": 25, "cone_length_m": 650, "cone_angle_deg": 34},
    5: {"slope_min": 20, "cone_length_m": 900, "cone_angle_deg": 42},
}

ASPECT_DEGREES = {
    "N":  0,   "NE": 45,  "E":  90,  "SE": 135,
    "S":  180, "SW": 225, "W":  270, "NW": 315,
}

ASPECT_TOLERANCE_DEG = 25.0

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BERAInfo:
    massif_id:          int
    massif_name:        str
    risque_bas:         int
    risque_haut:        Optional[int]
    risque_altitude_m:  Optional[float]
    limite_nord_m:      Optional[float]
    limite_sud_m:       Optional[float]
    pentes_dangereuses: Dict[str, bool]

@dataclass
class StartZone:
    lat:        float
    lon:        float
    elevation:  float
    slope_deg:  float
    aspect_deg: float
    risque:     int

@dataclass
class AvalancheCone:
    start:          StartZone
    cone_length_m:  float
    cone_angle_deg: float
    polygon:        List[Tuple[float, float]]

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_slope_grid(massif_id: int) -> Optional[Dict[str, np.ndarray]]:
    path = SLOPE_GRIDS_DIR / f"{massif_id}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {
        "lat":       d["lat"],
        "lon":       d["lon"],
        "elevation": d["elevation"],
        "slope":     d["slope"],
        "aspect":    d["aspect"],
    }


def load_bera(massif_id: int) -> Optional[BERAInfo]:
    if not BERA_JSON_PATH.exists():
        return None
    with open(BERA_JSON_PATH) as f:
        data = json.load(f)
    for item in data:
        if item.get("id") == massif_id:
            risque_bas = item.get("risque_bas")
            if risque_bas is None:
                return None
            return BERAInfo(
                massif_id=massif_id,
                massif_name=item.get("massif", ""),
                risque_bas=int(risque_bas),
                risque_haut=int(item["risque_haut"]) if item.get("risque_haut") else None,
                risque_altitude_m=float(item["risque_altitude_m"]) if item.get("risque_altitude_m") else None,
                limite_nord_m=float(item["limite_nord_m"]) if item.get("limite_nord_m") else None,
                limite_sud_m=float(item["limite_sud_m"]) if item.get("limite_sud_m") else None,
                pentes_dangereuses=item.get("pentes_dangereuses", {}),
            )
    return None

# ---------------------------------------------------------------------------
# Helpers géographiques
# ---------------------------------------------------------------------------

def aspect_is_dangerous(aspect_deg: float, pentes_dangereuses: Dict[str, bool]) -> bool:
    for direction, dangerous in pentes_dangereuses.items():
        if not dangerous:
            continue
        center = ASPECT_DEGREES.get(direction, 0)
        diff = abs((aspect_deg - center + 180) % 360 - 180)
        if diff <= ASPECT_TOLERANCE_DEG:
            return True
    return False


def meters_to_deg_lat(meters: float) -> float:
    return meters / 111_000


def meters_to_deg_lon(meters: float, lat: float) -> float:
    return meters / (111_000 * math.cos(math.radians(lat)))


def destination_point(lat: float, lon: float,
                      bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    dlat = meters_to_deg_lat(distance_m) * math.cos(math.radians(bearing_deg))
    dlon = meters_to_deg_lon(distance_m, lat) * math.sin(math.radians(bearing_deg))
    return lat + dlat, lon + dlon

# ---------------------------------------------------------------------------
# Filtrage des zones de départ
# ---------------------------------------------------------------------------

def find_start_zones(
    grid: Dict[str, np.ndarray],
    bera: BERAInfo,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_zones: int = 500,
    risk_override: Optional[int] = None,
) -> List[StartZone]:
    """
    Filtre les cellules de départ selon pente, altitude, exposition.

    risk_override :
      Si fourni ET supérieur au risque réel du massif, le filtre exposition
      est désactivé — on suppose que le danger s'étend à tous les versants.
      Si absent ou inférieur/égal au risque réel, le filtre exposition BERA
      est appliqué normalement.

      Logique : un risque simulé aggravé représente un scénario où les
      conditions se sont dégradées sur l'ensemble du massif, pas seulement
      sur les versants identifiés dans le bulletin réel.
    """
    lats    = grid["lat"]
    lons    = grid["lon"]
    elevs   = grid["elevation"]
    slopes  = grid["slope"]
    aspects = grid["aspect"]

    # Risque "réel" maximal du massif pour comparaison avec l'override
    real_max_risque = max(
        bera.risque_bas,
        bera.risque_haut if bera.risque_haut is not None else 0
    )

    # Si override > risque réel → on désactive le filtre exposition.
    # Tous les versants deviennent éligibles (seuls pente et altitude filtrent).
    ignore_aspect_filter = (
        risk_override is not None
        and risk_override > real_max_risque
    )

    zones = []

    for i in range(len(lats)):
        lat, lon = float(lats[i]), float(lons[i])
        elev     = float(elevs[i])
        slope    = float(slopes[i])
        aspect   = float(aspects[i])

        # Filtre bbox
        if bbox:
            lat_min, lon_min, lat_max, lon_max = bbox
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue

        # Niveau de risque applicable : override prioritaire, sinon logique
        # altitude bas/haut BERA
        if risk_override is not None:
            risque = risk_override
        elif (bera.risque_haut is not None
              and bera.risque_altitude_m is not None
              and elev >= bera.risque_altitude_m):
            risque = bera.risque_haut
        else:
            risque = bera.risque_bas

        params = BERA_PARAMS.get(risque)
        if params is None:
            continue

        # Filtre altitude minimum d'enneigement
        is_north = aspect <= 90 or aspect >= 270
        limite = (bera.limite_nord_m if is_north else bera.limite_sud_m) or 1000
        if elev < limite:
            continue

        # Filtre pente
        if slope < params["slope_min"]:
            continue

        # Filtre exposition — désactivé si override aggravé
        if not ignore_aspect_filter:
            if not aspect_is_dangerous(aspect, bera.pentes_dangereuses):
                continue

        zones.append(StartZone(
            lat=lat, lon=lon, elevation=elev,
            slope_deg=slope, aspect_deg=aspect,
            risque=risque,
        ))

    if len(zones) > max_zones:
        step = len(zones) // max_zones
        zones = zones[::step][:max_zones]

    return zones

# ---------------------------------------------------------------------------
# Propagation des cônes
# ---------------------------------------------------------------------------

def propagate_cone(zone: StartZone) -> AvalancheCone:
    params     = BERA_PARAMS[zone.risque]
    length_m   = params["cone_length_m"]
    half_angle = params["cone_angle_deg"] / 2

    # Direction de descente = aspect + 180°
    # Les .npz sont générés avec atan2(dz_dx, -dz_dy) qui inverse l'aspect,
    # le +180° compense cette inversion → ne pas modifier sans regénérer les .npz
    downslope = (zone.aspect_deg + 180) % 360

    n_arc = max(5, int(params["cone_angle_deg"] / 5))
    arc_points = []
    for k in range(n_arc + 1):
        t = k / n_arc
        bearing = (downslope - half_angle + t * params["cone_angle_deg"]) % 360
        plat, plon = destination_point(zone.lat, zone.lon, bearing, length_m)
        arc_points.append((plon, plat))

    polygon = (
        [(zone.lon, zone.lat)]
        + arc_points
        + [(zone.lon, zone.lat)]
    )

    return AvalancheCone(
        start=zone,
        cone_length_m=length_m,
        cone_angle_deg=params["cone_angle_deg"],
        polygon=polygon,
    )

# ---------------------------------------------------------------------------
# Export GeoJSON
# ---------------------------------------------------------------------------

def to_geojson(
    start_zones: List[StartZone],
    cones: List[AvalancheCone],
    bera: BERAInfo,
    risk_override: Optional[int] = None,
) -> dict:
    features = []

    for z in start_zones:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [z.lon, z.lat],
            },
            "properties": {
                "type":       "start_zone",
                "elevation":  round(z.elevation),
                "slope_deg":  round(z.slope_deg, 1),
                "aspect_deg": round(z.aspect_deg, 1),
                "risque":     z.risque,
            },
        })

    for c in cones:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [c.polygon],
            },
            "properties": {
                "type":           "cone",
                "risque":         c.start.risque,
                "cone_length_m":  c.cone_length_m,
                "cone_angle_deg": c.cone_angle_deg,
                "start_lat":      c.start.lat,
                "start_lon":      c.start.lon,
                "elevation":      round(c.start.elevation),
                "slope_deg":      round(c.start.slope_deg, 1),
            },
        })

    return {
        "type": "FeatureCollection",
        "properties": {
            "massif_id":          bera.massif_id,
            "massif_name":        bera.massif_name,
            "risque_bas":         bera.risque_bas,
            "risque_haut":        bera.risque_haut,
            "risk_override":      risk_override,
            "aspect_filter_active": risk_override is None or risk_override <= max(
                bera.risque_bas,
                bera.risque_haut if bera.risque_haut is not None else 0
            ),
            "n_start_zones":      len(start_zones),
            "n_cones":            len(cones),
        },
        "features": features,
    }

# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _bera_info_from_dict(massif_id: int, d: dict) -> Optional[BERAInfo]:
    try:
        risque_bas = d.get("risque_bas")
        if risque_bas is None:
            return None
        return BERAInfo(
            massif_id=massif_id,
            massif_name=d.get("massif_name", ""),
            risque_bas=int(risque_bas),
            risque_haut=int(d["risque_haut"]) if d.get("risque_haut") else None,
            risque_altitude_m=float(d["risque_altitude_m"]) if d.get("risque_altitude_m") else None,
            limite_nord_m=float(d["limite_nord_m"]) if d.get("limite_nord_m") else None,
            limite_sud_m=float(d["limite_sud_m"]) if d.get("limite_sud_m") else None,
            pentes_dangereuses=d.get("pentes_dangereuses", {}),
        )
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def compute_avalanche_zones(
    massif_id: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_zones: int = 300,
    bera_data: Optional[dict] = None,
    risk_override: Optional[int] = None,
) -> Optional[dict]:
    """
    Calcule les zones d'avalanche pour un massif et une bbox optionnelle.

    Args:
        massif_id    : ID du massif (correspond au .npz)
        bbox         : (lat_min, lon_min, lat_max, lon_max) optionnel
        max_zones    : nombre max de zones de départ (perf)
        bera_data    : dict brut depuis BeraCorrector (évite relecture JSON)
        risk_override: niveau 1-5 simulé. Si > risque réel, désactive le
                       filtre exposition pour couvrir tous les versants.
    """
    grid = load_slope_grid(massif_id)
    if grid is None:
        return {"error": f"Grille pentes non disponible pour massif {massif_id}. "
                         f"Lancez scripts/build_slope_grids.py --massif {massif_id}"}

    if bera_data:
        bera = _bera_info_from_dict(massif_id, bera_data)
    else:
        bera = load_bera(massif_id)

    if bera is None:
        return {"error": f"Données BERA non disponibles pour massif {massif_id}"}

    start_zones = find_start_zones(
        grid, bera,
        bbox=bbox,
        max_zones=max_zones,
        risk_override=risk_override,
    )

    if not start_zones:
        return to_geojson([], [], bera, risk_override)

    cones = [propagate_cone(z) for z in start_zones]

    return to_geojson(start_zones, cones, bera, risk_override)
