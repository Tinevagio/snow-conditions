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
    1       >35°          180m            18°
    2       >32°          300m            22°
    3       >29°          500m            28°
    4       >25°          750m            34°
    5       >20°         1000m            42°
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

# Paramètres BERA → simulation avalanche
BERA_PARAMS = {
    1: {"slope_min": 35, "cone_length_m": 180,  "cone_angle_deg": 18},
    2: {"slope_min": 32, "cone_length_m": 300,  "cone_angle_deg": 22},
    3: {"slope_min": 29, "cone_length_m": 500,  "cone_angle_deg": 28},
    4: {"slope_min": 25, "cone_length_m": 750,  "cone_angle_deg": 34},
    5: {"slope_min": 20, "cone_length_m": 1000, "cone_angle_deg": 42},
}

# Correspondance exposition texte → degrés (centre de secteur)
ASPECT_DEGREES = {
    "N":  0,   "NE": 45,  "E":  90,  "SE": 135,
    "S":  180, "SW": 225, "W":  270, "NW": 315,
}

# Tolérance angulaire pour la correspondance exposition (±demi-secteur)
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
    pentes_dangereuses: Dict[str, bool]   # {"N": True, "NE": False, ...}

@dataclass
class StartZone:
    """Cellule de départ d'avalanche."""
    lat:        float
    lon:        float
    elevation:  float
    slope_deg:  float
    aspect_deg: float
    risque:     int    # niveau BERA applicable (bas ou haut selon altitude)

@dataclass
class AvalancheCone:
    """Cône de propagation depuis une zone de départ."""
    start:          StartZone
    cone_length_m:  float
    cone_angle_deg: float
    # Polygone GeoJSON du cône [lon, lat] (convention GeoJSON)
    polygon:        List[Tuple[float, float]]

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_slope_grid(massif_id: int) -> Optional[Dict[str, np.ndarray]]:
    """Charge le .npz précalculé pour un massif."""
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
    """Charge les données BERA pour un massif."""
    if not BERA_JSON_PATH.exists():
        return None
    with open(BERA_JSON_PATH) as f:
        data = json.load(f)

    for item in data:
        if item.get("id") == massif_id:
            risque_bas = item.get("risque_bas")
            if risque_bas is None:
                return None  # pas de données risque
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
    """Vérifie si une exposition est dans les pentes dangereuses BERA."""
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
    """Calcule le point destination depuis lat/lon, cap et distance."""
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
) -> List[StartZone]:
    """
    Filtre les cellules de départ selon :
      - bbox (si fournie)
      - altitude > limite d'enneigement
      - pente > seuil BERA
      - exposition dangereuse selon BERA
    """
    lats    = grid["lat"]
    lons    = grid["lon"]
    elevs   = grid["elevation"]
    slopes  = grid["slope"]
    aspects = grid["aspect"]

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

        # Niveau de risque applicable selon altitude
        if (bera.risque_haut is not None
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

        # Filtre exposition
        if not aspect_is_dangerous(aspect, bera.pentes_dangereuses):
            continue

        zones.append(StartZone(
            lat=lat, lon=lon, elevation=elev,
            slope_deg=slope, aspect_deg=aspect,
            risque=risque,
        ))

    # Si trop de zones, sous-échantillonner uniformément
    if len(zones) > max_zones:
        step = len(zones) // max_zones
        zones = zones[::step][:max_zones]

    return zones

# ---------------------------------------------------------------------------
# Propagation des cônes
# ---------------------------------------------------------------------------

def propagate_cone(zone: StartZone) -> AvalancheCone:
    """
    Calcule le polygone du cône d'avalanche depuis une zone de départ.

    Le cône suit la direction de plus grande pente (aspect) vers le bas
    (aspect + 180°), avec un angle d'ouverture latéral.
    """
    params = BERA_PARAMS[zone.risque]
    length_m   = params["cone_length_m"]
    half_angle = params["cone_angle_deg"] / 2

    # Direction de descente = aspect + 180° (vers le bas de la pente)
    # Les .npz sont générés avec atan2(dz_dx, -dz_dy) qui inverse l'aspect,
    # le +180° compense cette inversion → ne pas modifier sans regénérer les .npz
    downslope = (zone.aspect_deg + 180) % 360

    # Apex du cône = zone de départ
    apex = (zone.lat, zone.lon)

    # Point central du cône à distance length_m
    tip_lat, tip_lon = destination_point(zone.lat, zone.lon, downslope, length_m)

    # Bords gauche et droit du cône
    left_bearing  = (downslope - half_angle) % 360
    right_bearing = (downslope + half_angle) % 360

    left_lat,  left_lon  = destination_point(zone.lat, zone.lon, left_bearing,  length_m)
    right_lat, right_lon = destination_point(zone.lat, zone.lon, right_bearing, length_m)

    # Arc du cône : interpoler N points sur l'arc entre gauche et droite
    n_arc = max(5, int(params["cone_angle_deg"] / 5))
    arc_points = []
    for k in range(n_arc + 1):
        t = k / n_arc
        bearing = left_bearing + t * params["cone_angle_deg"]
        bearing = bearing % 360
        plat, plon = destination_point(zone.lat, zone.lon, bearing, length_m)
        arc_points.append((plon, plat))  # GeoJSON: [lon, lat]

    # Polygone fermé : apex → bord gauche → arc → bord droit → apex
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
) -> dict:
    """Construit le GeoJSON complet : zones de départ + cônes."""
    features = []

    # Zones de départ → points
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

    # Cônes → polygones
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
            "massif_id":   bera.massif_id,
            "massif_name": bera.massif_name,
            "risque_bas":  bera.risque_bas,
            "risque_haut": bera.risque_haut,
            "n_start_zones": len(start_zones),
            "n_cones":       len(cones),
        },
        "features": features,
    }

# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def _bera_info_from_dict(massif_id: int, d: dict) -> Optional[BERAInfo]:
    """Construit un BERAInfo depuis le dict retourné par BeraCorrector.get_massif_info()."""
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


def compute_avalanche_zones(
    massif_id: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    max_zones: int = 300,
    bera_data: Optional[dict] = None,
) -> Optional[dict]:
    """
    Calcule les zones d'avalanche pour un massif et une bbox optionnelle.

    Args:
        massif_id  : ID du massif (correspond au .npz)
        bbox       : (lat_min, lon_min, lat_max, lon_max) optionnel
        max_zones  : nombre max de zones de départ (perf)
        bera_data  : dict brut depuis BeraCorrector.get_massif_info() (optionnel)
                     Si fourni, évite de relire le fichier JSON local.

    Returns:
        GeoJSON dict ou None si données manquantes
    """
    # 1. Charger grille
    grid = load_slope_grid(massif_id)
    if grid is None:
        return {"error": f"Grille pentes non disponible pour massif {massif_id}. "
                         f"Lancez scripts/build_slope_grids.py --massif {massif_id}"}

    # 2. Charger BERA — depuis bera_data injecté ou depuis fichier local
    if bera_data:
        bera = _bera_info_from_dict(massif_id, bera_data)
    else:
        bera = load_bera(massif_id)

    if bera is None:
        return {"error": f"Données BERA non disponibles pour massif {massif_id}"}

    # 3. Filtrer les zones de départ
    start_zones = find_start_zones(grid, bera, bbox=bbox, max_zones=max_zones)

    if not start_zones:
        return to_geojson([], [], bera)

    # 4. Propager les cônes
    cones = [propagate_cone(z) for z in start_zones]

    # 5. Export GeoJSON
    return to_geojson(start_zones, cones, bera)
