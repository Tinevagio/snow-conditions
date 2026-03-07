"""
terrain.py
----------
Lecture du Modèle Numérique de Terrain (MNT) IGN pour extraire
l'élévation, l'exposition (aspect) et l'inclinaison (slope) par point GPS.

Deux modes de fonctionnement :
  1. Fichier GeoTIFF local (RGE ALTI® IGN, résolution 5m ou 25m)
  2. API distante IGN (WCS - Web Coverage Service) si pas de fichier local

Structure retournée par get_terrain_data() :
  TerrainPoint(lat, lon, elevation_m, aspect_deg, slope_deg)

Conventions :
  aspect_deg : 0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest
  slope_deg  : 0° = plat, 90° = vertical

Dépendances :
  - rasterio  (lecture GeoTIFF)
  - pyproj    (reprojection GPS ↔ Lambert93)
  - numpy
  - urllib    (stdlib, appel WCS IGN si pas de fichier local)
"""

import math
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    from pyproj import Transformer
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TerrainPoint:
    lat: float
    lon: float
    elevation_m: float
    aspect_deg: float   # 0=Nord, 90=Est, 180=Sud, 270=Ouest
    slope_deg: float    # 0=plat, 90=vertical
    is_padding: bool = False

    def aspect_label(self) -> str:
        """Retourne une étiquette lisible de l'exposition."""
        a = self.aspect_deg % 360
        labels = [
            (22.5,  "N"),  (67.5,  "NE"), (112.5, "E"),  (157.5, "SE"),
            (202.5, "S"),  (247.5, "SO"), (292.5, "O"),  (337.5, "NO"),
        ]
        for limit, label in labels:
            if a < limit:
                return label
        return "N"

    def is_north_facing(self) -> bool:
        a = self.aspect_deg % 360
        return a < 45 or a > 315

    def is_south_facing(self) -> bool:
        a = self.aspect_deg % 360
        return 135 < a < 225

# ---------------------------------------------------------------------------
# Calculs géométriques (pente + exposition) à partir d'une grille d'élévation
# ---------------------------------------------------------------------------

def _compute_aspect_slope(
    z: "np.ndarray",
    cell_size_m: float
) -> Tuple[float, float]:
    """
    Calcule aspect et slope à partir d'une fenêtre 3×3 de valeurs d'élévation.
    Algorithme Horn (1981) — standard SIG.

    z : tableau numpy 3×3, z[0,0] = coin nord-ouest
    cell_size_m : taille d'une cellule en mètres
    Retourne (aspect_deg, slope_deg)
    """
    # Gradients X (ouest→est) et Y (nord→sud)
    dz_dx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) -
              (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * cell_size_m)
    dz_dy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) -
              (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * cell_size_m)

    slope_rad = math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = math.degrees(slope_rad)

    # Aspect : 0° = Nord, sens horaire
    if dz_dx == 0 and dz_dy == 0:
        aspect_deg = 0.0  # plat
    else:
        aspect_rad = math.atan2(dz_dx, -dz_dy)
        aspect_deg = math.degrees(aspect_rad) % 360

    return aspect_deg, slope_deg


# ---------------------------------------------------------------------------
# Mode 1 : lecture d'un fichier GeoTIFF local (rasterio requis)
# ---------------------------------------------------------------------------

def _extract_from_geotiff(
    tiff_path: str,
    lat: float,
    lon: float
) -> Optional[TerrainPoint]:
    """
    Extrait élévation, aspect et slope depuis un GeoTIFF IGN.
    Supporte Lambert93 (EPSG:2154) et WGS84 (EPSG:4326).
    Retourne None si le point est hors emprise ou si rasterio est absent.
    """
    if not RASTERIO_AVAILABLE:
        return None

    try:
        import numpy as np
        with rasterio.open(tiff_path) as src:
            crs = src.crs

            # Reprojection GPS → CRS du raster
            #if crs.to_epsg() == 2154:
                #transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                #x, y = transformer.transform(lon, lat)
            #else:
                #x, y = lon, lat
            
            
            crs_epsg = crs.to_epsg() if crs is not None else None
            
            # Les fichiers ASC IGN sont implicitement en Lambert93
            if crs_epsg == 2154 or crs_epsg is None:
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat
            
            row, col = rowcol(src.transform, x, y)

            # Fenêtre 3×3 pour le calcul de pente/aspect
            row_min = max(row - 1, 0)
            col_min = max(col - 1, 0)
            window = rasterio.windows.Window(col_min, row_min, 3, 3)
            data = src.read(1, window=window).astype(float)

            if data.shape != (3, 3):
                return None  # bord de fichier

            elev = float(data[1, 1])

            # Résolution en mètres
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            cell_size = (res_x + res_y) / 2

            aspect, slope = _compute_aspect_slope(data, cell_size)

            return TerrainPoint(lat=lat, lon=lon,
                                elevation_m=elev,
                                aspect_deg=aspect,
                                slope_deg=slope)
    except Exception as e:
        print(f"[terrain] erreur GeoTIFF : {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Mode 2 : API WCS IGN (sans fichier local, réseau requis)
# ---------------------------------------------------------------------------

IGN_WCS_URL = (
    "https://data.geopf.fr/wcs"
    "?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID=RGEALTI_MNT_5M_ASC_LAMB93_IGN69"
    "&FORMAT=image/tiff"
    "&SUBSETTINGCRS=EPSG:4326"
    "&OUTPUTCRS=EPSG:4326"
    "&SUBSET=Long({lon_min},{lon_max})"
    "&SUBSET=Lat({lat_min},{lat_max})"
)

def _fetch_from_ign_wcs(lat: float, lon: float, delta: float = 0.003) -> Optional[TerrainPoint]:
    """
    Télécharge un petit patch du RGE ALTI via l'API WCS IGN (gratuit, sans clé).
    delta ~ 0.003° ≈ 300m, suffisant pour un voisinage 3×3 à 5m de résolution.
    Nécessite rasterio pour décoder le GeoTIFF retourné en mémoire.
    """
    if not RASTERIO_AVAILABLE:
        return None

    try:
        import numpy as np
        import rasterio
        from rasterio.io import MemoryFile

        url = IGN_WCS_URL.format(
            lon_min=lon - delta, lon_max=lon + delta,
            lat_min=lat - delta, lat_max=lat + delta,
        )
        req = urllib.request.Request(url, headers={"User-Agent": "snow-conditions/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()

        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                # Point central
                transformer = Transformer.from_crs("EPSG:4326", src.crs.to_epsg(), always_xy=True)
                x, y = transformer.transform(lon, lat)
                row, col = rowcol(src.transform, x, y)

                row_min = max(row - 1, 0)
                col_min = max(col - 1, 0)
                window = rasterio.windows.Window(col_min, row_min, 3, 3)
                data = src.read(1, window=window).astype(float)

                if data.shape != (3, 3):
                    return None

                elev = float(data[1, 1])
                res = abs(src.transform.a)  # résolution en degrés → à convertir en m
                # 1° lat ≈ 111 000m
                cell_m = res * 111_000
                aspect, slope = _compute_aspect_slope(data, cell_m)

                return TerrainPoint(lat=lat, lon=lon,
                                    elevation_m=elev,
                                    aspect_deg=aspect,
                                    slope_deg=slope)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Mode 3 : Open-Elevation API (fallback léger, élévation seulement)
#          + estimation aspect/slope depuis les voisins
# ---------------------------------------------------------------------------

def _fetch_elevation_open_elevation(lat: float, lon: float) -> Optional[float]:
    """
    Fallback gratuit pour l'élévation uniquement.
    https://api.open-elevation.com
    """
    try:
        # On récupère 9 points en grille 3×3 pour calculer aspect/slope
        delta = 0.001  # ~100m
        points = [
            {"latitude": lat + dy * delta, "longitude": lon + dx * delta}
            for dy in (1, 0, -1)
            for dx in (-1, 0, 1)
        ]
        payload = json.dumps({"locations": points}).encode()
        req = urllib.request.Request(
            "https://api.open-elevation.com/api/v1/lookup",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data["results"]
        elevations = [r["elevation"] for r in results]
        return float(elevations[4])  # point central
    except Exception:
        return None


def _estimate_terrain_from_neighbors(lat: float, lon: float) -> Optional[TerrainPoint]:
    """
    Récupère 9 élévations en grille 3×3 via Open-Elevation,
    puis calcule aspect et slope avec Horn.
    """
    try:
        import numpy as np
        delta = 0.001  # ~100m par cellule
        points = [
            {"latitude": lat + dy * delta, "longitude": lon + dx * delta}
            for dy in (1, 0, -1)
            for dx in (-1, 0, 1)
        ]
        payload = json.dumps({"locations": points}).encode()
        req = urllib.request.Request(
            "https://api.open-elevation.com/api/v1/lookup",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        elevs = [float(r["elevation"]) for r in data["results"]]
        z = np.array(elevs).reshape(3, 3)
        elev = z[1, 1]
        cell_m = delta * 111_000  # ~111m
        aspect, slope = _compute_aspect_slope(z, cell_m)
        return TerrainPoint(lat=lat, lon=lon,
                            elevation_m=elev,
                            aspect_deg=aspect,
                            slope_deg=slope)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback statique (quand tout échoue)
# ---------------------------------------------------------------------------

def _static_estimate(lat: float, lon: float) -> TerrainPoint:
    """
    Estimation grossière basée sur la latitude/longitude pour les Alpes.
    Utilisé uniquement si toutes les sources échouent.
    Élévation : 1500m par défaut, aspect plat.
    """
    return TerrainPoint(
        lat=lat, lon=lon,
        elevation_m=1500.0,
        aspect_deg=0.0,
        slope_deg=15.0
    )


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def get_terrain_data(
    lat: float,
    lon: float,
    tiff_path: Optional[str] = None
) -> TerrainPoint:
    """
    Retourne un TerrainPoint pour le point GPS donné.

    Cascade de sources :
      1. GeoTIFF local (si tiff_path fourni et rasterio dispo)
      2. API WCS IGN   (si réseau dispo et rasterio dispo)
      3. Open-Elevation API (fallback, élévation + approx. aspect/slope)
      4. Estimation statique (dernier recours)

    Usage :
        point = get_terrain_data(45.92, 6.87)
        print(point.elevation_m, point.aspect_label(), point.slope_deg)

        # Avec fichier local IGN :
        point = get_terrain_data(45.92, 6.87, tiff_path="data/rge_alti_chamonix.tif")
    """
    # 1. Fichier GeoTIFF local
    if tiff_path:
        result = _extract_from_geotiff(tiff_path, lat, lon)
        if result:
            return result

    # 2. API WCS IGN
    result = _fetch_from_ign_wcs(lat, lon)
    if result:
        return result

    # 3. Open-Elevation (fallback réseau léger)
    result = _estimate_terrain_from_neighbors(lat, lon)
    if result:
        return result

    # 4. Estimation statique
    return _static_estimate(lat, lon)


def get_terrain_grid(
    lat_min: float, lon_min: float,
    lat_max: float, lon_max: float,
    resolution_m: float = 500,
    tiff_path: Optional[str] = None,
    padding_m: float = 0,           # ← nouveau paramètre
) -> List[TerrainPoint]:
    """
    Retourne une grille de TerrainPoints sur la bbox donnée.
    padding_m : marge autour de la bbox pour le contexte terrain (ombres portées).
                Les points de padding ont is_padding=True — calculés mais non affichés.
    """
    # Conversion padding → degrés
    pad_lat = padding_m / 111_000
    pad_lon = padding_m / (111_000 * math.cos(math.radians((lat_min + lat_max) / 2)))

    # Bbox étendue pour le calcul
    lat_min_ext = lat_min - pad_lat
    lat_max_ext = lat_max + pad_lat
    lon_min_ext = lon_min - pad_lon
    lon_max_ext = lon_max + pad_lon

    delta_lat = resolution_m / 111_000
    delta_lon = resolution_m / (111_000 * math.cos(math.radians((lat_min + lat_max) / 2)))

    points = []
    lat = lat_min_ext
    while lat <= lat_max_ext + 1e-9:
        lon = lon_min_ext
        while lon <= lon_max_ext + 1e-9:
            tp = get_terrain_data(lat, lon, tiff_path=tiff_path)
            # Marquer si le point est dans la bbox visible ou dans le padding
            tp.is_padding = not (lat_min <= lat <= lat_max
                                 and lon_min <= lon <= lon_max)
            points.append(tp)
            lon += delta_lon
        lat += delta_lat

    return points

# ---------------------------------------------------------------------------
# Démo / test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== terrain.py — test standalone ===\n")

    # Test sur quelques points emblématiques des Alpes
    test_points = [
        ("Chamonix centre", 45.9237, 6.8694),
        ("Argentière",      45.9761, 6.9281),
        ("Col du Tour",     45.9979, 6.9994),
    ]

    for name, lat, lon in test_points:
        print(f"Point : {name} ({lat}, {lon})")
        p = get_terrain_data(lat, lon)
        print(f"  Élévation  : {p.elevation_m:.0f} m")
        print(f"  Exposition : {p.aspect_deg:.1f}° ({p.aspect_label()})")
        print(f"  Pente      : {p.slope_deg:.1f}°")
        print(f"  Versant N  : {p.is_north_facing()}")
        print(f"  Versant S  : {p.is_south_facing()}")
        print()

    print("\n=== Intégration avec snow_model.py ===")
    print("""
from terrain import get_terrain_grid
from openmeteo import get_hourly_weather
from snow_model import compute_snow_conditions
from datetime import datetime, timezone

today = datetime.now(timezone.utc)
bbox = (45.90, 6.85, 45.95, 6.95)

# 1. Grille terrain avec vraies données IGN
grid_terrain = get_terrain_grid(*bbox, resolution_m=500)

# 2. Convertir en GridPoints pour snow_model
from snow_model import GridPoint
grid = [GridPoint(lat=p.lat, lon=p.lon,
                  elevation=p.elevation_m,
                  aspect=p.aspect_deg,
                  slope=p.slope_deg)
        for p in grid_terrain]

# 3. Météo temps réel
center_lat = (bbox[0] + bbox[2]) / 2
center_lon = (bbox[1] + bbox[3]) / 2
weather = get_hourly_weather(center_lat, center_lon, target_date=today)

# 4. Conditions de neige
results = compute_snow_conditions(grid, weather, today.month, today.day)
for r in results[:5]:
    print(r)
""")