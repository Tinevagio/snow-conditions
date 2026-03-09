"""
terrain.py
----------
Lecture du Modèle Numérique de Terrain (MNT) IGN pour extraire
l'élévation, l'exposition (aspect) et l'inclinaison (slope) par point GPS.

Sources (par ordre de priorité) :
  1. Fichier GeoTIFF local (RGE ALTI® IGN, résolution 5m ou 25m)
  2. API Altimétrie IGN REST (data.geopf.fr) — batch, sans clé, RGEAlti
  3. Open-Elevation API (fallback)
  4. Estimation statique (dernier recours)

Conventions :
  aspect_deg : 0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest
  slope_deg  : 0° = plat, 90° = vertical
"""

import math
import json
import urllib.request
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
    aspect_deg: float
    slope_deg: float
    is_padding: bool = False

    def aspect_label(self) -> str:
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
# Calculs géométriques Horn (1981)
# ---------------------------------------------------------------------------

def _compute_aspect_slope(z: "np.ndarray", cell_size_m: float) -> Tuple[float, float]:
    dz_dx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) -
              (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * cell_size_m)
    dz_dy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) -
              (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * cell_size_m)

    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))

    if dz_dx == 0 and dz_dy == 0:
        aspect_deg = 0.0
    else:
        aspect_deg = math.degrees(math.atan2(dz_dx, -dz_dy)) % 360

    return aspect_deg, slope_deg


# ---------------------------------------------------------------------------
# Mode 1 : GeoTIFF local
# ---------------------------------------------------------------------------

def _extract_from_geotiff(tiff_path: str, lat: float, lon: float) -> Optional[TerrainPoint]:
    if not RASTERIO_AVAILABLE:
        return None
    try:
        import numpy as np
        with rasterio.open(tiff_path) as src:
            crs = src.crs
            crs_epsg = crs.to_epsg() if crs is not None else None
            if crs_epsg == 2154 or crs_epsg is None:
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
                x, y = transformer.transform(lon, lat)
            else:
                x, y = lon, lat
            row, col = rowcol(src.transform, x, y)
            row_min = max(row - 1, 0)
            col_min = max(col - 1, 0)
            window = rasterio.windows.Window(col_min, row_min, 3, 3)
            data = src.read(1, window=window).astype(float)
            if data.shape != (3, 3):
                return None
            elev = float(data[1, 1])
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            aspect, slope = _compute_aspect_slope(data, (res_x + res_y) / 2)
            return TerrainPoint(lat=lat, lon=lon, elevation_m=elev,
                                aspect_deg=aspect, slope_deg=slope)
    except Exception as e:
        print(f"[terrain] erreur GeoTIFF : {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Mode 2 : API Altimétrie IGN REST (batch multi-points)
# ---------------------------------------------------------------------------

IGN_ALTI_URL      = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
IGN_ALTI_RESOURCE = "ign_rge_alti_wld"
IGN_ALTI_CHUNK    = 200  # points par requête GET (limite URL ~8KB)


def _fetch_elevations_ign_chunk(locations: list) -> Optional[list]:
    """Requête IGN pour un chunk de points (max IGN_ALTI_CHUNK)."""
    lons = "|".join(str(p["longitude"]) for p in locations)
    lats = "|".join(str(p["latitude"])  for p in locations)
    url  = (f"{IGN_ALTI_URL}?lon={lons}&lat={lats}"
            f"&resource={IGN_ALTI_RESOURCE}&delimiter=|&zonly=true")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "snow-conditions/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return [float(z) for z in data["elevations"]]
    except Exception as e:
        print(f"[terrain] IGN chunk erreur: {type(e).__name__}: {e}")
        return None


def _fetch_elevations_ign(locations: list) -> Optional[list]:
    """
    Batch IGN altimétrie REST avec chunking automatique (200 pts/requête).
    locations : liste de dicts {"latitude": y, "longitude": x}
    Retourne liste de floats ou None si erreur sur un chunk.
    """
    all_elevs = []
    for i in range(0, len(locations), IGN_ALTI_CHUNK):
        chunk = locations[i:i + IGN_ALTI_CHUNK]
        result = _fetch_elevations_ign_chunk(chunk)
        if result is None:
            return None
        all_elevs.extend(result)
    print(f"[terrain] IGN Altimétrie REST OK — {len(all_elevs)} points")
    return all_elevs


def _fetch_elevation_ign_single(lat: float, lon: float) -> Optional[float]:
    """Point unique IGN — pour get_terrain_data."""
    result = _fetch_elevations_ign([{"latitude": lat, "longitude": lon}])
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Mode 3 : Open-Elevation (fallback batch)
# ---------------------------------------------------------------------------

def _fetch_elevations_open_elevation(locations: list) -> Optional[list]:
    try:
        payload = json.dumps({"locations": locations}).encode()
        req = urllib.request.Request(
            "https://api.open-elevation.com/api/v1/lookup",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return [float(r["elevation"]) for r in data["results"]]
    except Exception as e:
        print(f"[terrain] Open-Elevation erreur: {e}")
        return None


# ---------------------------------------------------------------------------
# Fallback statique
# ---------------------------------------------------------------------------

def _static_estimate(lat: float, lon: float) -> TerrainPoint:
    return TerrainPoint(lat=lat, lon=lon, elevation_m=1500.0,
                        aspect_deg=0.0, slope_deg=15.0)


# ---------------------------------------------------------------------------
# Point d'entrée principal (point unique)
# ---------------------------------------------------------------------------

def get_terrain_data(lat: float, lon: float, tiff_path: Optional[str] = None) -> TerrainPoint:
    # 1. GeoTIFF local
    if tiff_path:
        result = _extract_from_geotiff(tiff_path, lat, lon)
        if result:
            print(f"[terrain] GeoTIFF {lat:.4f},{lon:.4f} → {result.aspect_deg:.0f}°")
            return result

    # 2. IGN Altimétrie REST (point unique + 8 voisins pour Horn)
    delta = 0.001
    locations = [
        {"latitude": round(lat + dy * delta, 6), "longitude": round(lon + dx * delta, 6)}
        for dy in (1, 0, -1) for dx in (-1, 0, 1)
    ]
    elevs = _fetch_elevations_ign(locations)
    if elevs and len(elevs) == 9:
        import numpy as np
        z = np.array(elevs).reshape(3, 3)
        aspect, slope = _compute_aspect_slope(z, delta * 111_000)
        print(f"[terrain] IGN Alti {lat:.4f},{lon:.4f} → {aspect:.0f}°")
        return TerrainPoint(lat=lat, lon=lon, elevation_m=float(z[1, 1]),
                            aspect_deg=aspect, slope_deg=slope)

    # 3. Open-Elevation
    elevs = _fetch_elevations_open_elevation(locations)
    if elevs and len(elevs) == 9:
        import numpy as np
        z = np.array(elevs).reshape(3, 3)
        aspect, slope = _compute_aspect_slope(z, delta * 111_000)
        print(f"[terrain] Open-Elev {lat:.4f},{lon:.4f} → {aspect:.0f}°")
        return TerrainPoint(lat=lat, lon=lon, elevation_m=float(z[1, 1]),
                            aspect_deg=aspect, slope_deg=slope)

    # 4. Statique
    print(f"[terrain] STATIC {lat:.4f},{lon:.4f}")
    return _static_estimate(lat, lon)


# ---------------------------------------------------------------------------
# Grille — IGN Altimétrie REST batch + fallback Open-Elevation
# ---------------------------------------------------------------------------

def get_terrain_grid(
    lat_min: float, lon_min: float,
    lat_max: float, lon_max: float,
    resolution_m: float = 500,
    tiff_path: Optional[str] = None,
    padding_m: float = 0,
) -> List[TerrainPoint]:
    import numpy as np

    pad_lat = padding_m / 111_000
    pad_lon = padding_m / (111_000 * math.cos(math.radians((lat_min + lat_max) / 2)))
    lat_min_ext = lat_min - pad_lat
    lat_max_ext = lat_max + pad_lat
    lon_min_ext = lon_min - pad_lon
    lon_max_ext = lon_max + pad_lon

    delta_lat = resolution_m / 111_000
    delta_lon = resolution_m / (111_000 * math.cos(math.radians((lat_min + lat_max) / 2)))
    delta_elev = 0.001  # ~100m pour voisinage Horn

    # Construire la liste des centres
    centers = []
    lat = lat_min_ext
    while lat <= lat_max_ext + 1e-9:
        lon = lon_min_ext
        while lon <= lon_max_ext + 1e-9:
            clat = round(lat, 6)
            clon = round(lon, 6)
            is_padding = not (lat_min <= clat <= lat_max and lon_min <= clon <= lon_max)
            centers.append((clat, clon, is_padding))
            lon += delta_lon
        lat += delta_lat

    # Construire tous les points à interroger (centre + 8 voisins)
    all_locations = []
    for (clat, clon, _) in centers:
        for dy in (1, 0, -1):
            for dx in (-1, 0, 1):
                all_locations.append({
                    "latitude":  round(clat + dy * delta_elev, 6),
                    "longitude": round(clon + dx * delta_elev, 6),
                })

    print(f"[terrain] grille: {len(centers)} points, {len(all_locations)} requêtes elevation")

    # Tentative IGN Altimétrie REST
    elevations = _fetch_elevations_ign(all_locations)

    if elevations is None or len(elevations) != len(all_locations):
        print("[terrain] fallback Open-Elevation batch")
        elevations = _fetch_elevations_open_elevation(all_locations)

    if elevations is None or len(elevations) != len(all_locations):
        print("[terrain] fallback statique")
        elevations = [1500.0] * len(all_locations)

    # Calculer aspect/slope pour chaque centre
    cell_m = delta_elev * 111_000
    points = []
    for i, (clat, clon, is_padding) in enumerate(centers):
        block = elevations[i*9 : i*9 + 9]
        z = np.array(block).reshape(3, 3)
        elev = z[1, 1]
        aspect, slope = _compute_aspect_slope(z, cell_m)
        points.append(TerrainPoint(
            lat=clat, lon=clon,
            elevation_m=float(elev),
            aspect_deg=aspect,
            slope_deg=slope,
            is_padding=is_padding
        ))

    return points


# ---------------------------------------------------------------------------
# Démo / test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== terrain.py — test standalone ===\n")
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
        print()
