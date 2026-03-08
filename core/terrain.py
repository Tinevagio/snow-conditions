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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    dz_dx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) -
              (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8 * cell_size_m)
    dz_dy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) -
              (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (8 * cell_size_m)

    slope_rad = math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = math.degrees(slope_rad)

    if dz_dx == 0 and dz_dy == 0:
        aspect_deg = 0.0
    else:
        aspect_rad = math.atan2(dz_dx, -dz_dy)
        aspect_deg = math.degrees(aspect_rad) % 360

    return aspect_deg, slope_deg


# ---------------------------------------------------------------------------
# Mode 1 : lecture d'un fichier GeoTIFF local (rasterio requis)
# ---------------------------------------------------------------------------

def _extract_from_geotiff(tiff_path, lat, lon):
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
            cell_size = (res_x + res_y) / 2
            aspect, slope = _compute_aspect_slope(data, cell_size)
            return TerrainPoint(lat=lat, lon=lon, elevation_m=elev,
                                aspect_deg=aspect, slope_deg=slope)
    except Exception as e:
        print(f"[terrain] erreur GeoTIFF : {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Mode 2 : API WCS IGN
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
                res = abs(src.transform.a)
                cell_m = res * 111_000
                aspect, slope = _compute_aspect_slope(data, cell_m)
                return TerrainPoint(lat=lat, lon=lon, elevation_m=elev,
                                    aspect_deg=aspect, slope_deg=slope)
    except Exception as e:
        print(f"[IGN WCS] erreur: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Mode 3 : Open-Elevation (fallback batch)
# ---------------------------------------------------------------------------

def _estimate_terrain_from_neighbors(lat: float, lon: float) -> Optional[TerrainPoint]:
    try:
        import numpy as np
        delta = 0.001
        points = [
            {"latitude": lat + dy * delta, "longitude": lon + dx * delta}
            for dy in (1, 0, -1) for dx in (-1, 0, 1)
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
        cell_m = delta * 111_000
        aspect, slope = _compute_aspect_slope(z, cell_m)
        return TerrainPoint(lat=lat, lon=lon, elevation_m=elev,
                            aspect_deg=aspect, slope_deg=slope)
    except Exception:
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

def get_terrain_data(lat, lon, tiff_path=None) -> TerrainPoint:
    if tiff_path:
        result = _extract_from_geotiff(tiff_path, lat, lon)
        if result:
            print(f"[terrain] GeoTIFF {lat:.4f},{lon:.4f} → {result.aspect_deg:.0f}°")
            return result

    result = _fetch_from_ign_wcs(lat, lon)
    if result:
        print(f"[terrain] IGN WCS {lat:.4f},{lon:.4f} → {result.aspect_deg:.0f}°")
        return result

    result = _estimate_terrain_from_neighbors(lat, lon)
    if result:
        print(f"[terrain] Open-Elev {lat:.4f},{lon:.4f} → {result.aspect_deg:.0f}°")
        return result

    print(f"[terrain] STATIC {lat:.4f},{lon:.4f}")
    return _static_estimate(lat, lon)


# ---------------------------------------------------------------------------
# Grille — IGN WCS parallélisé + fallback Open-Elevation batch
# ---------------------------------------------------------------------------

def _fetch_ign_for_grid(args):
    """Worker pour ThreadPoolExecutor."""
    clat, clon, is_padding = args
    result = _fetch_from_ign_wcs(clat, clon)
    if result:
        result.is_padding = is_padding
    return (clat, clon, result)


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

    # -----------------------------------------------------------------------
    # Tentative IGN WCS : tester 1 point d'abord
    # -----------------------------------------------------------------------
    test_lat, test_lon, _ = centers[len(centers) // 2]
    test_result = _fetch_from_ign_wcs(test_lat, test_lon)
    use_ign = test_result is not None
    print(f"[terrain] source: {'IGN WCS (parallèle)' if use_ign else 'Open-Elevation (batch)'}")

    if use_ign:
        # Requêtes IGN WCS parallèles (10 threads)
        results_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_ign_for_grid, args): args for args in centers}
            for future in as_completed(futures):
                clat, clon, tp = future.result()
                if tp is not None:
                    results_map[(clat, clon)] = tp

        # Construire la liste finale dans l'ordre des centres
        points = []
        fallback_centers = []
        for clat, clon, is_padding in centers:
            tp = results_map.get((clat, clon))
            if tp:
                points.append(tp)
            else:
                fallback_centers.append((clat, clon, is_padding))

        if fallback_centers:
            print(f"[terrain] IGN WCS : {len(fallback_centers)} points en fallback Open-Elevation")
            fallback_pts = _open_elevation_batch(fallback_centers, delta_lat)
            points.extend(fallback_pts)

        # Remettre dans l'ordre lat/lon
        order = {(clat, clon): i for i, (clat, clon, _) in enumerate(centers)}
        points.sort(key=lambda p: order.get((p.lat, p.lon), 9999))
        return points

    else:
        # Fallback complet Open-Elevation batch
        return _open_elevation_batch(centers, delta_lat)


def _open_elevation_batch(
    centers: list,
    delta_lat: float,
) -> List[TerrainPoint]:
    """Requête batch Open-Elevation pour une liste de centres (clat, clon, is_padding)."""
    import numpy as np

    delta_elev = 0.001  # ~100m pour voisinage Horn

    all_locations = []
    for (clat, clon, _) in centers:
        for dy in (1, 0, -1):
            for dx in (-1, 0, 1):
                all_locations.append({
                    "latitude":  round(clat + dy * delta_elev, 6),
                    "longitude": round(clon + dx * delta_elev, 6),
                })

    try:
        payload = json.dumps({"locations": all_locations}).encode()
        req = urllib.request.Request(
            "https://api.open-elevation.com/api/v1/lookup",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        elevations = [float(r["elevation"]) for r in data["results"]]
    except Exception as e:
        print(f"[terrain] Open-Elevation batch failed: {e}")
        elevations = [1500.0] * len(all_locations)

    cell_m = delta_elev * 111_000
    points = []
    for i, (clat, clon, is_padding) in enumerate(centers):
        block = elevations[i*9 : i*9 + 9]
        z = np.array(block).reshape(3, 3)
        elev = z[1, 1]
        aspect, slope = _compute_aspect_slope(z, cell_m)
        points.append(TerrainPoint(
            lat=clat, lon=clon,
            elevation_m=elev,
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