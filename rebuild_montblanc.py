"""
rebuild_montblanc.py
--------------------
Recalcule le .npz du Mont-Blanc (id=3) avec une bbox carrée élargie
autour du centroïde (45.85, 6.87) au lieu du polygone OSM restrictif.

Couvre : Aiguilles Rouges, Chamonix, versant italien, Contamines.

Lancer depuis la racine du repo :
    python rebuild_montblanc.py

Remplace data/slope_grids/3.npz
"""

import json
import math
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
MASSIF_ID      = 3
CENTROID_LAT   = 45.85
CENTROID_LON   = 6.87
PAD_DEG        = 0.25        # ±0.25° soit ~56×56km
RESOLUTION_M   = 100
IGN_ALTI_URL   = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
IGN_RESOURCE   = "ign_rge_alti_wld"
IGN_CHUNK_SIZE = 200
RETRY_MAX      = 3
RETRY_DELAY    = 2.0
OUTPUT_PATH    = Path("data/slope_grids/3.npz")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_chunk(locations: list) -> Optional[List[float]]:
    lons = "|".join(f"{p['longitude']:.6f}" for p in locations)
    lats = "|".join(f"{p['latitude']:.6f}"  for p in locations)
    url  = (f"{IGN_ALTI_URL}?lon={lons}&lat={lats}"
            f"&resource={IGN_RESOURCE}&delimiter=|&zonly=true")
    for attempt in range(RETRY_MAX):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "snow-conditions/rebuild"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            elevs = [float(z) for z in data["elevations"]]
            if len(elevs) == len(locations):
                return elevs
        except Exception as e:
            if attempt < RETRY_MAX - 1:
                print(f"  ↻ retry {attempt+1} ({e})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ✗ chunk échoué : {e}")
    return None


def fetch_batch(locations: list) -> Optional[List[float]]:
    all_elevs = []
    n_chunks  = math.ceil(len(locations) / IGN_CHUNK_SIZE)
    for i in range(0, len(locations), IGN_CHUNK_SIZE):
        chunk_idx = i // IGN_CHUNK_SIZE + 1
        result    = fetch_chunk(locations[i:i + IGN_CHUNK_SIZE])
        if result is None:
            return None
        all_elevs.extend(result)
        if chunk_idx % 20 == 0 or chunk_idx == n_chunks:
            print(f"  → chunk {chunk_idx}/{n_chunks} ({chunk_idx/n_chunks*100:.0f}%) — {len(all_elevs)} pts")
        time.sleep(0.05)
    return all_elevs


def compute_aspect_slope(z: np.ndarray, cell_m: float) -> Tuple[float, float]:
    dz_dx = ((z[0,2]+2*z[1,2]+z[2,2]) - (z[0,0]+2*z[1,0]+z[2,0])) / (8*cell_m)
    dz_dy = ((z[0,0]+2*z[0,1]+z[0,2]) - (z[2,0]+2*z[2,1]+z[2,2])) / (8*cell_m)
    slope  = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = math.degrees(math.atan2(dz_dx, -dz_dy)) % 360 if (dz_dx or dz_dy) else 0.0
    return aspect, slope

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"Rebuild Mont-Blanc (id={MASSIF_ID}) — bbox élargie ±{PAD_DEG}°")
    print("=" * 60)

    pad_lon  = PAD_DEG / math.cos(math.radians(CENTROID_LAT))
    lat_min  = CENTROID_LAT - PAD_DEG
    lat_max  = CENTROID_LAT + PAD_DEG
    lon_min  = CENTROID_LON - pad_lon
    lon_max  = CENTROID_LON + pad_lon

    km_lat   = PAD_DEG * 2 * 111
    km_lon   = pad_lon * 2 * 111 * math.cos(math.radians(CENTROID_LAT))
    print(f"  Bbox     : {lat_min:.3f}→{lat_max:.3f}N, {lon_min:.3f}→{lon_max:.3f}E")
    print(f"  Taille   : {km_lat:.0f} × {km_lon:.0f} km")

    delta_lat = RESOLUTION_M / 111_000
    delta_lon = RESOLUTION_M / (111_000 * math.cos(math.radians(CENTROID_LAT)))

    n_lat = math.ceil((lat_max - lat_min) / delta_lat) + 1
    n_lon = math.ceil((lon_max - lon_min) / delta_lon) + 1

    # Grille complète (pas de masque polygone — c'est voulu)
    centers = []
    for i_lat in range(n_lat):
        clat = lat_min + i_lat * delta_lat
        if clat > lat_max + 1e-9: break
        for i_lon in range(n_lon):
            clon = lon_min + i_lon * delta_lon
            if clon > lon_max + 1e-9: break
            centers.append((round(clat, 6), round(i_lon, 6), i_lat, i_lon))

    # Reconstruire proprement
    centers = []
    for i_lat in range(n_lat):
        clat = round(lat_min + i_lat * delta_lat, 6)
        if clat > lat_max + 1e-9: break
        for i_lon in range(n_lon):
            clon = round(lon_min + i_lon * delta_lon, 6)
            if clon > lon_max + 1e-9: break
            centers.append((clat, clon, i_lat, i_lon))

    print(f"  Cellules : {len(centers):,}")

    # Cellules à fetcher (centres + voisins)
    to_fetch = set()
    for _, _, i_lat, i_lon in centers:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i_lat + di, i_lon + dj
                if 0 <= ni < n_lat and 0 <= nj < n_lon:
                    to_fetch.add((ni, nj))

    fetch_list = sorted(to_fetch)
    locations  = [
        {"latitude":  round(lat_min + i * delta_lat, 6),
         "longitude": round(lon_min + j * delta_lon, 6)}
        for i, j in fetch_list
    ]
    print(f"  Chunks IGN : {math.ceil(len(locations)/IGN_CHUNK_SIZE)}")
    print(f"  Temps estimé : ~{math.ceil(len(locations)/IGN_CHUNK_SIZE)*0.5/60:.0f} min\n")

    t0    = time.time()
    elevs = fetch_batch(locations)
    if elevs is None:
        print("✗ Fetch IGN échoué")
        raise SystemExit(1)
    print(f"✓ Fetch terminé en {time.time()-t0:.0f}s")

    # Index elevation
    elev_map = {}
    n_nodata = 0
    for (i, j), e in zip(fetch_list, elevs):
        if 0 <= e <= 9000:
            elev_map[(i, j)] = e
        else:
            n_nodata += 1
    if n_nodata:
        print(f"  ⚠ {n_nodata} pts hors données IGN ignorés")

    # Calcul aspect/slope
    out_lat, out_lon, out_elev, out_slope, out_aspect = [], [], [], [], []
    for clat, clon, i_lat, i_lon in centers:
        center_elev = elev_map.get((i_lat, i_lon))
        if center_elev is None:
            continue
        z = np.full((3, 3), center_elev)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                v = elev_map.get((i_lat+di, i_lon+dj))
                if v is not None:
                    z[di+1, dj+1] = v
        aspect, slope = compute_aspect_slope(z, RESOLUTION_M)
        if slope > 80:
            continue
        out_lat.append(clat)
        out_lon.append(clon)
        out_elev.append(center_elev)
        out_slope.append(slope)
        out_aspect.append(aspect)

    # Sauvegarde
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        lat       = np.array(out_lat,    dtype=np.float32),
        lon       = np.array(out_lon,    dtype=np.float32),
        elevation = np.array(out_elev,   dtype=np.float32),
        slope     = np.array(out_slope,  dtype=np.float32),
        aspect    = np.array(out_aspect, dtype=np.float32),
    )
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    slopes  = np.array(out_slope)
    print(f"\n✓ Sauvegardé → {OUTPUT_PATH} ({size_kb:.0f} KB, {len(out_lat):,} pts)")
    print(f"  Elev      : {min(out_elev):.0f} → {max(out_elev):.0f} m")
    print(f"  Pente moy : {slopes.mean():.1f}° | max : {slopes.max():.1f}°")
    print(f"  Pente >30° : {(slopes>30).sum():,} pts ({(slopes>30).mean()*100:.1f}%)")
    print(f"  Pente >35° : {(slopes>35).sum():,} pts ({(slopes>35).mean()*100:.1f}%)")
    print(f"\n  Durée totale : {time.time()-t0:.0f}s")
