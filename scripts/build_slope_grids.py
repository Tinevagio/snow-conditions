"""
scripts/build_slope_grids.py
-----------------------------
Précalcule les grilles de pente + exposition pour chaque massif.
Résolution : 100m. Masque polygone appliqué pour réduire le volume.

Sortie : data/slope_grids/{massif_id}.npz
  Arrays : lat, lon, slope_deg, aspect_deg, elevation_m

Lancer une fois en local :
  python scripts/build_slope_grids.py [--massif 3] [--all]

Dépendances : numpy, urllib (stdlib)
"""

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESOLUTION_M   = 100          # résolution de la grille en mètres
IGN_ALTI_URL   = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
IGN_RESOURCE   = "ign_rge_alti_wld"
IGN_CHUNK_SIZE = 200          # points par requête GET
RETRY_MAX      = 3            # tentatives par chunk
RETRY_DELAY    = 2.0          # secondes entre tentatives

DATA_DIR       = Path(__file__).parent.parent / "data"
POLYGONS_PATH  = DATA_DIR / "massif_polygons.json"
OUTPUT_DIR     = DATA_DIR / "slope_grids"

# ---------------------------------------------------------------------------
# Point-in-polygon (ray casting)
# ---------------------------------------------------------------------------

def point_in_polygon(lat: float, lon: float, polygon: List[List[float]]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        plat, plon = polygon[i][0], polygon[i][1]
        jlat, jlon = polygon[j][0], polygon[j][1]
        if ((plon > lon) != (jlon > lon)) and \
           (lat < (jlat - plat) * (lon - plon) / (jlon - plon) + plat):
            inside = not inside
        j = i
    return inside

# ---------------------------------------------------------------------------
# IGN Altimétrie batch
# ---------------------------------------------------------------------------

def fetch_elevations_chunk(locations: List[dict]) -> Optional[List[float]]:
    """Requête IGN pour un chunk de points (max IGN_CHUNK_SIZE)."""
    lons = "|".join(f"{p['longitude']:.6f}" for p in locations)
    lats = "|".join(f"{p['latitude']:.6f}"  for p in locations)
    url  = (f"{IGN_ALTI_URL}?lon={lons}&lat={lats}"
            f"&resource={IGN_RESOURCE}&delimiter=|&zonly=true")

    for attempt in range(RETRY_MAX):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "snow-conditions/slope-grids"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            elevs = [float(z) for z in data["elevations"]]
            if len(elevs) == len(locations):
                return elevs
            print(f"  ⚠ IGN retourné {len(elevs)} pts au lieu de {len(locations)}")
            return None
        except Exception as e:
            if attempt < RETRY_MAX - 1:
                print(f"  ↻ retry {attempt+1}/{RETRY_MAX} ({e})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ✗ chunk échoué : {e}")
                return None


def fetch_elevations_batch(locations: List[dict]) -> Optional[List[float]]:
    """Batch complet avec chunking et barre de progression."""
    all_elevs = []
    n_chunks  = math.ceil(len(locations) / IGN_CHUNK_SIZE)

    for i in range(0, len(locations), IGN_CHUNK_SIZE):
        chunk_idx = i // IGN_CHUNK_SIZE + 1
        chunk     = locations[i:i + IGN_CHUNK_SIZE]
        result    = fetch_elevations_chunk(chunk)

        if result is None:
            return None

        all_elevs.extend(result)

        # Progression toutes les 20 chunks
        if chunk_idx % 20 == 0 or chunk_idx == n_chunks:
            pct = chunk_idx / n_chunks * 100
            print(f"  → chunk {chunk_idx}/{n_chunks} ({pct:.0f}%) — {len(all_elevs)} pts")

        # Pause légère pour ne pas saturer l'API
        time.sleep(0.05)

    return all_elevs

# ---------------------------------------------------------------------------
# Calcul aspect + slope (Horn 1981) — même algo que terrain.py
# ---------------------------------------------------------------------------

def compute_aspect_slope(z: np.ndarray, cell_size_m: float) -> Tuple[float, float]:
    dz_dx = ((z[0,2] + 2*z[1,2] + z[2,2]) -
              (z[0,0] + 2*z[1,0] + z[2,0])) / (8 * cell_size_m)
    dz_dy = ((z[0,0] + 2*z[0,1] + z[0,2]) -
              (z[2,0] + 2*z[2,1] + z[2,2])) / (8 * cell_size_m)

    slope_deg = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))

    if dz_dx == 0 and dz_dy == 0:
        aspect_deg = 0.0
    else:
        aspect_deg = math.degrees(math.atan2(-dz_dx, dz_dy)) % 360

    return aspect_deg, slope_deg

# ---------------------------------------------------------------------------
# Construction grille d'un massif
# ---------------------------------------------------------------------------

def build_massif_grid(massif: dict) -> Optional[Path]:
    massif_id   = massif["id"]
    massif_name = massif["massif"]
    polygon     = massif.get("polygon", [])

    # Skip si déjà calculé
    out_path = OUTPUT_DIR / f"{massif_id}.npz"
    if out_path.exists():
        print(f"  → déjà calculé ({out_path.stat().st_size//1024} KB), skip")
        return out_path

    # Fallback : bbox carrée autour du centroïde si pas de polygone OSM
    if not polygon:
        centroid = massif.get("centroid")
        if not centroid:
            print(f"  ✗ pas de polygone ni de centroïde")
            return None
        clat, clon = centroid[0], centroid[1]
        pad_lat = 0.25   # ~28km
        pad_lon = 0.25 / math.cos(math.radians(clat))
        # Construire un polygone carré autour du centroïde
        polygon = [
            [clat - pad_lat, clon - pad_lon],
            [clat + pad_lat, clon - pad_lon],
            [clat + pad_lat, clon + pad_lon],
            [clat - pad_lat, clon + pad_lon],
            [clat - pad_lat, clon - pad_lon],
        ]
        print(f"  ⚠ pas de polygone OSM — bbox carrée autour du centroïde ({clat}, {clon})")

    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    mid_lat = (lat_min + lat_max) / 2

    delta_lat = RESOLUTION_M / 111_000
    delta_lon = RESOLUTION_M / (111_000 * math.cos(math.radians(mid_lat)))

    # ── 1. Générer les centres de cellule dans le polygone ──────────────────
    n_lat = math.ceil((lat_max - lat_min) / delta_lat) + 1
    n_lon = math.ceil((lon_max - lon_min) / delta_lon) + 1

    centers = []
    for i_lat in range(n_lat):
        clat = lat_min + i_lat * delta_lat
        if clat > lat_max + 1e-9:
            break
        for i_lon in range(n_lon):
            clon = lon_min + i_lon * delta_lon
            if clon > lon_max + 1e-9:
                break
            if point_in_polygon(clat, clon, polygon):
                centers.append((round(clat, 6), round(clon, 6)))

    if not centers:
        print(f"  ✗ aucun centre dans le polygone")
        return None

    print(f"  {len(centers):,} cellules dans le polygone")

    # ── 2. Construire la grille dense pour Horn (pas besoin de 9 requêtes) ──
    # On fait une grille complète et on calcule Horn par interpolation voisins
    # → 1 requête par cellule au lieu de 9
    # Pour cela on fetche toute la grille bbox dense à delta/2 de décalage
    # et on retrouve les voisins par index.

    # Grille régulière complète (centres uniquement, dans polygone + 1 cellule de marge)
    # On garde une grille indicée pour retrouver les voisins
    lat_grid = [lat_min + i * delta_lat for i in range(n_lat)]
    lon_grid = [lon_min + i * delta_lon for i in range(n_lon)]

    # Masque : quelles cellules fetcher (dans polygone OU voisine d'une cellule dans polygone)
    centers_set = set(centers)
    to_fetch = set()
    for clat, clon in centers:
        i_lat = round((clat - lat_min) / delta_lat)
        i_lon = round((clon - lon_min) / delta_lon)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i_lat + di, i_lon + dj
                if 0 <= ni < n_lat and 0 <= nj < n_lon:
                    to_fetch.add((ni, nj))

    print(f"  {len(to_fetch):,} cellules à fetcher (centres + voisins pour Horn)")

    # ── 3. Fetch IGN pour toutes les cellules nécessaires ───────────────────
    fetch_list   = sorted(to_fetch)
    locations    = [{"latitude": lat_grid[i], "longitude": lon_grid[j]}
                    for i, j in fetch_list]

    print(f"  Fetch IGN ({math.ceil(len(locations)/IGN_CHUNK_SIZE)} chunks)…")
    t0     = time.time()
    elevs  = fetch_elevations_batch(locations)

    if elevs is None:
        print(f"  ✗ fetch IGN échoué")
        return None

    print(f"  ✓ fetch terminé en {time.time()-t0:.0f}s")

    # Index elevation par (i_lat, i_lon) — filtrer les sentinelles IGN
    # IGN retourne -99999, mais aussi des valeurs négatives aberrantes pour
    # les zones hors couverture (lacs, frontières, etc.)
    elev_map = {}
    n_nodata = 0
    for (i, j), e in zip(fetch_list, elevs):
        if e < 0 or e > 9000:  # altitude impossible pour les Alpes/Pyrénées
            n_nodata += 1
        else:
            elev_map[(i, j)] = e
    if n_nodata:
        print(f"  ⚠ {n_nodata} pts hors données IGN ignorés (elev<0 ou >9000m)")

    # ── 4. Calcul aspect/slope pour chaque centre ────────────────────────────
    cell_m = RESOLUTION_M  # taille de cellule en mètres

    out_lat, out_lon, out_elev, out_slope, out_aspect = [], [], [], [], []

    for clat, clon in centers:
        i_lat = round((clat - lat_min) / delta_lat)
        i_lon = round((clon - lon_min) / delta_lon)

        center_elev = elev_map.get((i_lat, i_lon))

        # Ignorer les centres sans élévation valide
        if center_elev is None:
            continue

        # Récupérer la fenêtre 3×3 — remplacer les voisins invalides par le centre
        z = np.full((3, 3), center_elev)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i_lat + di, i_lon + dj
                v = elev_map.get((ni, nj))
                if v is not None:
                    z[di+1, dj+1] = v
                # sinon on garde center_elev → pente nulle sur ce bord

        aspect, slope = compute_aspect_slope(z, cell_m)

        out_lat.append(clat)
        out_lon.append(clon)
        out_elev.append(center_elev)
        out_slope.append(slope)
        out_aspect.append(aspect)

    # ── 5. Sauvegarde .npz ───────────────────────────────────────────────────
    # Filtrer les pentes aberrantes résiduelles (>80° = artefact de calcul)
    valid_mask = [s <= 80.0 for s in out_slope]
    out_lat    = [v for v, ok in zip(out_lat,    valid_mask) if ok]
    out_lon    = [v for v, ok in zip(out_lon,    valid_mask) if ok]
    out_elev   = [v for v, ok in zip(out_elev,   valid_mask) if ok]
    out_slope  = [v for v, ok in zip(out_slope,  valid_mask) if ok]
    out_aspect = [v for v, ok in zip(out_aspect, valid_mask) if ok]
    n_filtered = sum(1 for ok in valid_mask if not ok)
    if n_filtered:
        print(f"  ⚠ {n_filtered} pts avec pente >80° filtrés (artefacts Horn)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{massif_id}.npz"

    np.savez_compressed(
        out_path,
        lat       = np.array(out_lat,    dtype=np.float32),
        lon       = np.array(out_lon,    dtype=np.float32),
        elevation = np.array(out_elev,   dtype=np.float32),
        slope     = np.array(out_slope,  dtype=np.float32),
        aspect    = np.array(out_aspect, dtype=np.float32),
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ sauvegardé → {out_path} ({size_kb:.0f} KB, {len(out_lat):,} pts)")
    return out_path

# ---------------------------------------------------------------------------
# Validation rapide : stats sur la grille générée
# ---------------------------------------------------------------------------

def validate_grid(path: Path):
    d = np.load(path)
    slopes = d["slope"]
    aspects = d["aspect"]
    elevs = d["elevation"]
    print(f"\n  Validation {path.name}:")
    print(f"  Points      : {len(slopes):,}")
    print(f"  Elev        : {elevs.min():.0f} → {elevs.max():.0f} m")
    print(f"  Pente moy   : {slopes.mean():.1f}° | max : {slopes.max():.1f}°")
    print(f"  Pente >30°  : {(slopes>30).sum():,} pts ({(slopes>30).mean()*100:.1f}%)")
    print(f"  Pente >35°  : {(slopes>35).sum():,} pts ({(slopes>35).mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Précalcul grilles pentes/exposition par massif")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--massif", type=int, help="ID du massif à calculer (ex: 3 pour Mont-Blanc)")
    group.add_argument("--all",    action="store_true", help="Calculer tous les massifs")
    args = parser.parse_args()

    with open(POLYGONS_PATH) as f:
        polygons = json.load(f)

    if args.massif:
        massifs = [m for m in polygons if m["id"] == args.massif]
        if not massifs:
            print(f"✗ Massif id={args.massif} introuvable")
            return
    else:
        massifs = [m for m in polygons if m.get("polygon")]

    total_t0 = time.time()

    for m in massifs:
        print(f"\n{'='*60}")
        print(f"Massif {m['id']} — {m['massif']}")
        print(f"{'='*60}")
        t0   = time.time()
        path = build_massif_grid(m)
        if path:
            validate_grid(path)
            print(f"  Durée totale : {time.time()-t0:.0f}s")

    print(f"\n✓ Terminé en {time.time()-total_t0:.0f}s")


if __name__ == "__main__":
    main()
