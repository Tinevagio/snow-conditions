#!/usr/bin/env python3
"""
massif_locator.py
=================
Module utilitaire : retrouve l'id BERA d'un massif à partir de coordonnées GPS.
Utilise data/massif_polygons.json généré par build_massif_polygons.py.

Usage dans bera_corrector.py :
    from massif_locator import MassifLocator
    locator = MassifLocator("data/massif_polygons.json")
    massif_id = locator.find(lat=45.35, lon=5.88)  # → 8 (Belledonne)
"""

import json
import math
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Overrides géographiques explicites
# Certains polygones OSM débordent sur des massifs BERA voisins.
# Ces zones bbox forcent le bon massif, prioritaire sur tout le reste.
#
# Format : (lat_min, lat_max, lon_min, lon_max, massif_id)
# ──────────────────────────────────────────────────────────────
GEO_OVERRIDES = [
    # Haute-Tarentaise — haute vallée de l'Isère (Val d'Isère, Tignes)
    # Vanoise OSM déborde sur cette zone
    (45.10, 45.38, 6.65, 7.05, 6),
    # Haute-Tarentaise — Les Arcs, Bourg-St-Maurice, La Plagne
    # Beaufortain et Vanoise OSM débordent sur cette zone
    (45.45, 45.72, 6.35, 6.75, 6),
    # Ubaye — vallée de l'Ubaye (Barcelonnette)
    # Mercantour OSM s'étend trop loin vers l'ouest
    (44.25, 44.52, 6.50, 7.00, 21),
    # Belledonne — Chamrousse et versant est de Grenoble
    # Chartreuse OSM déborde sur le bord est
    (45.05, 45.42, 5.85, 6.00, 8),
    # Aure-Louron — Saint-Lary-Soulan et vallée d'Aure
    # Le polygone OSM id=67 est le Plantaurel (faux match) ; cet override corrige
    (42.68, 42.95, 0.22, 0.65, 67),
]

# Corrections de centroïdes appliquées à la volée (surcharge le JSON)
# Utilisé pour les massifs dont le centroïde par défaut est trop éloigné du terrain
CENTROID_CORRECTIONS = {
    15: [44.92, 6.22],  # Oisans : La Grave / Bourg-d'Oisans (était trop à l'ouest)
    67: [42.82, 0.40],  # Aure-Louron : Saint-Lary-Soulan (remplace faux polygone Plantaurel)
    73: [42.68, 2.05],  # Capcir-Puymorens : Formiguères / Matemale (était trop au sud)
    74: [42.50, 2.12],  # Cerdagne-Canigou : Font-Romeu / Err
}


def _dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """Ray-casting algorithm. polygon : [[lat, lon], ...]"""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][1], polygon[i][0]
        xj, yj = polygon[j][1], polygon[j][0]
        if ((yi > lon) != (yj > lon)) and (lat < (xj - xi) * (lon - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _dist_to_polygon(lat: float, lon: float, polygon: list) -> float:
    return min(_dist_km(lat, lon, p[0], p[1]) for p in polygon)


class MassifLocator:
    """
    Trouve le massif BERA le plus approprié pour des coordonnées GPS.

    Priorités :
    0. Override géographique explicite (corrige les chevauchements OSM connus)
    1. Point strictement dans un polygone OSM
    2. Plusieurs polygones → le plus petit
    3. Bord de polygone à moins de 15km
    4. Distance centroïde (polygone OSM ou fallback)
    """

    def __init__(self, polygons_path: str = "data/massif_polygons.json"):
        path = Path(polygons_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{polygons_path} introuvable — lance d'abord build_massif_polygons.py"
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._polygons  = []
        self._centroids = []

        for entry in data:
            # Appliquer les corrections de centroïdes
            if entry["id"] in CENTROID_CORRECTIONS:
                entry["source"]   = "centroid_fallback"
                entry["centroid"] = CENTROID_CORRECTIONS[entry["id"]]
                entry.pop("polygon", None)

            if entry.get("source") == "osm" and entry.get("polygon"):
                lats = [p[0] for p in entry["polygon"]]
                lons = [p[1] for p in entry["polygon"]]
                entry["_bbox"]     = (min(lats), max(lats), min(lons), max(lons))
                entry["_centroid"] = (sum(lats) / len(lats), sum(lons) / len(lons))
                self._polygons.append(entry)
            else:
                self._centroids.append(entry)

    def find(self, lat: float, lon: float) -> int:
        """Retourne l'id BERA du massif correspondant aux coordonnées."""

        # ── Priorité 0 : overrides explicites ────────────────
        for la_min, la_max, lo_min, lo_max, mid in GEO_OVERRIDES:
            if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
                return mid

        # ── Priorité 1-3 : polygones OSM ─────────────────────
        candidates_in   = []
        candidates_near = []

        for entry in self._polygons:
            la_min, la_max, lo_min, lo_max = entry["_bbox"]
            if not (la_min - 0.05 <= lat <= la_max + 0.05
                    and lo_min - 0.05 <= lon <= lo_max + 0.05):
                continue

            if _point_in_polygon(lat, lon, entry["polygon"]):
                candidates_in.append(entry)
            else:
                d = _dist_to_polygon(lat, lon, entry["polygon"])
                if d < 15:
                    candidates_near.append((d, entry))

        if len(candidates_in) == 1:
            return candidates_in[0]["id"]

        if len(candidates_in) > 1:
            def bbox_area(e):
                la_min, la_max, lo_min, lo_max = e["_bbox"]
                return (la_max - la_min) * (lo_max - lo_min)
            return min(candidates_in, key=bbox_area)["id"]

        if candidates_near:
            return min(candidates_near, key=lambda x: x[0])[1]["id"]

        # ── Priorité 4 : centroïde le plus proche ─────────────
        all_entries = self._polygons + self._centroids

        def dist_to_entry(e):
            if e.get("source") == "osm":
                c = e["_centroid"]
                return _dist_km(lat, lon, c[0], c[1])
            return _dist_km(lat, lon, e["centroid"][0], e["centroid"][1])

        return min(all_entries, key=dist_to_entry)["id"]

    def find_with_info(self, lat: float, lon: float) -> dict:
        """Comme find() mais retourne aussi le nom et la méthode utilisée."""
        # Détecter si override
        source = "polygon_or_centroid"
        for la_min, la_max, lo_min, lo_max, mid in GEO_OVERRIDES:
            if la_min <= lat <= la_max and lo_min <= lon <= lo_max:
                source = "geo_override"
                break

        massif_id = self.find(lat, lon)
        all_entries = self._polygons + self._centroids
        entry = next((e for e in all_entries if e["id"] == massif_id), None)

        return {
            "id":     massif_id,
            "massif": entry["massif"] if entry else "?",
            "source": source if source == "geo_override" else entry.get("source", "?"),
        }


# ──────────────────────────────────────────────────────────────
# Test si lancé directement
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    polygons_file = sys.argv[1] if len(sys.argv) > 1 else "data/massif_polygons.json"

    locator = MassifLocator(polygons_file)

    tests = [
        (45.92, 6.87,  "Chamonix",           3,  "Mont-Blanc"),
        (45.05, 6.65,  "Valloire",            13, "Thabor"),
        (44.66, 6.63,  "Vars",                20, "Embrunais-Parpaillon"),
        (42.80, 0.62,  "St-Lary",             67, "Aure-Louron"),
        (45.35, 5.88,  "Chamrousse",          8,  "Belledonne"),
        (45.18, 6.72,  "Val-d-Isère",         6,  "Haute-Tarentaise"),
        (44.45, 6.85,  "Barcelonnette",       21, "Ubaye"),
        (45.55, 6.35,  "Les Arcs",            6,  "Haute-Tarentaise"),
        (45.00, 5.75,  "Villard-de-Lans",     14, "Vercors"),
        (42.54, 2.10,  "Font-Romeu",          74, "Cerdagne-Canigou"),
        (45.48, 6.62,  "Bourg-St-Maurice",    6,  "Haute-Tarentaise"),
        (44.92, 6.30,  "La Grave",            15, "Oisans"),
        (45.10, 6.10,  "Alpe-d-Huez",         12, "Grandes-Rousses"),
        (45.39, 6.72,  "Pralognan",           10, "Vanoise"),
        (45.35, 5.72,  "St-Pierre-Chartreuse", 7, "Chartreuse"),
        (44.72, 6.82,  "Saint-Véran",         17, "Queyras"),
        (44.65, 5.88,  "Superdevoluy",        18, "Devoluy"),
        (42.88, 0.10,  "Pic-Midi-Bigorre",    66, "Haute-Bigorre"),
        (42.82, 1.28,  "Saint-Girons",        69, "Couserans"),
    ]

    ok = 0
    for lat, lon, lieu, exp_id, exp_name in tests:
        info = locator.find_with_info(lat, lon)
        status = "✅" if info["id"] == exp_id else "❌"
        if info["id"] == exp_id:
            ok += 1
        print(f"{status} {lieu:26s} → {info['massif']:28s} [{info['source']}]"
              f"  (attendu: {exp_name})")

    print(f"\n{ok}/{len(tests)} corrects")
