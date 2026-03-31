"""
test_avalanche.py
-----------------
Teste le module core/avalanche_model.py en local.

Lancer depuis la racine du repo snow-conditions :
    python test_avalanche.py

Prérequis :
  - data/slope_grids/3.npz  (généré par scripts/build_slope_grids.py --massif 3)
  - data/bera_enneigement.json  (copié depuis Ski-touring-live/data/)
     ou téléchargé depuis GitHub raw si absent

Sortie :
  - Affichage stats dans le terminal
  - test_avalanche_output.geojson  (à ouvrir sur geojson.io pour visualiser)
"""

import json
import sys
import os
import urllib.request
from pathlib import Path

# ── Chemin racine du projet ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Config test ──────────────────────────────────────────────────────────────
MASSIF_ID   = 3                                     # Mont-Blanc
BBOX        = (45.87, 6.85, 45.95, 6.95)           # bbox autour de Chamonix
MAX_ZONES   = 300

BERA_URL    = ("https://raw.githubusercontent.com/Tinevagio/Ski-touring-live"
               "/main/data/bera_enneigement.json")
BERA_LOCAL  = ROOT / "data" / "bera_enneigement.json"
OUTPUT_FILE = ROOT / "test_avalanche_output.geojson"

# ── 0. S'assurer que bera_enneigement.json est disponible ────────────────────
def ensure_bera():
    if BERA_LOCAL.exists():
        print(f"✓ BERA trouvé en local : {BERA_LOCAL}")
        return True
    print(f"⬇ BERA absent en local, téléchargement depuis GitHub…")
    try:
        BERA_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BERA_URL,
              headers={"User-Agent": "snow-conditions/test"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        BERA_LOCAL.write_bytes(content)
        print(f"✓ BERA téléchargé → {BERA_LOCAL}")
        return True
    except Exception as e:
        print(f"✗ Impossible de télécharger le BERA : {e}")
        print(f"  Copiez manuellement le fichier dans : {BERA_LOCAL}")
        return False

# ── 1. Vérifier le .npz ──────────────────────────────────────────────────────
def check_npz():
    npz_path = ROOT / "data" / "slope_grids" / f"{MASSIF_ID}.npz"
    if not npz_path.exists():
        print(f"✗ Grille pentes manquante : {npz_path}")
        print(f"  Lancez : python scripts/build_slope_grids.py --massif {MASSIF_ID}")
        return False

    import numpy as np
    d = np.load(npz_path)
    print(f"✓ Grille pentes chargée : {npz_path}")
    print(f"  Points     : {len(d['lat']):,}")
    print(f"  Elev       : {d['elevation'].min():.0f} → {d['elevation'].max():.0f} m")
    print(f"  Pente moy  : {d['slope'].mean():.1f}° | max : {d['slope'].max():.1f}°")
    print(f"  Pente >30° : {(d['slope']>30).sum():,} pts ({(d['slope']>30).mean()*100:.1f}%)")
    print(f"  Pente >35° : {(d['slope']>35).sum():,} pts ({(d['slope']>35).mean()*100:.1f}%)")
    return True

# ── 2. Tester compute_avalanche_zones ────────────────────────────────────────
def test_full():
    from core.avalanche_model import compute_avalanche_zones, load_bera, load_slope_grid

    print(f"\n{'='*60}")
    print(f"Test compute_avalanche_zones — massif {MASSIF_ID}, bbox {BBOX}")
    print(f"{'='*60}")

    # Test chargement BERA
    bera = load_bera(MASSIF_ID)
    if bera is None:
        print("✗ BERA non chargé")
        return None
    print(f"\n✓ BERA chargé : {bera.massif_name}")
    print(f"  Risque bas/haut : {bera.risque_bas} / {bera.risque_haut}")
    print(f"  Altitude transition : {bera.risque_altitude_m} m")
    print(f"  Limite enneigement N/S : {bera.limite_nord_m} / {bera.limite_sud_m} m")
    pentes_actives = [k for k, v in bera.pentes_dangereuses.items() if v]
    print(f"  Expositions dangereuses : {', '.join(pentes_actives)}")

    # Test calcul complet
    print(f"\nCalcul zones avalanche…")
    import time
    t0 = time.time()
    result = compute_avalanche_zones(
        massif_id=MASSIF_ID,
        bbox=BBOX,
        max_zones=MAX_ZONES,
    )
    elapsed = time.time() - t0

    if result is None:
        print("✗ Résultat None")
        return None

    if "error" in result:
        print(f"✗ Erreur : {result['error']}")
        return None

    props = result.get("properties", {})
    features = result.get("features", [])
    n_starts = props.get("n_start_zones", 0)
    n_cones  = props.get("n_cones", 0)

    print(f"\n✓ Calcul terminé en {elapsed:.2f}s")
    print(f"  Zones de départ : {n_starts}")
    print(f"  Cônes générés   : {n_cones}")

    if n_starts == 0:
        print("\n⚠ Aucune zone de départ trouvée dans cette bbox.")
        print("  Vérifications :")
        print("  - La bbox est-elle dans le massif ?")
        print("  - Le niveau BERA est-il suffisamment élevé ?")
        print("  - Y a-t-il du terrain >30° dans cette zone ?")
        return result

    # Détail des zones de départ
    starts  = [f for f in features if f["properties"]["type"] == "start_zone"]
    cones_f = [f for f in features if f["properties"]["type"] == "cone"]

    slopes  = [f["properties"]["slope_deg"]  for f in starts]
    elevs   = [f["properties"]["elevation"]  for f in starts]
    risques = [f["properties"]["risque"]      for f in starts]

    print(f"\n  Zones de départ :")
    print(f"    Pente moy   : {sum(slopes)/len(slopes):.1f}°  "
          f"| min : {min(slopes):.1f}°  | max : {max(slopes):.1f}°")
    print(f"    Altitude    : {min(elevs):.0f} → {max(elevs):.0f} m")
    r_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
    for r in risques: r_counts[r] = r_counts.get(r, 0) + 1
    for r, c in sorted(r_counts.items()):
        if c: print(f"    Risque {r}     : {c} zones")

    # Exemple de cône
    if cones_f:
        ex = cones_f[0]["properties"]
        print(f"\n  Exemple cône :")
        print(f"    Départ      : {ex['start_lat']:.4f}°N, {ex['start_lon']:.4f}°E")
        print(f"    Altitude    : {ex['elevation']} m")
        print(f"    Pente       : {ex['slope_deg']}°")
        print(f"    Risque      : {ex['risque']}")
        print(f"    Longueur    : {ex['cone_length_m']} m")
        print(f"    Angle       : {ex['cone_angle_deg']}°")

    return result

# ── 3. Sauvegarder le GeoJSON ────────────────────────────────────────────────
def save_geojson(result):
    if result is None:
        return
    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n✓ GeoJSON sauvegardé : {OUTPUT_FILE} ({size_kb:.1f} KB)")
    print(f"  → Ouvrez sur https://geojson.io pour visualiser les cônes")

# ── 4. Test bbox élargie (massif entier, sous-échantillonné) ─────────────────
def test_massif_entier():
    from core.avalanche_model import compute_avalanche_zones
    import numpy as np
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"Test massif entier (max 100 zones)")
    print(f"{'='*60}")

    import time
    t0 = time.time()
    result = compute_avalanche_zones(massif_id=MASSIF_ID, bbox=None, max_zones=100)
    elapsed = time.time() - t0

    if result and "error" not in result:
        props = result["properties"]
        print(f"✓ {props['n_start_zones']} zones | {props['n_cones']} cônes | {elapsed:.2f}s")

        # Sauvegarder aussi la version massif entier
        out = ROOT / "test_avalanche_massif_entier.geojson"
        out.write_text(json.dumps(result, indent=2))
        print(f"  → {out}")
    else:
        print(f"✗ Erreur ou résultat vide")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST AVALANCHE MODEL — snow-conditions")
    print("=" * 60)

    # Étape 0 : BERA disponible ?
    if not ensure_bera():
        sys.exit(1)

    # Étape 1 : NPZ valide ?
    if not check_npz():
        sys.exit(1)

    # Étape 2 : test bbox Chamonix
    result = test_full()
    save_geojson(result)

    # Étape 3 : test massif entier
    test_massif_entier()

    print(f"\n{'='*60}")
    print("Tests terminés.")
