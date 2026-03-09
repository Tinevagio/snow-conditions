#!/usr/bin/env python3
"""
bera_corrector.py
=================
Recalibre les champs HourlyWeather depuis les données BERA Météo France.

Champs corrigés :
  - snowfall_last_72h    (principal : corrige les erreurs Open-Meteo en altitude)
  - snowfall_last_24h    (secondaire : recalé sur la dernière analyse BERA)
  - hours_above_zero_last_48h  (secondaire : forcé à 0 si altitude sous limite N)

Usage dans main.py :
    from bera_corrector import BeraCorrector
    corrector = BeraCorrector(
        bera_json_path="data/bera_enneigement.json",
        polygons_path="data/massif_polygons.json",
    )
    corrected_hours = corrector.correct(hourly_list, lat, lon, elevation, aspect_deg)

Stratégie de correction snowfall_72h :
  - Si BERA donne 0cm fraîche sur 72h ET Open-Meteo donne >0 → réduire (BERA fait foi)
  - Si BERA donne >0cm ET Open-Meteo donne 0 → augmenter (BERA a des stations réelles)
  - Correction pondérée : alpha = confiance BERA (0.0 à 1.0)
  - Confiance élevée si les données BERA sont récentes (< 2 jours)
  - Confiance réduite si massif en fallback centroïde (moins précis géographiquement)
"""

import json
import math
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# URL raw GitHub pour fetch distant (optionnel)
# Utilisé si les fichiers JSON ne sont pas locaux
# ──────────────────────────────────────────────────────────────
BERA_JSON_GITHUB = (
    "https://raw.githubusercontent.com/Tinevagio/Ski-touring-live/main/data/bera_enneigement.json"
)
POLYGONS_GITHUB = (
    "https://raw.githubusercontent.com/Tinevagio/Ski-touring-live/main/data/massif_polygons.json"
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _aspect_weights(aspect_deg: float) -> tuple[float, float]:
    """
    Retourne (w_nord, w_sud) pour pondérer N_cm et S_cm du BERA
    selon l'exposition du point (0=N, 90=E, 180=S, 270=W).
    """
    a = aspect_deg % 360
    if a <= 45 or a >= 315:
        return 1.0, 0.0          # N
    elif a <= 90:
        t = (a - 45) / 45
        return 1 - t * 0.7, t * 0.7          # NE
    elif a <= 135:
        t = (a - 90) / 45
        return 0.3 - t * 0.3, 0.7 + t * 0.3  # SE
    elif a <= 225:
        return 0.0, 1.0          # S
    elif a <= 270:
        t = (a - 225) / 45
        return t * 0.3, 1 - t * 0.3           # SW
    else:
        t = (a - 270) / 45
        return 0.3 + t * 0.7, 0.7 - t * 0.7  # NW


def _interp_snow_depth(massif: dict, altitude: float, aspect_deg: float) -> Optional[float]:
    """
    Interpole l'épaisseur de neige (cm) à l'altitude donnée selon l'exposition.
    Retourne None si pas de données d'enneigement.
    """
    niveaux = massif.get("enneigement", [])
    if not niveaux:
        return None

    w_n, w_s = _aspect_weights(aspect_deg)
    niveaux_s = sorted(niveaux, key=lambda n: n["alti"])

    def depth_at(n: dict) -> float:
        return w_n * (n.get("N_cm") or 0) + w_s * (n.get("S_cm") or 0)

    if altitude <= niveaux_s[0]["alti"]:
        return 0.0
    if altitude >= niveaux_s[-1]["alti"]:
        return depth_at(niveaux_s[-1])

    for i in range(len(niveaux_s) - 1):
        lo, hi = niveaux_s[i], niveaux_s[i + 1]
        if lo["alti"] <= altitude <= hi["alti"]:
            t = (altitude - lo["alti"]) / (hi["alti"] - lo["alti"])
            return depth_at(lo) + t * (depth_at(hi) - depth_at(lo))
    return None


def _bera_snowfall(massif: dict, target_date: date, n_days: int = 3) -> Optional[float]:
    """
    Cumul neige fraîche BERA sur les n_days précédant target_date (cm).
    Utilise la moyenne (min + max) / 2 par jour.
    Retourne None si aucune donnée disponible.
    """
    nf = massif.get("neige_fraiche", [])
    if not nf:
        return None

    total = 0.0
    found = 0
    for entry in nf:
        try:
            d = date.fromisoformat(entry["date"])
        except (TypeError, ValueError):
            continue
        delta = (target_date - d).days
        if 1 <= delta <= n_days:
            mn = entry.get("min_cm") or 0
            mx = entry.get("max_cm") or 0
            total += (mn + mx) / 2
            found += 1

    return total if found > 0 else None


def _snow_line_for_aspect(massif: dict, aspect_deg: float) -> Optional[float]:
    """
    Altitude d'enneigement continu selon l'exposition.
    Versant nord si aspect ∈ [315,45], sud si ∈ [135,225], intermédiaire sinon.
    """
    w_n, w_s = _aspect_weights(aspect_deg)
    ln = massif.get("limite_nord_m")
    ls = massif.get("limite_sud_m")
    if ln is not None and ls is not None:
        return w_n * ln + w_s * ls
    return ln or ls


def _bera_confidence(massif: dict, source: str) -> float:
    """
    Confiance dans les données BERA (0.0 à 1.0).
    - Réduite si le massif est un centroïde fallback (moins précis)
    - Réduite si les données enneigement sont absentes
    - Maximale si données OSM + enneigement complet
    """
    conf = 1.0
    if source == "centroid_fallback":
        conf *= 0.6   # polygone OSM absent, centroïde approximatif
    if not massif.get("enneigement"):
        conf *= 0.5
    if not massif.get("neige_fraiche"):
        conf *= 0.5
    return conf


# ──────────────────────────────────────────────────────────────
# BeraCorrector
# ──────────────────────────────────────────────────────────────

class BeraCorrector:
    """
    Corrige les champs HourlyWeather en utilisant les données BERA.

    Paramètres :
        bera_json_path   : chemin local vers bera_enneigement.json
        polygons_path    : chemin local vers massif_polygons.json
        use_github       : si True et fichiers locaux absents, fetch depuis GitHub
        alpha_max        : pondération maximale accordée au BERA (défaut 0.8)
                           0.0 = désactive la correction BERA
                           1.0 = BERA remplace complètement Open-Meteo
    """

    def __init__(
        self,
        bera_json_path: str = "data/bera_enneigement.json",
        polygons_path:  str = "data/massif_polygons.json",
        use_github:     bool = True,
        alpha_max:      float = 0.8,
    ):
        self.alpha_max = alpha_max
        self._bera_by_id: dict = {}
        self._locator = None

        # Charger BERA JSON
        self._bera_by_id = self._load_bera(bera_json_path, use_github)

        # Charger MassifLocator
        if self._bera_by_id:
            self._locator = self._load_locator(polygons_path, use_github)

        if not self._bera_by_id or not self._locator:
            logger.warning("BeraCorrector : données BERA indisponibles, correction désactivée")

    # ── Loaders ──────────────────────────────────────────────

    def _load_bera(self, path: str, use_github: bool) -> dict:
        data = self._load_json(path, BERA_JSON_GITHUB if use_github else None)
        if data is None:
            return {}
        return {m["id"]: m for m in data}

    def _load_locator(self, path: str, use_github: bool):
        """Importe MassifLocator dynamiquement (même dossier ou package)."""
        try:
            from massif_locator import MassifLocator
        except ImportError:
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(__file__))
                from massif_locator import MassifLocator
            except ImportError:
                logger.error("massif_locator.py introuvable")
                return None

        # Si le fichier local est absent, télécharger depuis GitHub
        poly_path = Path(path)
        if not poly_path.exists() and use_github:
            poly_path = self._download_to_temp(POLYGONS_GITHUB, "massif_polygons.json")
        if poly_path is None or not poly_path.exists():
            return None

        try:
            return MassifLocator(str(poly_path))
        except Exception as e:
            logger.error(f"MassifLocator init error: {e}")
            return None

    def _load_json(self, local_path: str, github_url: Optional[str]) -> Optional[list]:
        p = Path(local_path)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erreur lecture {local_path}: {e}")

        if github_url:
            try:
                import urllib.request
                with urllib.request.urlopen(github_url, timeout=10) as r:
                    return json.loads(r.read())
            except Exception as e:
                logger.warning(f"Fetch GitHub échoué ({github_url}): {e}")
        return None

    def _download_to_temp(self, url: str, filename: str) -> Optional[Path]:
        try:
            import urllib.request, tempfile
            tmp = Path(tempfile.gettempdir()) / filename
            urllib.request.urlretrieve(url, tmp)
            return tmp
        except Exception as e:
            logger.warning(f"Download échoué {url}: {e}")
            return None

    # ── API publique ──────────────────────────────────────────

    def correct(
        self,
        hourly_list: list,
        lat: float,
        lon: float,
        elevation: float,
        aspect_deg: float,
    ) -> list:
        """
        Corrige une liste d'objets HourlyWeather pour un point GPS donné.

        Paramètres :
            hourly_list : liste de HourlyWeather (modifiés in-place ET retournés)
            lat, lon    : coordonnées GPS du point
            elevation   : altitude en mètres
            aspect_deg  : exposition en degrés (0=N, 90=E, 180=S, 270=W)

        Retourne la liste corrigée (même objets, champs modifiés).
        """
        if not self._bera_by_id or not self._locator or not hourly_list:
            return hourly_list

        # ── Trouver le massif BERA ────────────────────────────
        try:
            info = self._locator.find_with_info(lat, lon)
        except Exception as e:
            logger.warning(f"MassifLocator error: {e}")
            return hourly_list

        massif_id = info["id"]
        massif    = self._bera_by_id.get(massif_id)
        if massif is None:
            logger.debug(f"Massif {massif_id} absent du BERA JSON")
            return hourly_list

        source     = info.get("source", "centroid_fallback")
        confidence = _bera_confidence(massif, source)
        alpha      = self.alpha_max * confidence

        # ── Date de référence (premier hour de la liste) ──────
        try:
            ref_date = date.fromisoformat(str(hourly_list[0].hour)[:10])
        except Exception:
            ref_date = date.today()

        # ── Données BERA pour ce point ────────────────────────
        bera_72h  = _bera_snowfall(massif, ref_date, n_days=3)
        bera_24h  = _bera_snowfall(massif, ref_date, n_days=1)
        snow_line = _snow_line_for_aspect(massif, aspect_deg)

        # Log debug
        logger.debug(
            f"BERA [{massif_id} {massif['massif']}] "
            f"conf={confidence:.2f} α={alpha:.2f} "
            f"72h={bera_72h}cm 24h={bera_24h}cm "
            f"snow_line={snow_line}m"
        )

        # ── Appliquer les corrections à chaque heure ──────────
        for hw in hourly_list:
            self._correct_hour(hw, elevation, aspect_deg, alpha,
                               bera_72h, bera_24h, snow_line, massif, ref_date)

        return hourly_list

    def _correct_hour(
        self,
        hw,
        elevation: float,
        aspect_deg: float,
        alpha: float,
        bera_72h: Optional[float],
        bera_24h: Optional[float],
        snow_line: Optional[float],
        massif: dict,
        ref_date: date,
    ):
        """Corrige un seul HourlyWeather in-place."""

        # ── 1. Pas de neige sous la limite d'enneigement ─────
        if snow_line is not None and elevation < snow_line:
            hw.snowfall_last_72h = 0.0
            hw.snowfall_last_24h = 0.0
            return  # pas besoin d'aller plus loin

        # ── 2. Correction snowfall_last_72h ───────────────────
        if bera_72h is not None:
            om_72h = hw.snowfall_last_72h or 0.0
            # Fusion pondérée : (1-α)×Open-Meteo + α×BERA
            corrected_72h = (1 - alpha) * om_72h + alpha * bera_72h
            hw.snowfall_last_72h = round(corrected_72h, 1)

        # ── 3. Correction snowfall_last_24h ───────────────────
        if bera_24h is not None:
            om_24h = hw.snowfall_last_24h or 0.0
            corrected_24h = (1 - alpha) * om_24h + alpha * bera_24h
            hw.snowfall_last_24h = round(corrected_24h, 1)

        # ── 4. hours_above_zero_last_48h ─────────────────────
        # Si le point est bien au-dessus de la limite d'enneigement
        # et le BERA indique de la neige froide en altitude,
        # on peut corriger les faux positifs de dégel calculés depuis la vallée.
        if snow_line is not None and elevation > snow_line + 500:
            qualite = massif.get("qualite_texte") or ""
            keywords_froid = ["froide", "froid", "poudre", "poudreuse", "regelée"]
            if any(kw in qualite.lower() for kw in keywords_froid):
                # Réduire hours_above_zero si le BERA décrit de la neige froide
                if hw.hours_above_zero_last_48h is not None:
                    hw.hours_above_zero_last_48h = min(
                        hw.hours_above_zero_last_48h,
                        int(hw.hours_above_zero_last_48h * (1 - alpha * 0.5))
                    )

    # ── Méthode utilitaire pour debug / tests ─────────────────

    def get_massif_info(self, lat: float, lon: float) -> dict:
        """Retourne les infos BERA brutes pour un point GPS (debug)."""
        if not self._locator:
            return {}
        info      = self._locator.find_with_info(lat, lon)
        massif    = self._bera_by_id.get(info["id"], {})
        snow_line = _snow_line_for_aspect(massif, 0)
        return {
            "massif_id":          info["id"],
            "massif_name":        info["massif"],
            "source":             info["source"],
            "confidence":         _bera_confidence(massif, info["source"]),
            "alpha":              self.alpha_max * _bera_confidence(massif, info["source"]),
            "bera_date":          massif.get("date_enneigement"),
            "limite_nord_m":      massif.get("limite_nord_m"),
            "limite_sud_m":       massif.get("limite_sud_m"),
            "bera_72h_cm":        _bera_snowfall(massif, date.today(), 3),
            "bera_24h_cm":        _bera_snowfall(massif, date.today(), 1),
            "enneigement_niveaux": massif.get("enneigement", []),
            "pentes_dangereuses": massif.get("pentes_dangereuses", {}),
            "qualite_texte":      massif.get("qualite_texte", ""),
        }


# ──────────────────────────────────────────────────────────────
# Test autonome
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    bera_path     = sys.argv[1] if len(sys.argv) > 1 else "data/bera_enneigement.json"
    polygons_path = sys.argv[2] if len(sys.argv) > 2 else "data/massif_polygons.json"

    corrector = BeraCorrector(
        bera_json_path=bera_path,
        polygons_path=polygons_path,
        use_github=False,
    )

    # Quelques points de test
    test_points = [
        (45.92, 6.87, 2400, 0,   "Chamonix N 2400m"),
        (45.35, 5.88, 1800, 180, "Chamrousse S 1800m"),
        (45.18, 6.72, 2800, 45,  "Val-d-Isère NE 2800m"),
        (44.45, 6.85, 2000, 270, "Barcelonnette W 2000m"),
        (42.80, 0.62, 2200, 0,   "St-Lary N 2200m"),
    ]

    print("🔍 BeraCorrector — test get_massif_info\n")
    for lat, lon, elev, asp, label in test_points:
        info = corrector.get_massif_info(lat, lon)
        snow_depth = _interp_snow_depth(
            corrector._bera_by_id.get(info["massif_id"], {}), elev, asp
        )
        print(f"📍 {label}")
        print(f"   Massif : {info['massif_name']} [{info['source']}] conf={info['confidence']:.2f}")
        print(f"   Limite enneig. N/S : {info['limite_nord_m']}m / {info['limite_sud_m']}m")
        print(f"   Neige fraîche BERA 72h : {info['bera_72h_cm']}cm")
        print(f"   Épaisseur interpolée @{elev}m asp={asp}° : {snow_depth:.0f}cm" if snow_depth else "   Pas de données enneigement")
        print()
