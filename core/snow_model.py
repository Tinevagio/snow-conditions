"""
snow_model.py
-------------
Moteur physique central de prédiction des conditions de neige.
MVP — 5 états principaux, grille 500m.

Aucune dépendance externe : testable sans API.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
import math
from .solar_radiation import effective_radiation




# ---------------------------------------------------------------------------
# ENUMS & DATACLASSES
# ---------------------------------------------------------------------------

class SnowCondition(Enum):
    POWDER_COLD   = "poudre_froide"       # neige fraîche, froide, légère
    POWDER_WARM   = "poudre_rechauffee"   # neige fraîche mais surface > -3°C
    SPRING_SNOW   = "neige_de_printemps"  # transformée, agréable à skier
    CRUST         = "croute_regel"        # regel nocturne, surface dure
    WET_HEAVY     = "neige_humide"        # détrempée, lourde, dangereuse
    OLD_PACKED    = "neige_ancienne_tassee" # vieille neige tassée
    NO_SNOW       = "pas_de_neige"        # pas de neige

    def label(self) -> str:
        labels = {
            "poudre_froide":       "Poudre froide",
            "poudre_rechauffee":   "Poudre réchauffée",
            "neige_de_printemps":  "Neige de printemps",
            "croute_regel":        "Croûte de regel",
            "neige_humide":        "Neige humide lourde",
            "neige_ancienne_tassee":"Neige ancienne tassée",
            "pas_de_neige":        "Pas de neige",
        }
        return labels[self.value]

    def color(self) -> str:
        colors = {
            "poudre_froide":       "#4A90D9",   # bleu vif
            "poudre_rechauffee":   "#7FB3E8",   # bleu clair
            "neige_de_printemps":  "#F5A623",   # orange
            "croute_regel":        "#D0021B",   # rouge
            "neige_humide":        "#8B572A",   # marron
            "neige_ancienne_tassee":"#B8D4F0",  #bleu pale
            "pas_de_neige":        "#c8bfb0",   #beige / gris neutre
        }
        return colors[self.value]

    def ski_quality(self) -> int:
        """Score de qualité ski de 1 (mauvais) à 5 (excellent)"""
        scores = {
            "poudre_froide":       5,
            "poudre_rechauffee":   4,
            "neige_de_printemps":  3,
            "croute_regel":        2,
            "neige_humide":        1,
            "neige_ancienne_tassee":2,
            "pas_de_neige":        0,
        }
        return scores[self.value]


@dataclass
class GridPoint:
    """Un point de la grille de calcul."""
    lat: float
    lon: float
    elevation: float        # altitude en mètres
    aspect: float           # exposition en degrés (0=N, 90=E, 180=S, 270=O)
    slope: float            # inclinaison en degrés


@dataclass
class HourlyWeather:
    """Données météo pour une heure donnée à un point de référence."""
    hour: int                           # heure UTC (0-23)
    temperature_2m: float               # °C à 2m du sol
    reference_elevation: float          # altitude du point météo de référence
    wind_speed: float                   # km/h
    shortwave_radiation: float          # W/m² — rayonnement solaire global
    snowfall_last_24h: float            # cm de neige fraîche sur 24h
    snowfall_last_72h: float            # cm de neige fraîche sur 72h
    hours_above_zero_last_48h: int      # nb d'heures > 0°C sur les 48h passées
    hours_below_minus2_last_12h: int    # nb d'heures < -2°C sur les 12h passées
    direct_radiation:float              # radiation solaire


@dataclass
class SnowResult:
    """Résultat pour un point et une heure."""
    lat: float
    lon: float
    elevation: float
    aspect: float
    hour: int
    condition: SnowCondition
    temp_surface: float     # température effective en surface (corrigée)
    ski_quality: int        # score 1-5
    wind_speed: float = 0.0


# ---------------------------------------------------------------------------
# CORRECTIONS PHYSIQUES
# ---------------------------------------------------------------------------

def adjust_temperature_for_elevation(
    temp_ref: float,
    elevation_point: float,
    elevation_ref: float
) -> float:
    """
    Gradient thermique standard : -0.6°C par 100m de dénivelé.
    """
    delta_elevation = elevation_point - elevation_ref
    return temp_ref - (delta_elevation * 0.006)




def compute_surface_temperature(point, weather, month, day):
    temp_corrected = adjust_temperature_for_elevation(
        weather.temperature_2m,
        point.elevation,
        weather.reference_elevation
    )
    rad = effective_radiation(
        hour_utc   = weather.hour,
        lat        = point.lat,
        lon        = point.lon,
        month      = month,
        day        = day,
        aspect     = point.aspect,
        slope      = point.slope,
        altitude_m = point.elevation
    )
    return temp_corrected + rad.temperature_correction

# ---------------------------------------------------------------------------
# RÈGLES PHYSIQUES — CLASSIFIEUR PRINCIPAL
# ---------------------------------------------------------------------------

def classify_snow_condition(
    point: GridPoint,
    weather: HourlyWeather,
    month: int,
    day: int
) -> tuple[SnowCondition, float]:
    """
    Applique les règles physiques et retourne (SnowCondition, temp_surface).

    Ordre des règles : du plus contraignant au plus général.
    La première règle qui matche l'emporte.
    """
    temp_surface = compute_surface_temperature(point, weather, month, day)
    fresh_snow   = weather.snowfall_last_72h
    
    # ------------------------------------------------------------------
    # RÈGLE 0 — PAS DE NEIGE
    # En dessous du seuil d'enneigement climatologique pour les Alpes
    # ------------------------------------------------------------------
    snow_line = max(800, 1800 - (month - 1) * 80)  # ~1800m en janvier, ~1100m en juin
    if (point.elevation < snow_line
            and weather.snowfall_last_72h == 0
            and temp_surface > 2):
        return SnowCondition.NO_SNOW, temp_surface
    
    

    # ------------------------------------------------------------------
    # RÈGLE 1 — CROÛTE DE REGEL
    # Nuit froide après période de redoux → surface gelée dure
    # ------------------------------------------------------------------
    if (weather.hours_below_minus2_last_12h >= 4
            and weather.hours_above_zero_last_48h >= 2
            and temp_surface < 0):
        return SnowCondition.CRUST, temp_surface

    # ------------------------------------------------------------------
    # RÈGLE 2 — NEIGE HUMIDE LOURDE
    # Surface trop chaude, neige détrempée
    # ------------------------------------------------------------------
    if (temp_surface > 1.5
            and weather.hours_above_zero_last_48h >= 6):
        return SnowCondition.WET_HEAVY, temp_surface

    # ------------------------------------------------------------------
    # RÈGLE 3 — POUDRE FROIDE
    # Neige fraîche abondante + froid + pas de vent violent
    # ------------------------------------------------------------------
    if (fresh_snow >= 15
            and temp_surface < -3
            and weather.wind_speed < 30):
        return SnowCondition.POWDER_COLD, temp_surface

    # ------------------------------------------------------------------
    # RÈGLE 4 — POUDRE RÉCHAUFFÉE
    # Neige fraîche mais température de surface modérée
    # ------------------------------------------------------------------
    if (fresh_snow >= 10
            and -3 <= temp_surface <= 1
            and weather.wind_speed < 35):
        return SnowCondition.POWDER_WARM, temp_surface

    # ------------------------------------------------------------------
    # RÈGLE 5 — NEIGE DE PRINTEMPS
    # Neige ancienne, surface positive, transformation en cours
    # ------------------------------------------------------------------
    if (0 <= temp_surface <= 1.5
            and weather.hours_above_zero_last_48h >= 3
            and fresh_snow < 10
            and weather.direct_radiation > 50):
        return SnowCondition.SPRING_SNOW, temp_surface

    # ------------------------------------------------------------------
    # DÉFAUT — cas non couvert → on choisit selon température
    # ------------------------------------------------------------------
    if temp_surface < -2:
        return SnowCondition.POWDER_COLD, temp_surface
    elif temp_surface <= 0:
        return SnowCondition.OLD_PACKED, temp_surface
    else:
        return SnowCondition.WET_HEAVY, temp_surface


# ---------------------------------------------------------------------------
# ORCHESTRATEUR PRINCIPAL
# ---------------------------------------------------------------------------

def compute_snow_conditions(
    grid_points: List[GridPoint],
    hourly_weather: List[HourlyWeather],
    month: int,
    day: int
) -> List[SnowResult]:
    """
    Point d'entrée principal du moteur.

    Pour chaque point de la grille et chaque heure,
    calcule et retourne l'état de neige prédit.

    Args:
        grid_points    : liste de points GridPoint (grille 500m)
        hourly_weather : liste de HourlyWeather (une entrée par heure)

    Returns:
        Liste de SnowResult — une entrée par (point x heure)
    """
    results = []

    for point in grid_points:
        for weather in hourly_weather:
            condition, temp_surface = classify_snow_condition(point, weather, month, day)

            results.append(SnowResult(
                lat=point.lat,
                lon=point.lon,
                elevation=point.elevation,
                aspect=point.aspect,
                hour=weather.hour,
                condition=condition,
                temp_surface=round(temp_surface, 1),
                ski_quality=condition.ski_quality(),
                wind_speed=round(weather.wind_speed, 1)
            ))

    return results


# ---------------------------------------------------------------------------
# UTILITAIRE — CRÉATION DE GRILLE MOCK (sans MNT réel)
# ---------------------------------------------------------------------------

def create_mock_grid(
    lat_min: float, lon_min: float,
    lat_max: float, lon_max: float,
    resolution_m: int = 500
) -> List[GridPoint]:
    """
    Génère une grille de points sur la bbox.
    Version mock MVP : élévation et aspect fixes.
    Sera remplacé par les vraies données MNT IGN dans terrain.py.

    1° de latitude ~ 111km -> 500m ~ 0.0045°
    """
    step = resolution_m / 111_000
    points = []

    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            points.append(GridPoint(
                lat=round(lat, 6),
                lon=round(lon, 6),
                elevation=2000.0,   # placeholder -> MNT IGN
                aspect=180.0,       # placeholder -> plein sud
                slope=30.0          # placeholder -> 30° typique
            ))
            lon += step
        lat += step

    return points


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from data.fetchers.openmeteo import get_hourly_weather
#from snow_model import compute_snow_conditions, create_mock_grid
today   = datetime.now(timezone.utc)
grid    = create_mock_grid(45.90, 6.85, 45.95, 6.95)
weather = get_hourly_weather(45.92, 6.87, target_date=today)



if __name__ == "__main__":

    # Box autour de Chamonix
    """
    grid = create_mock_grid(
        lat_min=45.90, lon_min=6.85,
        lat_max=45.95, lon_max=6.95,
        resolution_m=500
    )"""
    print(f"Grille : {len(grid)} points")

    # 3 scénarios horaires représentatifs
    weather_sequence = [
        HourlyWeather(   # 8h — matin froid, belle poudre
            hour=8,
            temperature_2m=-5.0,
            reference_elevation=1000,
            wind_speed=20,
            shortwave_radiation=200,
            snowfall_last_24h=25,
            snowfall_last_72h=40,
            hours_above_zero_last_48h=0,
            hours_below_minus2_last_12h=8
        ),
        HourlyWeather(   # 11h — soleil qui monte
            hour=11,
            temperature_2m=-1.0,
            reference_elevation=1000,
            wind_speed=25,
            shortwave_radiation=500,
            snowfall_last_24h=25,
            snowfall_last_72h=40,
            hours_above_zero_last_48h=1,
            hours_below_minus2_last_12h=4
        ),
        HourlyWeather(   # 14h — après-midi printanier
            hour=14,
            temperature_2m=2.0,
            reference_elevation=1000,
            wind_speed=30,
            shortwave_radiation=700,
            snowfall_last_24h=5,
            snowfall_last_72h=10,
            hours_above_zero_last_48h=5,
            hours_below_minus2_last_12h=0
        ),
    ]

    results = compute_snow_conditions(grid, weather, today.month, today.day)

    # Résumé par heure
    for hour in [8, 11, 14]:
        hour_results = [r for r in results if r.hour == hour]
        conditions: dict[str, int] = {}
        for r in hour_results:
            label = r.condition.label()
            conditions[label] = conditions.get(label, 0) + 1

        print(f"\n{hour}h00 — {len(hour_results)} points")
        for label, count in sorted(conditions.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"  {label:25s} {bar} ({count})")