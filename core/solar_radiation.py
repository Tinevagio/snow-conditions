"""
solar_radiation.py
------------------
Calculs astronomiques de rayonnement solaire pour le modele de neige.
Implementation from scratch — aucune dependance externe.
"""

import math
from dataclasses import dataclass


@dataclass
class SolarPosition:
    elevation_deg: float
    azimuth_deg: float
    is_above_horizon: bool


@dataclass
class RadiationResult:
    solar_position: SolarPosition
    incidence_angle_deg: float
    direct_radiation: float
    diffuse_radiation: float
    total_radiation: float
    temperature_correction: float


def _deg2rad(deg): return math.radians(deg)
def _rad2deg(rad): return math.degrees(rad)
def _cos(deg): return math.cos(math.radians(deg))
def _sin(deg): return math.sin(math.radians(deg))
def _acos(x): return _rad2deg(math.acos(max(-1.0, min(1.0, x))))
def _asin(x): return _rad2deg(math.asin(max(-1.0, min(1.0, x))))


def day_of_year(month: int, day: int) -> int:
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return sum(days_per_month[:month]) + day


def solar_declination(doy: int) -> float:
    """Declinaison solaire — Spencer (1971)."""
    B = 2 * math.pi * (doy - 1) / 365
    decl = (0.006918
            - 0.399912 * math.cos(B)
            + 0.070257 * math.sin(B)
            - 0.006758 * math.cos(2 * B)
            + 0.000907 * math.sin(2 * B)
            - 0.002697 * math.cos(3 * B)
            + 0.00148  * math.sin(3 * B))
    return _rad2deg(decl)


def equation_of_time(doy: int) -> float:
    """Equation du temps en minutes — Spencer (1971)."""
    B = 2 * math.pi * (doy - 1) / 365
    eot = (0.0000075
           + 0.001868  * math.cos(B)
           - 0.032077  * math.sin(B)
           - 0.014615  * math.cos(2 * B)
           - 0.04089   * math.sin(2 * B))
    return _rad2deg(eot) * 4


def solar_hour_angle(hour_utc: float, lon: float, doy: int) -> float:
    """Angle horaire solaire en degres. 0 = midi solaire."""
    eot_minutes = equation_of_time(doy)
    solar_time = hour_utc + lon / 15.0 + eot_minutes / 60.0
    return (solar_time - 12.0) * 15.0


def solar_position(hour_utc: float, lat: float, lon: float,
                   month: int, day: int) -> SolarPosition:
    """Position exacte du soleil dans le ciel."""
    doy   = day_of_year(month, day)
    decl  = solar_declination(doy)
    omega = solar_hour_angle(hour_utc, lon, doy)

    sin_elev = (_sin(lat) * _sin(decl)
                + _cos(lat) * _cos(decl) * _cos(omega))
    elevation = _asin(sin_elev)

    if elevation <= 0:
        return SolarPosition(elevation_deg=elevation,
                             azimuth_deg=0.0, is_above_horizon=False)

    cos_azimuth = ((_sin(decl) - _sin(lat) * sin_elev)
                   / (_cos(lat) * math.cos(math.radians(elevation))))
    azimuth_raw = _acos(cos_azimuth)
    azimuth = azimuth_raw if omega <= 0 else 360.0 - azimuth_raw

    return SolarPosition(elevation_deg=round(elevation, 2),
                         azimuth_deg=round(azimuth, 2),
                         is_above_horizon=True)


def incidence_angle_on_slope(solar_pos: SolarPosition,
                              aspect: float, slope: float) -> float:
    """
    Angle d'incidence du rayonnement sur un versant incline.
    > 90 = versant a l'ombre.
    """
    if not solar_pos.is_above_horizon:
        return 90.0

    elev = solar_pos.elevation_deg
    azim = solar_pos.azimuth_deg
    delta_azimuth = azim - aspect

    cos_incidence = (_sin(elev) * _cos(slope)
                     + _cos(elev) * _sin(slope) * _cos(delta_azimuth))
    return round(_acos(cos_incidence), 2)


def extraterrestrial_radiation(doy: int) -> float:
    """Rayonnement extraterrestre (W/m2). Constante solaire 1367 W/m2."""
    B = 2 * math.pi * doy / 365
    correction = (1.00011
                  + 0.034221 * math.cos(B)
                  + 0.00128  * math.sin(B)
                  + 0.000719 * math.cos(2 * B)
                  + 0.000077 * math.sin(2 * B))
    return 1367.0 * correction


def atmospheric_transmittance(elevation_deg: float, altitude_m: float) -> float:
    """Fraction du rayonnement qui atteint le sol apres traversee de l'atmosphere."""
    if elevation_deg <= 0:
        return 0.0
    elevation_rad = math.radians(elevation_deg)
    air_mass = 1.0 / (math.sin(elevation_rad)
                      + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)
    altitude_factor = math.exp(-altitude_m / 8500.0)
    transmittance = (0.7 ** (air_mass ** 0.678)) ** altitude_factor
    return max(0.0, min(1.0, transmittance))


def diffuse_radiation_on_slope(direct_horizontal: float, slope: float,
                                albedo_snow: float = 0.8) -> float:
    """Rayonnement diffus : ciel visible + reflexion neige environnante."""
    sky_view_factor    = (1 + _cos(slope)) / 2
    ground_view_factor = (1 - _cos(slope)) / 2
    diffuse_sky    = 0.15 * direct_horizontal * sky_view_factor
    diffuse_ground = albedo_snow * direct_horizontal * ground_view_factor
    return round(diffuse_sky + diffuse_ground, 1)


def solar_to_temperature_correction(total_radiation: float,
                                     slope: float) -> float:
    """
    Correction de temperature surface due au rayonnement solaire.
    La neige absorbe ~20% du rayonnement (albedo ~0.8).
    100 W/m2 absorbes ~= +1C en surface.
    """
    absorbed = total_radiation * 0.20
    #base_correction = absorbed / 100.0
    #slope_bonus = 1.0 + (slope / 90.0) * 0.2
    
    base_correction = total_radiation / 100.0
    slope_bonus = 1.0 + (slope / 90.0) * 0.2
    
    return min(base_correction * slope_bonus, 6.0)


def effective_radiation(hour_utc: float, lat: float, lon: float,
                         month: int, day: int, aspect: float, slope: float,
                         altitude_m: float,
                         albedo_snow: float = 0.8) -> RadiationResult:
    """
    Calcul complet du rayonnement effectif sur un versant.
    Fonction principale appelee depuis snow_model.py.
    """
    doy     = day_of_year(month, day)
    sol_pos = solar_position(hour_utc, lat, lon, month, day)

    if not sol_pos.is_above_horizon:
        return RadiationResult(solar_position=sol_pos,
                               incidence_angle_deg=90.0,
                               direct_radiation=0.0, diffuse_radiation=0.0,
                               total_radiation=0.0, temperature_correction=0.0)

    inc_angle         = incidence_angle_on_slope(sol_pos, aspect, slope)
    I0                = extraterrestrial_radiation(doy)
    tau               = atmospheric_transmittance(sol_pos.elevation_deg, altitude_m)
    direct_horizontal = I0 * tau * _sin(sol_pos.elevation_deg)
    direct_on_slope   = 0.0 if inc_angle >= 90.0 else I0 * tau * _cos(inc_angle)
    diffuse           = diffuse_radiation_on_slope(direct_horizontal, slope, albedo_snow)
    total             = round(direct_on_slope + diffuse, 1)
    temp_corr         = solar_to_temperature_correction(total, slope)

    return RadiationResult(solar_position=sol_pos,
                           incidence_angle_deg=round(inc_angle, 2),
                           direct_radiation=round(direct_on_slope, 1),
                           diffuse_radiation=round(diffuse, 1),
                           total_radiation=total,
                           temperature_correction=round(temp_corr, 2))


def sunrise_sunset(lat: float, lon: float,
                   month: int, day: int) -> tuple[float, float]:
    """Heures de lever et coucher du soleil (UTC decimal)."""
    doy  = day_of_year(month, day)
    decl = solar_declination(doy)
    cos_omega = -_sin(lat) * _sin(decl) / (_cos(lat) * _cos(decl))
    if cos_omega > 1:  return (12.0, 12.0)
    if cos_omega < -1: return (0.0, 24.0)
    omega_deg      = _acos(cos_omega)
    eot            = equation_of_time(doy)
    solar_noon_utc = 12.0 - lon / 15.0 - eot / 60.0
    return (round(solar_noon_utc - omega_deg / 15.0, 2),
            round(solar_noon_utc + omega_deg / 15.0, 2))


def best_powder_window(lat: float, lon: float, month: int, day: int,
                        aspect: float, slope: float, altitude_m: float,
                        radiation_threshold: float = 300.0) -> tuple | None:
    """
    Fenetre horaire ou le rayonnement depasse le seuil (poudre menacee).
    Retourne (heure_debut, heure_fin) UTC ou None si jamais depasse.
    """
    start_hour = end_hour = None
    for h_tenth in range(240):
        h = h_tenth / 10.0
        r = effective_radiation(h, lat, lon, month, day,
                                aspect, slope, altitude_m)
        if r.total_radiation > radiation_threshold:
            if start_hour is None: start_hour = h
            end_hour = h
    if start_hour is None: return None
    return (round(start_hour, 1), round(end_hour, 1))


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lat, lon   = 45.92, 6.87   # Chamonix
    month, day = 2, 15
    altitude   = 2400

    print("=" * 62)
    print(f"  Calculs solaires — Chamonix {day:02d}/{month:02d} | {altitude}m")
    print("=" * 62)

    sunrise, sunset = sunrise_sunset(lat, lon, month, day)
    print(f"\n  Lever  : {int(sunrise)}h{int((sunrise % 1)*60):02d} UTC")
    print(f"  Coucher: {int(sunset)}h{int((sunset % 1)*60):02d} UTC")

    print(f"\n  {'Heure':>5} | {'NORD 0deg':>16} | {'EST 90deg':>16} | {'SUD 180deg':>16}")
    print("  " + "-" * 58)

    for hour in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        nord = effective_radiation(hour, lat, lon, month, day, 0,   35, altitude)
        est  = effective_radiation(hour, lat, lon, month, day, 90,  35, altitude)
        sud  = effective_radiation(hour, lat, lon, month, day, 180, 35, altitude)

        def fmt(r):
            return f"{r.total_radiation:5.0f}W ({r.temperature_correction:+.1f}C)"

        print(f"  {hour:02d}h   | {fmt(nord):>16} | {fmt(est):>16} | {fmt(sud):>16}")

    print("\n  --- Fenetre poudre preservee (< 300 W/m2) ---")
    for aspect, label in [(0,"Nord"),(90,"Est"),(180,"Sud"),(270,"Ouest")]:
        w = best_powder_window(lat, lon, month, day, aspect, 35, altitude)
        if w is None:
            print(f"  {label:6s}: poudre preservee toute la journee")
        else:
            print(f"  {label:6s}: rayonnement fort de {w[0]}h a {w[1]}h UTC")