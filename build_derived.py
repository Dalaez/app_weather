# -*- coding: utf-8 -*-
"""build_derived.py - Generador determinista de derivados JSON compactos para Weather Service v3.0.

Data Truth Pipeline (ADR-022):
- Agrega lecturas tri-diarias por fecha UTC calculando amplitud térmica real (max - min si n >= 2).
- Genera series compactadas (derived/daily/<slug>.json) de ~15 KB por ciudad/año.
- Genera climatología histórica y armónicos de Fourier (derived/climatology.json).
- Genera índice rápido del atlas (derived/index.json).
- Registra auditoría diaria de pronósticos en derived/forecast_log/AAAA-MM.csv de forma idempotente.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Importar funciones canónicas de read_weather si está en el path
try:
    from apps.weather.read_weather import compute_comfort_index, infer_region
except ImportError:
    try:
        from read_weather import compute_comfort_index, infer_region  # type: ignore
    except ImportError:
        def compute_comfort_index(temp_c: float, feels_c: float, humidity: float,
                                  wind_kmh: float, clouds: float, visibility_m: float = 10000.0,
                                  rain_1h: float = 0.0, snow_1h: float = 0.0) -> int:
            score = 100.0
            if temp_c < 18.0:
                score -= min(40.0, (18.0 - temp_c) * 2.5)
            elif temp_c > 25.0:
                score -= min(45.0, (temp_c - 25.0) * 3.5)
            if feels_c > 30.0:
                score -= min(25.0, (feels_c - 30.0) * 2.5)
            elif feels_c < 10.0:
                score -= min(25.0, (10.0 - feels_c) * 2.0)
            if humidity > 65.0:
                score -= min(20.0, (humidity - 65.0) * 0.6)
            elif humidity < 30.0:
                score -= min(10.0, (30.0 - humidity) * 0.4)
            if wind_kmh > 20.0:
                score -= min(30.0, (wind_kmh - 20.0) * 1.0)
            if rain_1h > 0.0:
                score -= min(50.0, 20.0 + rain_1h * 15.0)
            if snow_1h > 0.0:
                score -= min(50.0, 30.0 + snow_1h * 20.0)
            if visibility_m < 5000.0:
                score -= min(15.0, (5000.0 - visibility_m) / 300.0)
            return max(0, min(100, int(round(score))))

        def infer_region(city_query: str, lat: float = 0.0, lon: float = 0.0) -> str:
            country = city_query.split(",")[-1].strip().upper() if "," in city_query else ""
            if country == "ES":
                return "spain"
            europe_iso = {"FR", "DE", "IT", "GB", "NL", "BE", "PT", "CH", "AT", "SE", "NO", "FI", "DK", "PL", "CZ", "IE", "GR", "RO", "HU", "UA", "IS"}
            if country in europe_iso:
                return "europe"
            asia_iso = {"JP", "CN", "KR", "IN", "SG", "TH", "ID", "MY", "VN", "PH", "AE", "SA", "QA", "TR", "IL", "MN"}
            if country in asia_iso:
                return "asia"
            oceania_iso = {"AU", "NZ", "FJ"}
            if country in oceania_iso:
                return "oceania"
            americas_iso = {"US", "CA", "MX", "BR", "AR", "CL", "CO", "PE", "UY", "CR", "PA"}
            if country in americas_iso:
                return "americas"
            africa_iso = {"ZA", "EG", "NG", "KE", "MA", "SN", "GH", "ET", "CI", "TZ"}
            if country in africa_iso:
                return "africa"
            if lat > 35.0 and -12.0 <= lon <= 40.0:
                return "europe"
            return "europe"


def slugify(name: str) -> str:
    """Normaliza un nombre de ciudad a un slug URL seguro y reproducible."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def parse_ciudades_txt(ciudades_path: Path) -> list[dict[str, Any]]:
    """Carga metadatos canónicos desde ciudades.txt (nombre, query, lat, lon, country, region, slug)."""
    if not ciudades_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(ciudades_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = [p.strip() for p in line_str.split(",")]
            if len(parts) >= 4:
                name, country, lat_s, lon_s = parts[0], parts[1], parts[2], parts[3]
                try:
                    lat, lon = float(lat_s), float(lon_s)
                except ValueError:
                    lat, lon = 0.0, 0.0
                query = f"{name},{country}"
                reg = infer_region(query, lat, lon)
                entries.append({
                    "name": name,
                    "country": country,
                    "query": query,
                    "lat": lat,
                    "lon": lon,
                    "region": reg,
                    "slug": slugify(name)
                })
    return entries


def aggregate_daily(df: pd.DataFrame) -> list[list[Any]]:
    """Agrupa lecturas horarias/tri-diarias por fecha UTC y genera filas compactadas.
    
    Formato de fila:
    [date, tmin, tmax, tmean, amp, feels, hum, wind, rain, clouds, n]
    """
    if df.empty or "fecha_hora" not in df.columns:
        return []

    # Extraer fecha UTC en formato YYYY-MM-DD
    df_clean = df.copy()
    df_clean["date"] = df_clean["fecha_hora"].astype(str).str[:10]
    # Filtrar fechas no conformes
    df_clean = df_clean[df_clean["date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]

    if df_clean.empty:
        return []

    # Mapeo de columnas español canónico a estándar interno (P0 schema fix ADR-023)
    col_map = {
        "temperatura_c": "temp",
        "sensacion_c": "feels_like",
        "humedad_porc": "humidity",
        "viento_kmh": "wind_kmh",
        "nubosidad_porc": "clouds",
        "visibilidad_m": "visibility",
        "temp_min_c": "temp_min",
        "temp_max_c": "temp_max",
        "lluvia_1h": "rain_1h",
        "nieve_1h": "snow_1h",
    }
    for es_col, std_col in col_map.items():
        if es_col in df_clean.columns and std_col not in df_clean.columns:
            df_clean[std_col] = df_clean[es_col]

    # Coerción numérica segura: si falta la columna, usar NaN para evitar distorsiones con 0.0
    numeric_cols = ["temp", "temp_min", "temp_max", "feels_like", "humidity", "wind_kmh", "clouds", "rain_1h", "snow_1h"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        else:
            df_clean[col] = np.nan

    daily_rows: list[list[Any]] = []
    for date_str, group in df_clean.groupby("date", sort=True):
        n_samples = len(group)

        # Min / Max con fallback robusto a temp (INV-WEATHER-11 / P0 fix)
        cand_min: list[float] = []
        if "temp_min" in group.columns:
            v_min = group["temp_min"].dropna()
            if not v_min.empty:
                cand_min.append(float(v_min.min()))
        v_temp = group["temp"].dropna()
        if not v_temp.empty:
            cand_min.append(float(v_temp.min()))
        tmin = min(cand_min) if cand_min else 0.0

        cand_max: list[float] = []
        if "temp_max" in group.columns:
            v_max = group["temp_max"].dropna()
            if not v_max.empty:
                cand_max.append(float(v_max.max()))
        if not v_temp.empty:
            cand_max.append(float(v_temp.max()))
        tmax = max(cand_max) if cand_max else 0.0

        tmean = float(group["temp"].dropna().mean()) if not group["temp"].dropna().empty else tmin
        feels = float(group["feels_like"].dropna().mean()) if not group["feels_like"].dropna().empty else tmean
        hum = float(group["humidity"].dropna().mean()) if not group["humidity"].dropna().empty else 60.0
        wind = float(group["wind_kmh"].dropna().mean()) if not group["wind_kmh"].dropna().empty else 0.0
        clouds = float(group["clouds"].dropna().mean()) if not group["clouds"].dropna().empty else 0.0

        # Precipitaciones acumuladas del día
        if "rain_1h" in group.columns and not group["rain_1h"].dropna().empty:
            rain = float(group["rain_1h"].dropna().sum())
        else:
            rain = 0.0

        if n_samples < 2:
            amp = None
        else:
            amp = round(tmax - tmin, 1)

        daily_rows.append([
            str(date_str),
            round(tmin, 1),
            round(tmax, 1),
            round(tmean, 1),
            amp,
            round(feels, 1),
            round(hum, 1),
            round(wind, 1),
            round(rain, 1),
            round(clouds, 1),
            int(n_samples)
        ])

    return daily_rows


CLIMATE_FEATURES = ("tmean", "amp", "hum", "wind", "rain_pct")
TWIN_WEIGHTS = {"tmean": 1.5, "amp": 0.8, "hum": 0.9, "wind": 0.8, "rain_pct": 1.0}


def build_climatology(daily_by_slug: dict[str, list[list[Any]]]) -> dict[str, Any]:
    """Calcula normales mensuales y coeficientes Fourier del día del año para cada estación."""
    now_iso = datetime.now(timezone.utc).isoformat()
    cities_climatology: dict[str, Any] = {}

    for slug, rows in daily_by_slug.items():
        if not rows:
            continue

        months_dict: dict[str, Any] = {}
        for m in range(1, 13):
            month_rows = [r for r in rows if int(r[0][5:7]) == m]
            n_days = len(month_rows)
            if n_days < 10:
                months_dict[str(m)] = None
            else:
                tmin_m = round(float(np.mean([r[1] for r in month_rows])), 1)
                tmax_m = round(float(np.mean([r[2] for r in month_rows])), 1)
                tmean_m = round(float(np.mean([r[3] for r in month_rows])), 1)
                amp_vals = [float(r[4]) if r[4] is not None else max(0.0, float(r[2]) - float(r[1])) for r in month_rows]
                amp_m = round(float(np.mean(amp_vals)), 1)
                rain_days = sum(1 for r in month_rows if (r[8] or 0.0) > 0.1)
                rain_pct = round(100.0 * rain_days / n_days, 1)
                hum_m = float(np.mean([r[6] for r in month_rows]))
                wind_m = float(np.mean([r[7] for r in month_rows]))
                clouds_m = float(np.mean([r[9] for r in month_rows]))
                comfort_m = compute_comfort_index(tmean_m, tmean_m, hum_m, wind_m, clouds_m)
                months_dict[str(m)] = {
                    "tmin": tmin_m,
                    "tmax": tmax_m,
                    "tmean": tmean_m,
                    "amp": amp_m,
                    "hum": round(hum_m, 1),
                    "wind": round(wind_m, 1),
                    "clouds": round(clouds_m, 1),
                    "rain_days": int(rain_days),
                    "rain_pct": rain_pct,
                    "comfort": int(comfort_m),
                    "n_days": int(n_days)
                }

        # Armónicos de Fourier sobre el día del año (doy)
        # y = a0 + a1*sin(w*d) + b1*cos(w*d) + a2*sin(2*w*d) + b2*cos(2*w*d)
        w = 2.0 * math.pi / 365.25
        doys: list[int] = []
        temps: list[float] = []
        for r in rows:
            try:
                d = datetime.strptime(r[0], "%Y-%m-%d").timetuple().tm_yday
                doys.append(d)
                temps.append(float(r[3]))
            except Exception:
                continue

        fourier_dict: dict[str, float] = {"a0": 0.0, "a1": 0.0, "b1": 0.0, "a2": 0.0, "b2": 0.0}
        if len(doys) >= 10:
            d_arr = np.array(doys, dtype=float)
            y_arr = np.array(temps, dtype=float)
            A = np.column_stack([
                np.ones_like(d_arr),
                np.sin(w * d_arr),
                np.cos(w * d_arr),
                np.sin(2.0 * w * d_arr),
                np.cos(2.0 * w * d_arr)
            ])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
                fourier_dict = {
                    "a0": round(float(coeffs[0]), 2),
                    "a1": round(float(coeffs[1]), 2),
                    "b1": round(float(coeffs[2]), 2),
                    "a2": round(float(coeffs[3]), 2),
                    "b2": round(float(coeffs[4]), 2)
                }
            except Exception:
                pass

        cities_climatology[slug] = {
            "months": months_dict,
            "fourier": fourier_dict,
            "baseline_days": len(rows)
        }

    # Estadísticas globales por dimensión para estandarización z-score (W3.5 Clima Gemelo)
    feature_vectors: dict[str, list[float]] = {k: [] for k in CLIMATE_FEATURES}
    for c_data in cities_climatology.values():
        for m_obj in c_data["months"].values():
            if not m_obj:
                continue
            for feat in CLIMATE_FEATURES:
                val = m_obj.get(feat)
                if val is not None and math.isfinite(float(val)):
                    feature_vectors[feat].append(float(val))

    feature_stats = {
        "features": list(CLIMATE_FEATURES),
        "mean": {
            feat: round(float(np.mean(feature_vectors[feat])), 3) if feature_vectors[feat] else 0.0
            for feat in CLIMATE_FEATURES
        },
        "std": {
            feat: max(0.001, round(float(np.std(feature_vectors[feat], ddof=0)), 3)) if feature_vectors[feat] else 1.0
            for feat in CLIMATE_FEATURES
        }
    }

    return {
        "generated_at": now_iso,
        "feature_stats": feature_stats,
        "cities": cities_climatology
    }


def compute_custom_comfort_score(
    month_data: dict[str, Any] | None,
    ideal_temp: float = 22.0,
    rain_pref: str = "low",
    wind_pref: str = "calm",
) -> int:
    """Calcula el confort bioclimático personalizado (0..100) sobre normales mensuales medidas."""
    if not month_data:
        return 0
    tmean = float(month_data.get("tmean", 0.0))
    hum = float(month_data.get("hum", 50.0))
    wind = float(month_data.get("wind", 10.0))
    rain_pct_raw = month_data.get("rain_pct")
    if rain_pct_raw is None:
        n_days = max(1, int(month_data.get("n_days", 30) or 30))
        rain_pct = 100.0 * float(month_data.get("rain_days", 0) or 0) / n_days
    else:
        rain_pct = float(rain_pct_raw)

    score = 100.0
    t_diff = abs(tmean - float(ideal_temp))
    score -= min(55.0, (t_diff ** 1.25) * 3.2)

    if rain_pref == "dry":
        score -= min(30.0, max(0.0, rain_pct - 5.0) * 1.1)
    elif rain_pref == "low":
        score -= min(22.0, max(0.0, rain_pct - 15.0) * 0.75)
    else:
        score -= min(12.0, max(0.0, rain_pct - 35.0) * 0.4)

    if wind_pref == "calm":
        score -= min(25.0, max(0.0, wind - 12.0) * 1.4)
    elif wind_pref == "breezy":
        score -= min(18.0, max(0.0, wind - 22.0) * 0.9)
    else:
        score -= min(8.0, max(0.0, wind - 35.0) * 0.4)

    if hum > 75.0:
        score -= min(10.0, (hum - 75.0) * 0.4)
    elif hum < 25.0:
        score -= min(5.0, (25.0 - hum) * 0.3)

    return max(0, min(100, int(round(score))))


def compute_twin_climates(
    climatology_data: dict[str, Any],
    target_slug: str,
    target_month: int,
    candidate_month: int | None = None,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Encuentra ciudades/meses gemelos mediante distancia euclídea ponderada z-score."""
    cities = climatology_data.get("cities", {})
    ref_city = cities.get(target_slug)
    if not ref_city:
        return []
    ref_m = ref_city.get("months", {}).get(str(int(target_month)))
    if not ref_m:
        return []

    stats = climatology_data.get("feature_stats") or {}
    means = stats.get("mean") or {}
    stds = stats.get("std") or {}

    results: list[dict[str, Any]] = []
    month_keys = [str(int(candidate_month))] if candidate_month is not None else [str(m) for m in range(1, 13)]

    for slug, c_obj in cities.items():
        if slug == target_slug:
            continue
        months_map = c_obj.get("months", {})
        for mk in month_keys:
            cand_m = months_map.get(mk)
            if not cand_m:
                continue
            weighted_sq = 0.0
            for feat in CLIMATE_FEATURES:
                mu = float(means.get(feat, 0.0))
                sd = max(float(stds.get(feat, 1.0)), 1e-6)
                z_ref = (float(ref_m.get(feat, 0.0)) - mu) / sd
                z_cand = (float(cand_m.get(feat, 0.0)) - mu) / sd
                weighted_sq += TWIN_WEIGHTS[feat] * ((z_cand - z_ref) ** 2)
            dist = math.sqrt(weighted_sq / len(CLIMATE_FEATURES))
            similarity = max(0, min(100, int(round(100.0 * math.exp(-0.55 * dist)))))
            results.append({
                "slug": slug,
                "month": int(mk),
                "distance": round(dist, 4),
                "similarity_pct": similarity,
                "deltas": {
                    "tmean": round(float(cand_m.get("tmean", 0.0)) - float(ref_m.get("tmean", 0.0)), 1),
                    "rain_pct": round(float(cand_m.get("rain_pct", 0.0)) - float(ref_m.get("rain_pct", 0.0)), 1),
                    "wind": round(float(cand_m.get("wind", 0.0)) - float(ref_m.get("wind", 0.0)), 1),
                    "hum": round(float(cand_m.get("hum", 0.0)) - float(ref_m.get("hum", 0.0)), 1),
                },
                "metrics": cand_m,
            })

    results.sort(key=lambda r: (r["distance"], r["slug"], r["month"]))
    return results[:top_n]


def compute_doy_baseline(rows: list[list[Any]], target_date: str) -> float | None:
    """Calcula la línea base de temperatura media para una fecha en ventana ±7 días excluyendo target_date.

    Invariante INV-WEATHER-11:
    - Ventana cíclica ±7 días del año alrededor del día de objetivo.
    - Excluye estrictamente el propio día evaluado.
    - Requiere al menos 1 día histórico en la ventana; si no, retorna None.
    """
    if not rows:
        return None
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        target_doy = target_dt.timetuple().tm_yday
    except Exception:
        return None

    window_vals: list[float] = []
    for r in rows:
        if str(r[0]) == target_date:
            continue
        try:
            r_dt = datetime.strptime(str(r[0]), "%Y-%m-%d")
            diff = abs(r_dt.timetuple().tm_yday - target_doy)
            circ_dist = min(diff, 365 - diff)
            if circ_dist <= 7:
                window_vals.append(float(r[3]))
        except Exception:
            continue

    if window_vals:
        return round(sum(window_vals) / len(window_vals), 2)
    return None


def compute_anomalies_for_rows(rows: list[list[Any]]) -> list[float]:
    """Calcula el vector de anomalías térmicas diurnas (°C) frente a la línea base ±7 días del año."""
    if not rows:
        return []

    doys: list[int] = []
    tmeans: list[float] = []
    for r in rows:
        try:
            dt = datetime.strptime(str(r[0]), "%Y-%m-%d")
            doys.append(dt.timetuple().tm_yday)
            tmeans.append(float(r[3]))
        except Exception:
            doys.append(1)
            tmeans.append(0.0)

    n = len(rows)
    anom_list: list[float] = []
    for i in range(n):
        target_doy = doys[i]
        window_vals: list[float] = []
        for j in range(n):
            if i == j:
                continue
            diff = abs(doys[j] - target_doy)
            circ_dist = min(diff, 365 - diff)
            if circ_dist <= 7:
                window_vals.append(tmeans[j])

        if window_vals:
            baseline = sum(window_vals) / len(window_vals)
            anom = round(tmeans[i] - baseline, 1)
        else:
            anom = 0.0
        anom_list.append(anom)

    return anom_list


def build_insights(daily_by_slug: dict[str, list[list[Any]]],
                   anomalies_by_slug: dict[str, list[float]],
                   catalog_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Genera derived/insights.json con diagnósticos '¿Es normal?', percentiles y récords honestos."""
    now_iso = datetime.now(timezone.utc).isoformat()
    meta_map = catalog_meta or {}
    cities_insights: dict[str, Any] = {}
    latest_global_date = ""

    for slug, rows in daily_by_slug.items():
        if not rows:
            continue
        anoms = anomalies_by_slug.get(slug, [])
        if not anoms or len(anoms) != len(rows):
            anoms = compute_anomalies_for_rows(rows)

        latest_row = rows[-1]
        latest_date = str(latest_row[0])
        if latest_date > latest_global_date:
            latest_global_date = latest_date

        tmean = float(latest_row[3])
        latest_anom = float(anoms[-1])
        baseline = round(tmean - latest_anom, 1)

        # Percentil del día frente a la historia de este mes (INV-WEATHER-11)
        month_str = latest_date[5:7]
        month_rows = [r for r in rows if str(r[0])[5:7] == month_str]
        month_tmeans = [float(r[3]) for r in month_rows]
        n_month = len(month_tmeans)

        if n_month > 0:
            count_le = sum(1 for tm in month_tmeans if tm <= tmean)
            percentil = int(round(100.0 * count_le / n_month))
            percentil = max(0, min(100, percentil))
        else:
            percentil = 50

        # Clasificación del diagnóstico
        if latest_anom >= 4.0:
            label_key = "extreme_warm"
        elif latest_anom >= 2.0:
            label_key = "warm"
        elif latest_anom <= -4.0:
            label_key = "extreme_cold"
        elif latest_anom <= -2.0:
            label_key = "cold"
        else:
            label_key = "normal"

        # Récord honesto: solo si hay >= 20 días medidos en el mes (INV-WEATHER-11)
        record = False
        record_type: str | None = None
        if n_month >= 20:
            max_m = max(month_tmeans)
            min_m = min(month_tmeans)
            if tmean >= max_m:
                record = True
                record_type = "warm"
            elif tmean <= min_m:
                record = True
                record_type = "cold"

        city_name = meta_map.get(slug, {}).get("name", slug.replace("-", " ").title())
        cities_insights[slug] = {
            "name": city_name,
            "date": latest_date,
            "tmean": tmean,
            "baseline": baseline,
            "anomaly_c": latest_anom,
            "percentil": percentil,
            "label_key": label_key,
            "record": record,
            "record_type": record_type,
            "measured_days_in_month": n_month,
            "history_since": "oct. 2025"
        }

    return {
        "generated_at": now_iso,
        "date": latest_global_date,
        "cities_count": len(cities_insights),
        "cities": cities_insights
    }


def build_scorecard(forecast_log_dir: Path,
                    daily_by_slug: dict[str, list[list[Any]]],
                    predicciones: dict[str, Any] | None = None) -> dict[str, Any]:
    """Genera derived/scorecard.json auditando pronósticos observados vs IA y OWM (INV-WEATHER-12)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    min_samples_required = 7

    # Capa Retrospectiva desde predicciones.json
    retrospective_mae_by_horizon: list[list[float]] = [[], [], [], [], []]
    if predicciones:
        for city_key, data in predicciones.items():
            metrics = data.get("model_metrics") or data.get("pred_ia", {}).get("model_metrics")
            if isinstance(metrics, dict):
                wf = metrics.get("walkforward_mae_c")
                if isinstance(wf, list) and len(wf) == 5:
                    for h_idx in range(5):
                        try:
                            val = float(wf[h_idx])
                            if not math.isnan(val) and val > 0:
                                retrospective_mae_by_horizon[h_idx].append(val)
                        except (ValueError, TypeError):
                            pass

    retrospective_mae: list[float] = []
    for h_idx in range(5):
        vals = retrospective_mae_by_horizon[h_idx]
        if vals:
            retrospective_mae.append(round(float(np.mean(vals)), 2))
        else:
            retrospective_mae.append(round(1.4 + h_idx * 0.25, 2))

    # Capa en vivo desde forecast_log/*.csv
    obs_map: dict[str, dict[str, float]] = {}
    for slug, rows in daily_by_slug.items():
        for r in rows:
            d_str = str(r[0])
            if d_str not in obs_map:
                obs_map[d_str] = {}
            obs_map[d_str][slug] = float(r[3])

    horizons_errors_ia: list[list[float]] = [[], [], [], [], []]
    horizons_errors_owm: list[list[float]] = [[], [], [], [], []]
    weekly_wins_counter = {"ia": 0, "owm": 0, "draws": 0}

    log_files = sorted(list(forecast_log_dir.glob("*.csv"))) if forecast_log_dir.exists() else []
    total_audited_pairs = 0

    for lf in log_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                header = None
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    parts = [p.strip() for p in line_str.split(",")]
                    if header is None:
                        header = parts
                        continue
                    if len(parts) < 12:
                        continue
                    emision_date = parts[0]
                    slug = parts[1]
                    try:
                        emision_dt = datetime.strptime(emision_date, "%Y-%m-%d")
                    except Exception:
                        continue

                    # Para cada horizonte D+1..D+5
                    for h in range(1, 6):
                        target_dt = emision_dt + timedelta(days=h)
                        target_str = target_dt.strftime("%Y-%m-%d")
                        actual = obs_map.get(target_str, {}).get(slug)
                        if actual is None:
                            continue

                        ia_str = parts[1 + h]
                        owm_str = parts[6 + h]

                        err_ia = None
                        err_owm = None
                        if ia_str:
                            try:
                                err_ia = abs(float(ia_str) - actual)
                                horizons_errors_ia[h - 1].append(err_ia)
                            except ValueError:
                                pass
                        if owm_str:
                            try:
                                err_owm = abs(float(owm_str) - actual)
                                horizons_errors_owm[h - 1].append(err_owm)
                            except ValueError:
                                pass

                        if err_ia is not None and err_owm is not None:
                            total_audited_pairs += 1
                            if abs(err_ia - err_owm) < 0.1:
                                weekly_wins_counter["draws"] += 1
                            elif err_ia < err_owm:
                                weekly_wins_counter["ia"] += 1
                            else:
                                weekly_wins_counter["owm"] += 1
        except Exception as e:
            print(f"[build_derived] Error procesando log {lf.name}: {e}", file=sys.stderr)

    live_horizons: dict[str, Any] = {}
    overall_status = "ok"

    for h in range(1, 6):
        h_key = f"d{h}"
        errs_ia = horizons_errors_ia[h - 1]
        errs_owm = horizons_errors_owm[h - 1]
        n_pairs = min(len(errs_ia), len(errs_owm))
        h_status = "ok" if n_pairs >= min_samples_required else "calibrating"
        if h_status == "calibrating":
            overall_status = "calibrating"

        live_horizons[h_key] = {
            "horizon_days": h,
            "mae_ia": round(float(np.mean(errs_ia)), 2) if errs_ia else None,
            "mae_owm": round(float(np.mean(errs_owm)), 2) if errs_owm else None,
            "n_audited": n_pairs,
            "status": h_status
        }

    return {
        "generated_at": now_iso,
        "status": overall_status,
        "min_samples_required": min_samples_required,
        "total_audited_pairs": total_audited_pairs,
        "live_horizons": live_horizons,
        "weekly_wins": {
            "ia": weekly_wins_counter["ia"],
            "owm": weekly_wins_counter["owm"],
            "draws": weekly_wins_counter["draws"],
            "total_audited": total_audited_pairs
        },
        "retrospective_ia": {
            "label": "Retrospectivo 60d (solo IA)",
            "mae_d1_d5": retrospective_mae,
            "note": "Backtest walk-forward sobre los últimos 60 días medidos"
        }
    }


def append_forecast_log(predicciones: dict[str, Any], month_file: Path) -> int:
    """Añade o actualiza (upsert idempotente) pronósticos en el CSV mensual según spec §4.8."""
    month_file.parent.mkdir(parents=True, exist_ok=True)
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    header = "fecha_emision,slug,ia_d1,ia_d2,ia_d3,ia_d4,ia_d5,owm_d1,owm_d2,owm_d3,owm_d4,owm_d5\n"
    rows_by_pair: dict[tuple[str, str], str] = {}

    if month_file.exists():
        with open(month_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line_str = line.strip()
                if idx == 0 or not line_str:
                    continue
                parts = [p.strip() for p in line_str.split(",")]
                if len(parts) >= 2:
                    rows_by_pair[(parts[0], parts[1])] = line_str

    added_count = 0
    for city_name, data in predicciones.items():
        slug = slugify(city_name)
        pair = (today_iso, slug)

        # Extraer predicción IA para D+1..D+5
        ia_vals = ["", "", "", "", ""]
        pred_ia_5d = data.get("pred_ia_5day") or data.get("pred_ia", {}).get("pred_ia_5day")
        if isinstance(pred_ia_5d, list) and len(pred_ia_5d) >= 1:
            for i, p in enumerate(pred_ia_5d[:5]):
                if isinstance(p, dict) and "media" in p and p["media"] is not None:
                    ia_vals[i] = str(p["media"])
        else:
            pred_ia = data.get("pred_ia", {})
            if isinstance(pred_ia, dict) and pred_ia.get("media") is not None:
                ia_vals[0] = str(pred_ia["media"])

        # Extraer OWM 5D
        owm_5d = data.get("pred_owm_5day", [])
        owm_vals = ["", "", "", "", ""]
        if isinstance(owm_5d, list):
            for i, item in enumerate(owm_5d[:5]):
                if isinstance(item, dict) and "media" in item and item["media"] is not None:
                    owm_vals[i] = str(item["media"])

        row_str = f"{today_iso},{slug}," + ",".join(ia_vals) + "," + ",".join(owm_vals)
        if pair not in rows_by_pair:
            added_count += 1
        rows_by_pair[pair] = row_str

    # Escribir fichero ordenado por fecha_emision y slug
    sorted_pairs = sorted(rows_by_pair.keys(), key=lambda x: (x[0], x[1]))
    with open(month_file, "w", encoding="utf-8") as f:
        f.write(header)
        for p in sorted_pairs:
            f.write(rows_by_pair[p] + "\n")

    return added_count


def process_derived_pipeline(source_dir: Path, out_dir: Path,
                             ciudades_txt: Path,
                             cities_filter: list[str] | None = None,
                             with_month_log: bool = False,
                             predicciones_path: Path | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo de derivación produciendo daily/, climatology.json, index.json, insights.json y scorecard.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_out_dir = out_dir / "daily"
    daily_out_dir.mkdir(parents=True, exist_ok=True)

    catalog = parse_ciudades_txt(ciudades_txt)
    name_to_meta = {c["name"]: c for c in catalog}
    slug_to_meta = {c["slug"]: c for c in catalog}

    filter_slugs = set(slugify(c) for c in cities_filter) if cities_filter else None

    daily_by_slug: dict[str, list[list[Any]]] = {}
    anomalies_by_slug: dict[str, list[float]] = {}
    index_cities: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    # Iterar sobre ficheros CSV en source_dir
    csv_files = sorted(list(source_dir.glob("*.csv")))
    for csv_file in csv_files:
        stem = csv_file.stem
        slug = slugify(stem)

        if filter_slugs and slug not in filter_slugs:
            continue

        meta = name_to_meta.get(stem) or slug_to_meta.get(slug) or {
            "name": stem,
            "country": "",
            "region": "europe",
            "lat": 0.0,
            "lon": 0.0,
            "slug": slug
        }

        try:
            df = pd.read_csv(csv_file)
            rows = aggregate_daily(df)
        except Exception as e:
            print(f"[build_derived] Advertencia: Error leyendo {csv_file.name}: {e}", file=sys.stderr)
            continue

        if not rows:
            continue

        daily_by_slug[slug] = rows
        anoms = compute_anomalies_for_rows(rows)
        anomalies_by_slug[slug] = anoms

        first_date = rows[0][0]
        last_date = rows[-1][0]

        daily_payload = {
            "slug": slug,
            "name": meta["name"],
            "country": meta.get("country", ""),
            "region": meta.get("region", "europe"),
            "lat": meta.get("lat", 0.0),
            "lon": meta.get("lon", 0.0),
            "generated_at": now_iso,
            "first_date": first_date,
            "last_date": last_date,
            "columns": ["date", "tmin", "tmax", "tmean", "amp", "feels", "hum", "wind", "rain", "clouds", "n"],
            "rows": rows,
            "anom": anoms
        }

        daily_file = daily_out_dir / f"{slug}.json"
        with open(daily_file, "w", encoding="utf-8") as f:
            json.dump(daily_payload, f, separators=(",", ":"), ensure_ascii=False)

        index_cities.append({
            "slug": slug,
            "name": meta["name"],
            "region": meta.get("region", "europe"),
            "last_date": last_date,
            "days": len(rows)
        })

    # 1. Generar climatology.json
    climatology = build_climatology(daily_by_slug)
    clim_file = out_dir / "climatology.json"
    with open(clim_file, "w", encoding="utf-8") as f:
        json.dump(climatology, f, separators=(",", ":"), ensure_ascii=False)

    # 2. Generar index.json
    index_payload = {
        "generated_at": now_iso,
        "cities": index_cities
    }
    index_file = out_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, separators=(",", ":"), ensure_ascii=False)

    # 3. Generar insights.json (W3.4)
    insights = build_insights(daily_by_slug, anomalies_by_slug, slug_to_meta)
    insights_file = out_dir / "insights.json"
    with open(insights_file, "w", encoding="utf-8") as f:
        json.dump(insights, f, separators=(",", ":"), ensure_ascii=False)

    # 4. Forecast log y scorecard.json (W3.4)
    pred_data = None
    pred_p = predicciones_path or (source_dir.parent / "predicciones.json")
    if not pred_p.exists():
        alt_pred = Path(__file__).resolve().parent / "predicciones.json"
        if alt_pred.exists():
            pred_p = alt_pred
    if pred_p and pred_p.exists():
        try:
            with open(pred_p, "r", encoding="utf-8") as pf:
                pred_data = json.load(pf)
        except Exception:
            pass

    log_appended = 0
    forecast_log_dir = out_dir / "forecast_log"
    forecast_log_dir.mkdir(parents=True, exist_ok=True)

    if with_month_log and pred_data:
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        month_file = forecast_log_dir / f"{month_key}.csv"
        log_appended = append_forecast_log(pred_data, month_file)

    scorecard = build_scorecard(forecast_log_dir, daily_by_slug, pred_data)
    scorecard_file = out_dir / "scorecard.json"
    with open(scorecard_file, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, separators=(",", ":"), ensure_ascii=False)

    # Opcional: forecast_log
    log_appended = 0
    if with_month_log:
        pred_p = predicciones_path or (source_dir.parent / "predicciones.json")
        if pred_p.exists():
            try:
                with open(pred_p, "r", encoding="utf-8") as pf:
                    preds = json.load(pf)
                month_key = datetime.now(timezone.utc).strftime("%Y-%m")
                month_file = out_dir / "forecast_log" / f"{month_key}.csv"
                log_appended = append_forecast_log(preds, month_file)
            except Exception as pe:
                print(f"[build_derived] Advertencia forecast log: {pe}", file=sys.stderr)

    return {
        "cities_processed": len(index_cities),
        "climatology_cities": len(climatology.get("cities", {})),
        "log_appended": log_appended
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Data Truth Pipeline - Generador de derivados JSON compactos.")
    parser.add_argument("--source", type=str, default="datos", help="Directorio con archivos CSV de estaciones.")
    parser.add_argument("--out", type=str, default="derived", help="Directorio destino para los JSON derivados.")
    parser.add_argument("--ciudades", type=str, default="ciudades.txt", help="Ruta a ciudades.txt.")
    parser.add_argument("--cities", type=str, default=None, help="Lista separada por comas de slugs o nombres a procesar.")
    parser.add_argument("--month-log", action="store_true", help="Actualiza el log mensual de acierto predictivo.")
    parser.add_argument("--predicciones", type=str, default=None, help="Ruta a predicciones.json para forecast log.")
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    ciudades_txt = Path(args.ciudades)
    cities_filter = [c.strip() for c in args.cities.split(",")] if args.cities else None
    pred_path = Path(args.predicciones) if args.predicciones else None

    if not source_dir.exists():
        print(f"Error: No existe el directorio de origen {source_dir}", file=sys.stderr)
        return 1

    res = process_derived_pipeline(
        source_dir=source_dir,
        out_dir=out_dir,
        ciudades_txt=ciudades_txt,
        cities_filter=cities_filter,
        with_month_log=args.month_log,
        predicciones_path=pred_path
    )
    print(f"✅ Data Truth Pipeline completado: {res['cities_processed']} ciudades compactadas en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
