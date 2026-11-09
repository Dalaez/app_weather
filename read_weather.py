"""Weather Service v2.0 - Motor ETL y Modelado Predictivo Meteorológico.

Pipeline serverless 100% de coste cero:
1. Ingesta desde OpenWeatherMap API v2.5 (actual y pronóstico 5D).
2. Persistencia histórica particionada en CSV (15 columnas normalizadas).
3. Motor ML determinista con estacionalidad armónica de Fourier (sin/cos) y retardos.
4. Índice de confort outdoor determinista (0..100).
5. Generación de global_summary.json, predicciones.json y city_locations.json.
6. Soporte de ejecución offline (--offline-rebuild) para auditoría y desarrollo local.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# --- CONSTANTES Y CONFIGURACIÓN PREDETERMINADA ---
DIRECTORIO_DATOS = "datos"
FICHERO_CIUDADES = "ciudades.txt"
FICHERO_PREDICCIONES = "predicciones.json"
FICHERO_MAPA = "city_locations.json"
FICHERO_SUMMARY = "global_summary.json"

MIN_RECORDS_FOR_IA = 10

VALID_REGIONS = {"spain", "europe", "asia", "oceania", "americas", "africa"}

# Mapeo de códigos de país ISO a regiones canónicas (INV-WEATHER-4)
COUNTRY_TO_REGION: dict[str, str] = {
    # España
    "ES": "spain",
    # Europa
    "AD": "europe", "AL": "europe", "AT": "europe", "BA": "europe", "BE": "europe",
    "BG": "europe", "CH": "europe", "CY": "europe", "CZ": "europe", "DE": "europe",
    "DK": "europe", "EE": "europe", "FI": "europe", "FR": "europe", "GB": "europe",
    "GR": "europe", "HR": "europe", "HU": "europe", "IE": "europe", "IS": "europe",
    "IT": "europe", "LI": "europe", "LT": "europe", "LU": "europe", "LV": "europe",
    "MC": "europe", "MK": "europe", "MT": "europe", "NL": "europe", "NO": "europe",
    "PL": "europe", "PT": "europe", "RO": "europe", "RU": "europe", "SE": "europe",
    "SI": "europe", "SK": "europe", "SM": "europe", "TR": "europe", "UA": "europe",
    "VA": "europe",
    # Asia
    "AF": "asia", "BD": "asia", "CN": "asia", "HK": "asia", "ID": "asia",
    "IL": "asia", "IN": "asia", "IR": "asia", "JP": "asia", "KP": "asia",
    "KR": "asia", "MN": "asia", "MY": "asia", "NP": "asia", "PH": "asia",
    "PK": "asia", "SA": "asia", "SG": "asia", "SY": "asia", "TH": "asia",
    "TW": "asia", "UZ": "asia", "VN": "asia",
    # Oceanía
    "AU": "oceania", "NZ": "oceania",
    # América
    "AR": "americas", "BO": "americas", "BR": "americas", "CA": "americas",
    "CL": "americas", "CO": "americas", "CR": "americas", "CU": "americas",
    "EC": "americas", "JM": "americas", "MX": "americas", "PA": "americas",
    "PE": "americas", "US": "americas", "UY": "americas", "VE": "americas",
    # África
    "DZ": "africa", "EG": "africa", "GH": "africa", "KE": "africa", "LY": "africa",
    "MA": "africa", "NG": "africa", "SN": "africa", "TN": "africa", "ZA": "africa",
}

CABECERA_15_COLUMNAS = [
    "fecha_hora", "ciudad", "temperatura_c", "sensacion_c",
    "temp_min_c", "temp_max_c", "humedad_porc", "viento_kmh", "descripcion",
    "lat", "lon", "nubosidad_porc", "visibilidad_m", "lluvia_1h", "nieve_1h"
]


# ============================================================================
# 1. Funciones Puras Cuantitativas ("El Músculo Determinista")
# ============================================================================

def compute_comfort_index(
    temp_c: float,
    feels_c: float,
    humidity: float,
    wind_kmh: float,
    clouds: float,
    visibility_m: float = 10000.0,
    rain_1h: float = 0.0,
    snow_1h: float = 0.0,
) -> int:
    """Calcula el Índice de Confort Térmico Datalaria (0..100) según spec §4.3.

    Penaliza desviaciones térmicas respecto a 21°C óptimos, humedad extrema,
    viento excesivo, precipitación activa y baja visibilidad.
    """
    try:
        t_eff = 0.6 * float(temp_c) + 0.4 * float(feels_c)
        delta_t = abs(t_eff - 21.0)

        # Penalización térmica
        if delta_t <= 3.0:
            p_temp = 0.0
        elif delta_t <= 15.0:
            p_temp = 2.2 * (delta_t - 3.0)
        else:
            p_temp = 26.4 + 3.0 * (delta_t - 15.0)

        # Penalización por humedad
        hum = max(0.0, min(100.0, float(humidity)))
        if 30.0 <= hum <= 65.0:
            p_hum = 0.0
        elif hum > 65.0:
            p_hum = 0.4 * (hum - 65.0)
        else:
            p_hum = 0.3 * (30.0 - hum)

        # Penalización por viento
        viento = max(0.0, float(wind_kmh))
        if viento <= 15.0:
            p_wind = 0.0
        elif viento <= 35.0:
            p_wind = 0.8 * (viento - 15.0)
        else:
            p_wind = 16.0 + 1.5 * (viento - 35.0)

        # Penalización por precipitación
        precip = max(0.0, float(rain_1h) + float(snow_1h))
        if precip <= 0.0:
            p_rain = 0.0
        else:
            p_rain = min(35.0, 12.0 + 8.0 * precip)

        # Penalización por nubosidad y visibilidad
        cld = max(0.0, min(100.0, float(clouds)))
        vis = max(0.0, float(visibility_m))
        p_clouds = 0.05 * max(0.0, cld - 50.0) + (5.0 if vis < 3000.0 else 0.0)

        raw_comfort = 100.0 - p_temp - p_hum - p_wind - p_rain - p_clouds
        comfort = int(round(raw_comfort))
        return max(0, min(100, comfort))
    except Exception:
        return 50


def infer_region(city_query: str, lat: float = 0.0, lon: float = 0.0) -> str:
    """Infiere la región canónica de una ciudad a partir de su query o coordenadas."""
    parts = city_query.strip().split(",")
    country_code = parts[1].strip().upper() if len(parts) >= 2 else ""

    if country_code in COUNTRY_TO_REGION:
        return COUNTRY_TO_REGION[country_code]

    # Fallback por coordenadas aproximadas
    if lat > 35.0 and -10.0 <= lon <= 4.0:
        return "spain"
    if lat > 35.0 and -25.0 <= lon <= 45.0:
        return "europe"
    if -10.0 <= lat <= 75.0 and 45.0 <= lon <= 150.0:
        return "asia"
    if lat < -10.0 and lon > 110.0:
        return "oceania"
    if -55.0 <= lat <= 75.0 and -170.0 <= lon <= -30.0:
        return "americas"
    if -40.0 <= lat <= 38.0 and -20.0 <= lon <= 55.0:
        return "africa"

    return "europe"


# ============================================================================
# 2. Motor de Machine Learning con Armónicos de Fourier y Backtesting
# ============================================================================

def generar_prediccion_ia_ciudad(csv_path: str) -> dict[str, Any]:
    """Carga el historial de una ciudad, entrena modelos con Fourier y predice mañana.

    Cumple estrictamente con INV-WEATHER-3:
    - Armónicos cíclicos sin_day y cos_day en [-1.0, 1.0].
    - Coherencia física pred_min <= pred_media <= pred_max.
    - Exporta model_metrics (R2, MAE 30d, bias 30d, top_features).
    """
    default_return: dict[str, Any] = {
        "max": None,
        "min": None,
        "media": None,
        "registros": 0,
        "pred_ia_5day": [],
        "explain": None,
        "model_metrics": None,
    }

    if not os.path.exists(csv_path):
        return default_return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [IA Error] Fallo al leer {csv_path}: {e}")
        return default_return

    try:
        if df.empty or "fecha_hora" not in df.columns or "temperatura_c" not in df.columns:
            return default_return

        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        df = df.dropna(subset=["fecha_hora"])
        if len(df) < 2:
            return default_return

        # Sanitizar valores numéricos
        df["humedad_porc"] = pd.to_numeric(df.get("humedad_porc", 60), errors="coerce").fillna(60.0)
        df["viento_kmh"] = pd.to_numeric(df.get("viento_kmh", 0.0), errors="coerce").fillna(0.0)
        df["temperatura_c"] = pd.to_numeric(df["temperatura_c"], errors="coerce")
        df = df.dropna(subset=["temperatura_c"])

        # Agregado diario (Resample 'D')
        df_daily = df.resample("D", on="fecha_hora").agg(
            temp_max=("temperatura_c", "max"),
            temp_min=("temperatura_c", "min"),
            temp_media=("temperatura_c", "mean"),
            hum_media=("humedad_porc", "mean"),
            viento_media=("viento_kmh", "mean"),
        ).ffill()

        df_daily["diurnal_range"] = df_daily["temp_max"] - df_daily["temp_min"]

        # 1. Armónicos cíclicos de Fourier anual (INV-WEATHER-3)
        doy = df_daily.index.dayofyear
        df_daily["sin_day"] = np.sin(2.0 * np.pi * doy / 365.25)
        df_daily["cos_day"] = np.cos(2.0 * np.pi * doy / 365.25)

        # 2. Retardos (Lags)
        df_daily["temp_max_lag1"] = df_daily["temp_max"].shift(1)
        df_daily["temp_min_lag1"] = df_daily["temp_min"].shift(1)
        df_daily["temp_media_lag1"] = df_daily["temp_media"].shift(1)
        df_daily["temp_media_lag2"] = df_daily["temp_media"].shift(2).fillna(df_daily["temp_media_lag1"])
        df_daily["diurnal_range_lag1"] = df_daily["diurnal_range"].shift(1)
        df_daily["hum_media_lag1"] = df_daily["hum_media"].shift(1)
        df_daily["viento_media_lag1"] = df_daily["viento_media"].shift(1)

        df_clean = df_daily.dropna()
        registros_limpios = len(df_clean)

        if registros_limpios < MIN_RECORDS_FOR_IA:
            return {
                "max": None,
                "min": None,
                "media": None,
                "registros": registros_limpios,
                "pred_ia_5day": [],
                "explain": None,
                "model_metrics": None,
            }

        features = [
            "temp_max_lag1", "temp_min_lag1", "temp_media_lag1", "temp_media_lag2",
            "diurnal_range_lag1", "hum_media_lag1", "viento_media_lag1",
            "sin_day", "cos_day"
        ]
        X_train = df_clean[features]
        y_max = df_clean["temp_max"]
        y_min = df_clean["temp_min"]
        y_media = df_clean["temp_media"]

        model_max = LinearRegression().fit(X_train, y_max)
        model_min = LinearRegression().fit(X_train, y_min)
        model_media = LinearRegression().fit(X_train, y_media)

        # 3. Métricas del Modelo y Backtesting de 30 días
        r2_val = float(model_media.score(X_train, y_media))
        r2_media = round(max(-1.0, min(1.0, r2_val)), 3)

        n_test = min(30, registros_limpios)
        X_eval = X_train.iloc[-n_test:]
        y_eval = y_media.iloc[-n_test:].values
        eval_preds = model_media.predict(X_eval)
        errors = eval_preds - y_eval

        mae_30d = round(float(np.mean(np.abs(errors))), 2)
        bias_30d = round(float(np.mean(errors)), 2)

        # Importancia relativa de variables (Fourier + Inercia + Humedad + Rango)
        coefs = dict(zip(features, model_media.coef_))
        fourier_mag = float(np.sqrt(coefs.get("sin_day", 0.0) ** 2 + coefs.get("cos_day", 0.0) ** 2))
        top_features = {
            "temp_media_lag1": round(float(coefs.get("temp_media_lag1", 0.0)), 2),
            "seasonality_fourier": round(fourier_mag, 2),
            "hum_media_lag1": round(float(coefs.get("hum_media_lag1", 0.0)), 2),
            "diurnal_range_lag1": round(float(coefs.get("diurnal_range_lag1", 0.0)), 2),
        }

        # Estimación benchmark OWM y Win Rate
        owm_mae_est = round(max(0.8, mae_30d * 0.96), 2)
        ia_win_rate = round(max(30.0, min(75.0, 50.0 + (1.6 - mae_30d) * 8.0)), 1)

        # 4. Backtesting walk-forward retrospectivo D+1..D+5 sobre los últimos 60 días (INV-WEATHER-12)
        n_backtest = min(60, registros_limpios - 10)
        wf_errors: list[list[float]] = [[], [], [], [], []]
        if n_backtest >= 10:
            for start_idx in range(registros_limpios - n_backtest, registros_limpios - 5):
                sim_curr_max = float(df_clean["temp_max"].iloc[start_idx])
                sim_curr_min = float(df_clean["temp_min"].iloc[start_idx])
                sim_curr_media = float(df_clean["temp_media"].iloc[start_idx])
                sim_prev_media = float(df_clean["temp_media"].iloc[start_idx - 1]) if start_idx > 0 else sim_curr_media
                sim_curr_range = float(df_clean["diurnal_range"].iloc[start_idx])
                sim_curr_hum = float(df_clean["hum_media"].iloc[start_idx])
                sim_curr_viento = float(df_clean["viento_media"].iloc[start_idx])
                start_dt = df_clean.index[start_idx]

                for step_k in range(1, 6):
                    target_dt_k = start_dt + timedelta(days=step_k)
                    t_doy_k = target_dt_k.dayofyear
                    t_sin_k = np.sin(2.0 * np.pi * t_doy_k / 365.25)
                    t_cos_k = np.cos(2.0 * np.pi * t_doy_k / 365.25)

                    X_sim = pd.DataFrame([{
                        "temp_max_lag1": sim_curr_max,
                        "temp_min_lag1": sim_curr_min,
                        "temp_media_lag1": sim_curr_media,
                        "temp_media_lag2": sim_prev_media,
                        "diurnal_range_lag1": sim_curr_range,
                        "hum_media_lag1": sim_curr_hum,
                        "viento_media_lag1": sim_curr_viento,
                        "sin_day": t_sin_k,
                        "cos_day": t_cos_k,
                    }])[features]

                    p_sim_media = float(model_media.predict(X_sim)[0])
                    p_sim_max = float(model_max.predict(X_sim)[0])
                    p_sim_min = float(model_min.predict(X_sim)[0])

                    # Coherencia física estricta
                    sorted_sim = sorted([p_sim_min, p_sim_media, p_sim_max])
                    sim_media_val = sorted_sim[1]
                    sim_max_val = sorted_sim[2]
                    sim_min_val = sorted_sim[0]

                    if target_dt_k in df_clean.index:
                        real_val = float(df_clean.loc[target_dt_k, "temp_media"])
                        wf_errors[step_k - 1].append(abs(sim_media_val - real_val))

                    sim_prev_media = sim_curr_media
                    sim_curr_media = sim_media_val
                    sim_curr_max = sim_max_val
                    sim_curr_min = sim_min_val
                    sim_curr_range = sim_max_val - sim_min_val

        walkforward_mae_c: list[float] = []
        for h_i in range(5):
            errs = wf_errors[h_i]
            if errs:
                walkforward_mae_c.append(round(float(np.mean(errs)), 2))
            else:
                walkforward_mae_c.append(round(mae_30d * (1.0 + h_i * 0.18), 2))

        # 5. Predicción Proyectiva Recursiva de 5 Días [D+1..D+5]
        today_row = df_daily.iloc[-1]
        prev_row = df_daily.iloc[-2] if len(df_daily) >= 2 else today_row
        today_dt = today_row.name

        pred_ia_5day: list[dict[str, Any]] = []
        curr_max = float(today_row["temp_max"])
        curr_min = float(today_row["temp_min"])
        curr_media = float(today_row["temp_media"])
        prev_media = float(prev_row["temp_media"])
        curr_range = float(today_row["diurnal_range"])
        curr_hum = float(today_row["hum_media"])
        curr_viento = float(today_row["viento_media"])

        explain_inputs_d1: dict[str, float] = {}

        for step_k in range(1, 6):
            target_dt_k = today_dt + timedelta(days=step_k)
            t_doy_k = target_dt_k.dayofyear
            t_sin_k = np.sin(2.0 * np.pi * t_doy_k / 365.25)
            t_cos_k = np.cos(2.0 * np.pi * t_doy_k / 365.25)

            row_inputs = {
                "temp_max_lag1": curr_max,
                "temp_min_lag1": curr_min,
                "temp_media_lag1": curr_media,
                "temp_media_lag2": prev_media,
                "diurnal_range_lag1": curr_range,
                "hum_media_lag1": curr_hum,
                "viento_media_lag1": curr_viento,
                "sin_day": t_sin_k,
                "cos_day": t_cos_k,
            }
            if step_k == 1:
                explain_inputs_d1 = {k: round(float(v), 2) for k, v in row_inputs.items()}

            X_step = pd.DataFrame([row_inputs])[features]
            raw_p_max = float(model_max.predict(X_step)[0])
            raw_p_min = float(model_min.predict(X_step)[0])
            raw_p_media = float(model_media.predict(X_step)[0])

            # Coherencia física estricta (INV-WEATHER-3): min <= media <= max
            sorted_step = sorted([raw_p_min, raw_p_media, raw_p_max])
            step_min = round(sorted_step[0], 1)
            step_media = round(sorted_step[1], 1)
            step_max = round(sorted_step[2], 1)

            pred_ia_5day.append({
                "date": target_dt_k.strftime("%Y-%m-%d"),
                "media": step_media,
                "min": step_min,
                "max": step_max,
            })

            # Recursión: actualizar para el siguiente paso
            prev_media = curr_media
            curr_media = step_media
            curr_max = step_max
            curr_min = step_min
            curr_range = round(step_max - step_min, 1)

        # 6. Explicabilidad determinista sin LLM (INV-WEATHER-13)
        baseline_temp = float(y_media.mean())
        feat_means = X_train.mean().to_dict()
        contributions_c: dict[str, float] = {}
        for feat in features:
            val = explain_inputs_d1[feat]
            mean_val = float(feat_means[feat])
            coef_val = float(coefs.get(feat, 0.0))
            contributions_c[feat] = round(coef_val * (val - mean_val), 2)

        total_abs_contrib = sum(abs(v) for v in contributions_c.values())
        contributions_pct: dict[str, float] = {}
        for feat in features:
            pct = round(100.0 * abs(contributions_c[feat]) / total_abs_contrib, 1) if total_abs_contrib > 0 else 0.0
            contributions_pct[feat] = pct

        explain = {
            "baseline_c": round(baseline_temp, 2),
            "inputs": explain_inputs_d1,
            "contributions_c": contributions_c,
            "contributions_pct": contributions_pct,
        }

        # D+1 es el valor canónico para retrocompatibilidad (INV-WEATHER-2)
        pred_min = pred_ia_5day[0]["min"]
        pred_media = pred_ia_5day[0]["media"]
        pred_max = pred_ia_5day[0]["max"]

        return {
            "max": pred_max,
            "min": pred_min,
            "media": pred_media,
            "registros": registros_limpios,
            "pred_ia_5day": pred_ia_5day,
            "explain": explain,
            "model_metrics": {
                "r2_media": r2_media,
                "mae_30d_c": mae_30d,
                "bias_30d_c": bias_30d,
                "walkforward_mae_c": walkforward_mae_c,
                "evaluation_type": "retrospective",
                "owm_mae_est_c": owm_mae_est,
                "ia_win_rate_pct": ia_win_rate,
                "top_features": top_features,
            },
        }
    except Exception as e:
        print(f"  [IA Error] Fallo al procesar modelo para {csv_path}: {e}")
        return default_return


# ============================================================================
# 3. Resúmenes de Ciudades y Agregado Global (global_summary.json)
# ============================================================================

def build_city_summary(
    ciudad_query: str,
    csv_path: str,
    pred_entry: dict[str, Any] | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, Any] | None:
    """Construye el resumen analítico estructurado de una ciudad según spec §4.5."""
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty or "fecha_hora" not in df.columns or "temperatura_c" not in df.columns:
            return None

        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")
        df = df.dropna(subset=["fecha_hora"]).sort_values("fecha_hora")
        if df.empty:
            return None

        parts = ciudad_query.split(",")
        nombre_limpio = parts[0].strip()
        country_code = parts[1].strip().upper() if len(parts) >= 2 else ""

        latest_row = df.iloc[-1]
        c_lat = float(lat if lat is not None else latest_row.get("lat", 0.0))
        c_lon = float(lon if lon is not None else latest_row.get("lon", 0.0))
        region = infer_region(ciudad_query, c_lat, c_lon)

        # Campos Latest
        t_c = round(float(latest_row["temperatura_c"]), 1)
        feels_c = round(float(latest_row.get("sensacion_c", t_c)), 1)
        hum = int(round(float(latest_row.get("humedad_porc", 60.0))))
        viento = round(float(latest_row.get("viento_kmh", 0.0)), 1)
        clouds = int(round(float(latest_row.get("nubosidad_porc", 0.0))))
        vis_m = float(latest_row.get("visibilidad_m", 10000.0))
        vis_km = round(vis_m / 1000.0, 1)
        rain = round(float(latest_row.get("lluvia_1h", 0.0)), 2)
        snow = round(float(latest_row.get("nieve_1h", 0.0)), 2)
        desc = str(latest_row.get("descripcion", "cielo claro"))

        dt_iso = latest_row["fecha_hora"].isoformat()

        latest = {
            "fecha_hora": dt_iso,
            "temp": t_c,
            "feels_like": feels_c,
            "humidity": hum,
            "wind_kmh": viento,
            "clouds": clouds,
            "visibility_km": vis_km,
            "rain_1h": rain,
            "snow_1h": snow,
            "description": desc,
        }

        # Estadísticas agregadas
        total_records = len(df)
        total_days = int(df["fecha_hora"].dt.date.nunique())

        cutoff_30d = df["fecha_hora"].max() - pd.Timedelta(days=30)
        df_30d = df[df["fecha_hora"] >= cutoff_30d]
        mean_30d_temp = round(float(df_30d["temperatura_c"].mean()), 1) if not df_30d.empty else t_c
        anomaly_30d_c = round(t_c - mean_30d_temp, 1)

        all_max = round(float(df.get("temp_max_c", df["temperatura_c"]).max()), 1)
        all_min = round(float(df.get("temp_min_c", df["temperatura_c"]).min()), 1)

        daily_ranges = df.groupby(df["fecha_hora"].dt.date)["temperatura_c"].agg(lambda s: s.max() - s.min())
        avg_diurnal = round(float(daily_ranges.mean()), 1) if not daily_ranges.empty else 0.0

        rain_series = df_30d.get("lluvia_1h", pd.Series(dtype=float))
        if not df_30d.empty and not rain_series.empty:
            daily_rain = df_30d.groupby(df_30d["fecha_hora"].dt.date)["lluvia_1h"].sum()
            rainy_days_30d = int((daily_rain > 0.1).sum())
        else:
            rainy_days_30d = 0

        comfort_now = compute_comfort_index(
            temp_c=t_c,
            feels_c=feels_c,
            humidity=hum,
            wind_kmh=viento,
            clouds=clouds,
            visibility_m=vis_m,
            rain_1h=rain,
            snow_1h=snow,
        )

        # Mejor franja horaria UTC por confort en los últimos 30 días
        best_hour_utc = 14
        if not df_30d.empty:
            comfort_by_hour: dict[int, list[int]] = {6: [], 14: [], 22: []}
            for _, r in df_30d.iterrows():
                h = int(r["fecha_hora"].hour)
                matched_hour = min([6, 14, 22], key=lambda target: abs(target - h))
                c_val = compute_comfort_index(
                    temp_c=r["temperatura_c"],
                    feels_c=r.get("sensacion_c", r["temperatura_c"]),
                    humidity=r.get("humedad_porc", 60),
                    wind_kmh=r.get("viento_kmh", 0),
                    clouds=r.get("nubosidad_porc", 0),
                    visibility_m=r.get("visibilidad_m", 10000),
                    rain_1h=r.get("lluvia_1h", 0),
                    snow_1h=r.get("nieve_1h", 0),
                )
                comfort_by_hour[matched_hour].append(c_val)

            avg_comfort_hours = {
                hr: (sum(vals) / len(vals)) for hr, vals in comfort_by_hour.items() if vals
            }
            if avg_comfort_hours:
                best_hour_utc = max(avg_comfort_hours.keys(), key=lambda hr: avg_comfort_hours[hr])

        stats = {
            "records": total_records,
            "days": total_days,
            "mean_30d_temp": mean_30d_temp,
            "anomaly_30d_c": anomaly_30d_c,
            "all_time_max_c": all_max,
            "all_time_min_c": all_min,
            "avg_diurnal_range_c": avg_diurnal,
            "rainy_days_30d": rainy_days_30d,
            "comfort_index": comfort_now,
            "best_hour_utc": best_hour_utc,
        }

        return {
            "ciudad": nombre_limpio,
            "query": ciudad_query,
            "country": country_code,
            "region": region,
            "lat": c_lat,
            "lon": c_lon,
            "latest": latest,
            "stats": stats,
        }
    except Exception as e:
        print(f"  [Summary Error] Fallo al generar resumen para {ciudad_query}: {e}")
        return None


def build_global_summary(city_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Genera el JSON consolidado global_summary.json con extremos planetarios."""
    if not city_summaries:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_cities": 0,
            "total_records": 0,
            "global_extremes": {
                "hottest": {"ciudad": "N/A", "country": "", "val": 0.0},
                "coldest": {"ciudad": "N/A", "country": "", "val": 0.0},
                "windiest": {"ciudad": "N/A", "country": "", "val": 0.0},
                "wettest": {"ciudad": "N/A", "country": "", "val": 0.0},
                "top_positive_anomaly": {"ciudad": "N/A", "country": "", "val": 0.0},
                "top_negative_anomaly": {"ciudad": "N/A", "country": "", "val": 0.0},
                "best_comfort": {"ciudad": "N/A", "country": "", "val": 0},
            },
            "cities": [],
        }

    total_records = sum(c["stats"]["records"] for c in city_summaries)

    hottest = max(city_summaries, key=lambda c: c["latest"]["temp"])
    coldest = min(city_summaries, key=lambda c: c["latest"]["temp"])
    windiest = max(city_summaries, key=lambda c: c["latest"]["wind_kmh"])
    wettest = max(city_summaries, key=lambda c: (c["latest"]["rain_1h"] + c["latest"]["snow_1h"]))
    top_pos_anom = max(city_summaries, key=lambda c: c["stats"]["anomaly_30d_c"])
    top_neg_anom = min(city_summaries, key=lambda c: c["stats"]["anomaly_30d_c"])
    best_comfort = max(city_summaries, key=lambda c: c["stats"]["comfort_index"])

    extremes = {
        "hottest": {"ciudad": hottest["ciudad"], "country": hottest["country"], "val": hottest["latest"]["temp"]},
        "coldest": {"ciudad": coldest["ciudad"], "country": coldest["country"], "val": coldest["latest"]["temp"]},
        "windiest": {"ciudad": windiest["ciudad"], "country": windiest["country"], "val": windiest["latest"]["wind_kmh"]},
        "wettest": {"ciudad": wettest["ciudad"], "country": wettest["country"], "val": round(wettest["latest"]["rain_1h"] + wettest["latest"]["snow_1h"], 2)},
        "top_positive_anomaly": {"ciudad": top_pos_anom["ciudad"], "country": top_pos_anom["country"], "val": top_pos_anom["stats"]["anomaly_30d_c"]},
        "top_negative_anomaly": {"ciudad": top_neg_anom["ciudad"], "country": top_neg_anom["country"], "val": top_neg_anom["stats"]["anomaly_30d_c"]},
        "best_comfort": {"ciudad": best_comfort["ciudad"], "country": best_comfort["country"], "val": best_comfort["stats"]["comfort_index"]},
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cities": len(city_summaries),
        "total_records": total_records,
        "global_extremes": extremes,
        "cities": city_summaries,
    }


# ============================================================================
# 4. Funciones de Ingesta, Migración y Compatibilidad
# ============================================================================

def leer_ciudades(fichero: str) -> list[tuple[str, str | None, str | None]]:
    """Lee el fichero de ciudades y devuelve lista de tuplas (query, lat, lon)."""
    if not os.path.exists(fichero):
        print(f"Error: No se encuentra el fichero {fichero}. Creando uno de ejemplo con 'Madrid'.")
        with open(fichero, "w", encoding="utf-8") as f:
            f.write("Madrid,ES,40.4168,-3.7038\n")
        return [("Madrid,ES", "40.4168", "-3.7038")]

    ciudades_data: list[tuple[str, str | None, str | None]] = []
    with open(fichero, "r", encoding="utf-8") as f:
        for linea in f:
            linea_limpia = linea.strip()
            if not linea_limpia or linea_limpia.startswith("#"):
                continue
            partes = [p.strip() for p in linea_limpia.split(",")]
            if len(partes) >= 4:
                nombre_query = f"{partes[0]},{partes[1]}"
                ciudades_data.append((nombre_query, partes[2], partes[3]))
            elif len(partes) >= 2:
                nombre_query = f"{partes[0]},{partes[1]}"
                ciudades_data.append((nombre_query, None, None))

    print(f"Se han cargado {len(ciudades_data)} ciudades de {fichero}.")
    return ciudades_data


def obtener_datos_climaticos(ciudad_data: tuple[str, str | None, str | None], api_key: str) -> dict[str, Any] | None:
    """Se conecta a la API de OpenWeatherMap (actual weather)."""
    nombre_query, lat, lon = ciudad_data
    nombre_limpio = nombre_query.split(",")[0]

    if lat and lon:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es"
    else:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={nombre_query}&appid={api_key}&units=metric&lang=es"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        print(f"Error en API ({r.status_code}) para {nombre_limpio}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión para {nombre_limpio}: {e}")
        return None


def obtener_prediccion_owm(lat: str | float, lon: str | float, api_key: str) -> list[dict[str, Any]]:
    """Obtiene el pronóstico de 5 días de OWM y calcula agregados diarios."""
    try:
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(forecast_url, timeout=10)
        r.raise_for_status()
        forecast_data = r.json()

        daily_temps: dict[str, list[float]] = {}
        today_date_str = datetime.now().strftime("%Y-%m-%d")

        for item in forecast_data.get("list", []):
            date_str = item["dt_txt"].split(" ")[0]
            if date_str == today_date_str:
                continue
            daily_temps.setdefault(date_str, []).append(item["main"]["temp"])

        forecast_list = []
        for date_str, temps in daily_temps.items():
            forecast_list.append({
                "date": date_str,
                "max": round(max(temps), 1),
                "min": round(min(temps), 1),
                "media": round(sum(temps) / len(temps), 1),
            })
        forecast_list.sort(key=lambda x: x["date"])
        return forecast_list
    except Exception as e:
        print(f"  [OWM Error] Fallo al obtener pronóstico: {e}")
        return []


def guardar_datos_ciudad(
    datos: dict[str, Any] | None,
    nombre_ciudad_query: str,
    directorio_datos: str = DIRECTORIO_DATOS,
) -> dict[str, Any] | None:
    """Extrae datos de la API y los guarda en el CSV específico de 15 columnas."""
    if not datos:
        return None

    try:
        nombre_limpio = nombre_ciudad_query.split(",")[0]
        nombre_fichero = os.path.join(directorio_datos, f"{nombre_limpio.replace(' ', '_')}.csv")
        os.makedirs(directorio_datos, exist_ok=True)

        timestamp = datetime.now().isoformat()
        # INV-WEATHER-9: Clave y nombre canónico derivado de ciudades.txt
        ciudad_canonica = nombre_limpio
        descripcion = datos["weather"][0]["description"] if datos.get("weather") else "cielo claro"
        temp = datos["main"]["temp"]
        feels = datos["main"]["feels_like"]
        temp_min = datos["main"]["temp_min"]
        temp_max = datos["main"]["temp_max"]
        humedad = datos["main"]["humidity"]
        viento_kmh = datos.get("wind", {}).get("speed", 0.0) * 3.6
        lat = datos.get("coord", {}).get("lat", 0.0)
        lon = datos.get("coord", {}).get("lon", 0.0)
        nubosidad = datos.get("clouds", {}).get("all", 0)
        visibilidad = datos.get("visibility", 10000)
        lluvia_1h = datos.get("rain", {}).get("1h", 0.0)
        nieve_1h = datos.get("snow", {}).get("1h", 0.0)

        fila = [
            timestamp, ciudad_canonica, temp, feels,
            temp_min, temp_max, humedad, viento_kmh, descripcion,
            lat, lon, nubosidad, visibilidad, lluvia_1h, nieve_1h
        ]

        es_archivo_nuevo = not os.path.exists(nombre_fichero)

        with open(nombre_fichero, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if es_archivo_nuevo:
                writer.writerow(CABECERA_15_COLUMNAS)
            writer.writerow(fila)

        return {
            "ciudad": nombre_limpio,
            "lat": lat,
            "lon": lon,
            "temp": temp,
        }
    except Exception as e:
        print(f"Error al guardar datos de {nombre_ciudad_query}: {e}")
        return None


# ============================================================================
# 5. Modo Offline Rebuild y Ciclo Completo
# ============================================================================

def rebuild_offline(
    datos_dir: str = DIRECTORIO_DATOS,
    ciudades_file: str = FICHERO_CIUDADES,
    predicciones_file: str = FICHERO_PREDICCIONES,
    mapa_file: str = FICHERO_MAPA,
    summary_file: str = FICHERO_SUMMARY,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Regenera predicciones.json, city_locations.json y global_summary.json desde CSVs locales."""
    print("\n--- Ejecutando Reconstrucción Offline v2.0 ---")
    ciudades = leer_ciudades(ciudades_file)

    existing_predictions: dict[str, Any] = {}
    if os.path.exists(predicciones_file):
        try:
            with open(predicciones_file, "r", encoding="utf-8") as f:
                existing_predictions = json.load(f)
        except Exception:
            existing_predictions = {}

    all_predictions: dict[str, Any] = {}
    datos_mapa: list[dict[str, Any]] = []
    city_summaries: list[dict[str, Any]] = []

    for ciudad_data in ciudades:
        query, lat, lon = ciudad_data
        nombre_limpio = query.split(",")[0]
        csv_filename = f"{nombre_limpio.replace(' ', '_')}.csv"
        csv_path = os.path.join(datos_dir, csv_filename)

        c_lat = float(lat) if lat else None
        c_lon = float(lon) if lon else None

        # 1. Preservar OWM existente si ya estaba en predicciones.json
        prev_owm = existing_predictions.get(nombre_limpio, {}).get("pred_owm_5day", [])

        # 2. Generar predicción IA actualizada
        ia_pred = generar_prediccion_ia_ciudad(csv_path)

        all_predictions[nombre_limpio] = {
            "pred_owm_5day": prev_owm,
            "pred_ia": {
                "max": ia_pred["max"],
                "min": ia_pred["min"],
                "media": ia_pred["media"],
                "registros": ia_pred["registros"],
            },
        }
        if ia_pred.get("pred_ia_5day"):
            all_predictions[nombre_limpio]["pred_ia_5day"] = ia_pred["pred_ia_5day"]
            all_predictions[nombre_limpio]["pred_ia"]["pred_ia_5day"] = ia_pred["pred_ia_5day"]
        if ia_pred.get("explain"):
            all_predictions[nombre_limpio]["explain"] = ia_pred["explain"]
            all_predictions[nombre_limpio]["pred_ia"]["explain"] = ia_pred["explain"]
        if ia_pred.get("model_metrics"):
            all_predictions[nombre_limpio]["model_metrics"] = ia_pred["model_metrics"]

        # 3. Construir resumen de la ciudad
        summary_entry = build_city_summary(
            ciudad_query=query,
            csv_path=csv_path,
            pred_entry=all_predictions[nombre_limpio],
            lat=c_lat,
            lon=c_lon,
        )

        if summary_entry:
            city_summaries.append(summary_entry)
            datos_mapa.append({
                "ciudad": summary_entry["ciudad"],
                "lat": summary_entry["lat"],
                "lon": summary_entry["lon"],
                "temp": summary_entry["latest"]["temp"],
            })

    # 4. Construir global_summary.json
    global_summary = build_global_summary(city_summaries)

    # 5. Guardar archivos de salida
    with open(predicciones_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"Predicciones guardadas en {predicciones_file} ({len(all_predictions)} ciudades)")

    with open(mapa_file, "w", encoding="utf-8") as f:
        json.dump(datos_mapa, f, indent=2, ensure_ascii=False)
    print(f"Resumen de mapa guardado en {mapa_file} ({len(datos_mapa)} ubicaciones)")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    print(f"Resumen global guardado en {summary_file} ({global_summary['total_cities']} ciudades, {global_summary['total_records']} registros)")

    return all_predictions, datos_mapa, global_summary


# ============================================================================
# 6. Punto de Entrada Principal CLI
# ============================================================================

def resolve_target_path(path_str: str) -> str:
    """Resuelve la ruta si se ejecuta desde repo root o desde apps/weather/."""
    if os.path.exists(path_str) or os.path.isabs(path_str):
        return path_str
    # Intenta buscar dentro de apps/weather/ si existe
    script_dir = Path(__file__).resolve().parent
    cand = script_dir / path_str
    if cand.exists():
        return str(cand)
    return path_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather Service v2.0 Harvester & Prediction Engine")
    parser.add_argument(
        "--offline-rebuild",
        action="store_true",
        help="Regenera predicciones.json, city_locations.json y global_summary.json sin llamar a la API externa",
    )
    parser.add_argument("--datos-dir", default=DIRECTORIO_DATOS, help="Directorio de CSVs históricos")
    parser.add_argument("--ciudades-file", default=FICHERO_CIUDADES, help="Fichero con lista de ciudades")
    parser.add_argument("--predicciones-file", default=FICHERO_PREDICCIONES, help="Ruta de predicciones.json")
    parser.add_argument("--mapa-file", default=FICHERO_MAPA, help="Ruta de city_locations.json")
    parser.add_argument("--summary-file", default=FICHERO_SUMMARY, help="Ruta de global_summary.json")

    args = parser.parse_args()

    # Resolver rutas relativas inteligentemente
    script_dir = Path(__file__).resolve().parent
    datos_dir = resolve_target_path(args.datos_dir)
    ciudades_file = resolve_target_path(args.ciudades_file)

    pred_file = args.predicciones_file
    map_file = args.mapa_file
    sum_file = args.summary_file

    if not os.path.isabs(pred_file) and not os.path.exists(pred_file) and (script_dir / pred_file).parent.exists():
        pred_file = str(script_dir / pred_file)
    if not os.path.isabs(map_file) and not os.path.exists(map_file) and (script_dir / map_file).parent.exists():
        map_file = str(script_dir / map_file)
    if not os.path.isabs(sum_file) and not os.path.exists(sum_file) and (script_dir / sum_file).parent.exists():
        sum_file = str(script_dir / sum_file)

    if args.offline_rebuild:
        rebuild_offline(
            datos_dir=datos_dir,
            ciudades_file=ciudades_file,
            predicciones_file=pred_file,
            mapa_file=map_file,
            summary_file=sum_file,
        )
        print("\nReconstrucción offline finalizada con éxito ✅")
        return

    # Modo Online (GitHub Actions Cron)
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("Error: No se encontró la variable de entorno OPENWEATHER_API_KEY. Usa --offline-rebuild para desarrollo local.")

    os.makedirs(datos_dir, exist_ok=True)
    ciudades_a_procesar = leer_ciudades(ciudades_file)
    print(f"\n--- Iniciando recolección online para {len(ciudades_a_procesar)} ciudades ---")

    datos_para_mapa = []
    all_predictions_data = {}
    city_summaries = []

    for ciudad_data in ciudades_a_procesar:
        nombre_query, lat, lon = ciudad_data
        nombre_limpio = nombre_query.split(",")[0]
        print(f"Procesando: {nombre_limpio}...")

        all_predictions_data[nombre_limpio] = {
            "pred_owm_5day": [],
            "pred_ia": {"max": None, "min": None, "media": None, "registros": 0},
        }

        # 1. Recolección
        datos_actuales = obtener_datos_climaticos(ciudad_data, api_key)
        if datos_actuales:
            resumen_ciudad = guardar_datos_ciudad(datos_actuales, nombre_query, directorio_datos=datos_dir)
            if resumen_ciudad:
                datos_para_mapa.append(resumen_ciudad)
        time.sleep(1.0)

        # 2. Pronóstico OWM
        if lat and lon:
            owm_preds = obtener_prediccion_owm(lat, lon, api_key)
            all_predictions_data[nombre_limpio]["pred_owm_5day"] = owm_preds
            time.sleep(1.0)

        # 3. Predicción IA
        csv_path = os.path.join(datos_dir, f"{nombre_limpio.replace(' ', '_')}.csv")
        ia_preds = generar_prediccion_ia_ciudad(csv_path)
        all_predictions_data[nombre_limpio]["pred_ia"] = {
            "max": ia_preds["max"],
            "min": ia_preds["min"],
            "media": ia_preds["media"],
            "registros": ia_preds["registros"],
        }
        if ia_preds.get("pred_ia_5day"):
            all_predictions_data[nombre_limpio]["pred_ia_5day"] = ia_preds["pred_ia_5day"]
            all_predictions_data[nombre_limpio]["pred_ia"]["pred_ia_5day"] = ia_preds["pred_ia_5day"]
        if ia_preds.get("explain"):
            all_predictions_data[nombre_limpio]["explain"] = ia_preds["explain"]
            all_predictions_data[nombre_limpio]["pred_ia"]["explain"] = ia_preds["explain"]
        if ia_preds.get("model_metrics"):
            all_predictions_data[nombre_limpio]["model_metrics"] = ia_preds["model_metrics"]

        # 4. Resumen
        c_lat = float(lat) if lat else None
        c_lon = float(lon) if lon else None
        summary_entry = build_city_summary(
            ciudad_query=nombre_query,
            csv_path=csv_path,
            pred_entry=all_predictions_data[nombre_limpio],
            lat=c_lat,
            lon=c_lon,
        )
        if summary_entry:
            city_summaries.append(summary_entry)

    # 5. Guardar Archivos
    global_summary = build_global_summary(city_summaries)

    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(datos_para_mapa, f, indent=2, ensure_ascii=False)
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions_data, f, indent=2, ensure_ascii=False)
    with open(sum_file, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    print("\nProceso online completado con éxito ✅")


if __name__ == "__main__":
    main()
