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
from datetime import datetime, timezone
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

    # Mapeo de columnas español canónico a estándar interno
    col_map = {
        "temperatura_c": "temp",
        "sensacion_c": "feels_like",
        "humedad_porc": "humidity",
        "viento_kmh": "wind_kmh",
        "nubosidad_porc": "clouds",
        "visibilidad_m": "visibility",
    }
    for es_col, std_col in col_map.items():
        if es_col in df_clean.columns and std_col not in df_clean.columns:
            df_clean[std_col] = df_clean[es_col]

    # Coerción numérica segura
    numeric_cols = ["temp", "temp_min", "temp_max", "feels_like", "humidity", "wind_kmh", "clouds", "rain_1h"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        else:
            df_clean[col] = 0.0

    daily_rows: list[list[Any]] = []
    for date_str, group in df_clean.groupby("date", sort=True):
        n_samples = len(group)
        t_min_val = group["temp_min"].min()
        t_main_min = group["temp"].min()
        tmin = float(min(t_min_val, t_main_min))

        t_max_val = group["temp_max"].max()
        t_main_max = group["temp"].max()
        tmax = float(max(t_max_val, t_main_max))

        tmean = float(group["temp"].mean())
        feels = float(group["feels_like"].mean())
        hum = float(group["humidity"].mean())
        wind = float(group["wind_kmh"].mean())
        clouds = float(group["clouds"].mean())
        rain = float(group["rain_1h"].sum()) if "rain_1h" in group.columns else 0.0

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
                rain_days = sum(1 for r in month_rows if (r[8] or 0.0) > 0.1)
                hum_m = float(np.mean([r[6] for r in month_rows]))
                wind_m = float(np.mean([r[7] for r in month_rows]))
                clouds_m = float(np.mean([r[9] for r in month_rows]))
                comfort_m = compute_comfort_index(tmean_m, tmean_m, hum_m, wind_m, clouds_m)
                months_dict[str(m)] = {
                    "tmin": tmin_m,
                    "tmax": tmax_m,
                    "tmean": tmean_m,
                    "rain_days": int(rain_days),
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

    return {
        "generated_at": now_iso,
        "cities": cities_climatology
    }


def append_forecast_log(predicciones: dict[str, Any], month_file: Path) -> int:
    """Añade los pronósticos diarios al log mensual en CSV de forma idempotente."""
    month_file.parent.mkdir(parents=True, exist_ok=True)
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Leer registros existentes si el fichero ya existe
    existing_pairs: set[tuple[str, str]] = set()
    header = "fecha_emision,slug,ia_d1,ia_d2,ia_d3,ia_d4,ia_d5,owm_d1,owm_d2,owm_d3,owm_d4,owm_d5\n"
    if month_file.exists():
        with open(month_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    existing_pairs.add((parts[0], parts[1]))
    else:
        with open(month_file, "w", encoding="utf-8") as f:
            f.write(header)

    added_count = 0
    new_lines: list[str] = []

    for city_name, data in predicciones.items():
        slug = slugify(city_name)
        pair = (today_iso, slug)
        if pair in existing_pairs:
            continue

        # Extraer predicción IA (mañana = D+1)
        pred_ia = data.get("pred_ia", {})
        ia_d1 = pred_ia.get("media", "") if isinstance(pred_ia, dict) else ""
        ia_d2, ia_d3, ia_d4, ia_d5 = "", "", "", ""

        # Extraer OWM 5D
        owm_5d = data.get("pred_owm_5day", [])
        owm_vals = ["", "", "", "", ""]
        if isinstance(owm_5d, list):
            for i, item in enumerate(owm_5d[:5]):
                if isinstance(item, dict) and "media" in item:
                    owm_vals[i] = str(item["media"])

        row_str = f"{today_iso},{slug},{ia_d1},{ia_d2},{ia_d3},{ia_d4},{ia_d5}," + ",".join(owm_vals) + "\n"
        new_lines.append(row_str)
        existing_pairs.add(pair)
        added_count += 1

    if new_lines:
        with open(month_file, "a", encoding="utf-8") as f:
            f.writelines(new_lines)

    return added_count


def process_derived_pipeline(source_dir: Path, out_dir: Path,
                             ciudades_txt: Path,
                             cities_filter: list[str] | None = None,
                             with_month_log: bool = False,
                             predicciones_path: Path | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo de derivación produciendo daily/, climatology.json e index.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_out_dir = out_dir / "daily"
    daily_out_dir.mkdir(parents=True, exist_ok=True)

    catalog = parse_ciudades_txt(ciudades_txt)
    name_to_meta = {c["name"]: c for c in catalog}
    slug_to_meta = {c["slug"]: c for c in catalog}

    filter_slugs = set(slugify(c) for c in cities_filter) if cities_filter else None

    daily_by_slug: dict[str, list[list[Any]]] = {}
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
            "rows": rows
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

    # Generar climatology.json
    climatology = build_climatology(daily_by_slug)
    clim_file = out_dir / "climatology.json"
    with open(clim_file, "w", encoding="utf-8") as f:
        json.dump(climatology, f, separators=(",", ":"), ensure_ascii=False)

    # Generar index.json
    index_payload = {
        "generated_at": now_iso,
        "cities": index_cities
    }
    index_file = out_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, separators=(",", ":"), ensure_ascii=False)

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
