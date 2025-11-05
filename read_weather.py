import requests
import csv
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
# [OPCIONAL] Si quieres un modelo más potente, descomenta la siguiente línea
# from sklearn.ensemble import RandomForestRegressor 

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Error: No se encontró la variable de entorno OPENWEATHER_API_KEY.")

DIRECTORIO_DATOS = "datos" 
FICHERO_CIUDADES = "ciudades.txt"
FICHERO_PREDICCIONES = "predicciones.json"
FICHERO_MAPA = "city_locations.json"
# Mínimo de días con datos limpios para intentar entrenar un modelo
MIN_RECORDS_FOR_IA = 10 
# --- FIN DE LA CONFIGURACIÓN ---


# --- FUNCIONES DE RECOLECCIÓN (Sin cambios) ---

def obtener_datos_climaticos(ciudad_data, api_key):
    """
    Se conecta a la API. Usa lat/lon si está disponible,
    si no, usa el nombre (query).
    """
    nombre_query, lat, lon = ciudad_data # Desempaqueta la tupla

    if lat and lon:
        print(f"Obteniendo datos para: {nombre_query.split(',')[0]} (por lat/lon)...") # Mostramos nombre limpio
        URL = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es"
    else:
        print(f"Obteniendo datos para: {nombre_query.split(',')[0]} (por nombre)...") # Mostramos nombre limpio
        URL = f"https://api.openweathermap.org/data/2.5/weather?q={nombre_query}&appid={api_key}&units=metric&lang=es"
    
    try:
        respuesta = requests.get(URL)
        if respuesta.status_code == 200:
            print("¡Datos obtenidos con éxito!")
            return respuesta.json()
        elif respuesta.status_code == 404:
            print(f"Error 404: Ciudad '{nombre_query.split(',')[0]}' no encontrada en la API.")
            return None
        elif respuesta.status_code == 401:
            print("Error 401: API Key no válida o no activada.")
            return None
        else:
            print(f"Error en la API: {respuesta.status_code} para {nombre_query.split(',')[0]}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión para {nombre_query.split(',')[0]}: {e}")
        return None

def guardar_datos_ciudad(datos, nombre_ciudad_query): # Recibe "León,ES"
    """Extrae los datos relevantes y los guarda en el CSV específico de esa ciudad."""
    if not datos:
        print(f"No hay datos para guardar para {nombre_ciudad_query.split(',')[0]}.")
        return None

    try:
        nombre_ciudad_limpio = nombre_ciudad_query.split(',')[0]
        nombre_fichero = f"{DIRECTORIO_DATOS}/{nombre_ciudad_limpio.replace(' ', '_')}.csv"
        
        # 1. Extraemos los datos (sin cambios)
        timestamp = datetime.now().isoformat() 
        ciudad_api = datos['name']
        descripcion_cielo = datos['weather'][0]['description']
        temperatura = datos['main']['temp']
        sensacion_termica = datos['main']['feels_like']
        temp_min = datos['main']['temp_min']
        temp_max = datos['main']['temp_max']
        humedad = datos['main']['humidity']
        # [MEJORA] Convertir viento de m/s a km/h
        velocidad_viento_kmh = datos['wind']['speed'] * 3.6
        latitud = datos['coord']['lat']
        longitud = datos['coord']['lon']
        
        # 2. Preparamos la fila de datos
        fila_datos = [
            timestamp, ciudad_api, temperatura, sensacion_termica,
            temp_min, temp_max, humedad, velocidad_viento_kmh, descripcion_cielo,
            latitud, longitud
        ]
        
        # 3. Definimos la cabecera
        cabecera = [
            'fecha_hora', 'ciudad', 'temperatura_c', 'sensacion_c',
            'temp_min_c', 'temp_max_c', 'humedad_porc', 'viento_kmh', 'descripcion',
            'lat', 'lon'
        ]
        
        es_archivo_nuevo = not os.path.exists(nombre_fichero)
        
        with open(nombre_fichero, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if es_archivo_nuevo:
                writer.writerow(cabecera)
            writer.writerow(fila_datos)
            
        print(f"Datos guardados correctamente en {nombre_fichero}")
        
        return {
            "ciudad": nombre_ciudad_limpio,
            "lat": latitud,
            "lon": longitud,
            "temp": temperatura
        }
        
    except KeyError as e:
        print(f"Error: No se pudo encontrar la clave {e} en la respuesta JSON para {nombre_ciudad_limpio}.")
        return None
    except Exception as e:
        print(f"Error al guardar los datos para {nombre_ciudad_limpio}: {e}")
        return None

def leer_ciudades(fichero):
    """
    Lee el fichero de ciudades. Devuelve una lista de tuplas.
    Formato tupla: (nombre_query, lat, lon)
    Ej: ('Madrid,ES', None, None) o ('León,ES', '42.5984', '-5.5719')
    """
    if not os.path.exists(fichero):
        print(f"Error: No se encuentra el fichero {fichero}. Creando uno de ejemplo con 'Madrid'.")
        with open(fichero, 'w', encoding='utf-8') as f:
            f.write("Madrid,ES\n")
        return [("Madrid,ES", None, None)]
        
    with open(fichero, 'r', encoding='utf-8') as f:
        ciudades_data = []
        for linea in f:
            linea_limpia = linea.strip()
            if not linea_limpia or linea_limpia.startswith('#'): # Ignora comentarios
                continue
            
            partes = [p.strip() for p in linea_limpia.split(',')]
            
            if len(partes) == 4:
                nombre_query = f"{partes[0]},{partes[1]}"
                ciudades_data.append((nombre_query, partes[2], partes[3]))
            elif len(partes) == 2:
                nombre_query = f"{partes[0]},{partes[1]}"
                ciudades_data.append((nombre_query, None, None))
            
    print(f"Se han cargado {len(ciudades_data)} ciudades de {fichero}.")
    return ciudades_data

# --- [NUEVA] SECCIÓN DE PREDICCIÓN (IA) ---

def obtener_prediccion_owm(lat, lon, api_key):
    """
    Obtiene la predicción oficial de OWM para mañana (Max y Min).
    Usa la API OneCall 3.0, que REQUIERE lat/lon.
    """
    print("  Obteniendo predicción de OWM...")
    try:
        # Usamos la API One Call 3.0 que da 8 días de pronóstico
        forecast_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={api_key}&units=metric"
        
        r_forecast = requests.get(forecast_url)
        r_forecast.raise_for_status() # Lanza un error si la petición falla
        forecast_data = r_forecast.json()
        
        # 'daily[0]' es hoy, 'daily[1]' es mañana
        tomorrow_forecast = forecast_data['daily'][1]
        
        owm_max = tomorrow_forecast['temp']['max']
        owm_min = tomorrow_forecast['temp']['min']
        # OWM no da 'media', la calculamos
        owm_media = (owm_max + owm_min) / 2 
        
        print(f"  [OWM OK] Max: {owm_max}°C, Min: {owm_min}°C")
        return {
            "max": round(owm_max, 1),
            "min": round(owm_min, 1),
            "media": round(owm_media, 1)
        }

    except Exception as e:
        print(f"  [OWM Error] Fallo al obtener pronóstico: {e}")
        return {"max": None, "min": None, "media": None}

def generar_prediccion_ia_ciudad(csv_path):
    """
    Carga el historial de una ciudad, entrena 3 modelos (max, min, media)
    y predice los valores de mañana.
    """
    print("  Generando predicción de IA...")
    
    # Valores por defecto si falla
    default_return = {"max": None, "min": None, "media": None, "registros": 0}

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  [IA Error] No se encontró {csv_path}")
        return default_return

    # Tienes múltiples lecturas al día. 'temp_max_c' y 'temp_min_c' son del *momento*.
    # Necesitamos los agregados del *día*.
    try:
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        
        # Agrupar por día (Resample). Usamos 'temperatura_c' para los agregados diarios reales.
        df_daily = df.resample('D', on='fecha_hora').agg(
            temp_max=('temperatura_c', 'max'),
            temp_min=('temperatura_c', 'min'),
            temp_media=('temperatura_c', 'mean'),
            hum_media=('humedad_porc', 'mean'),
            viento_media=('viento_kmh', 'mean')
        )
        
        # Rellenar días sin datos (ej. fines de semana si el script no corrió)
        df_daily = df_daily.ffill() 

        # --- Crear Features (X) y Objetivos (y) ---
        
        # Features (X): Usamos los datos de "ayer" (lag=1) para predecir "hoy"
        df_daily['dia_del_anio'] = df_daily.index.dayofyear
        df_daily['temp_max_lag1'] = df_daily['temp_max'].shift(1)
        df_daily['temp_min_lag1'] = df_daily['temp_min'].shift(1)
        df_daily['temp_media_lag1'] = df_daily['temp_media'].shift(1)
        df_daily['hum_media_lag1'] = df_daily['hum_media'].shift(1)
        df_daily['viento_media_lag1'] = df_daily['viento_media'].shift(1)
        
        # Limpiar NaNs creados por .shift() y .agg()
        df_clean = df_daily.dropna()
        
        registros_limpios = len(df_clean)
        
        if registros_limpios < MIN_RECORDS_FOR_IA:
            print(f"  [IA Info] Historial insuficiente ({registros_limpios} días limpios). Se requieren {MIN_RECORDS_FOR_IA}.")
            return {"max": None, "min": None, "media": None, "registros": registros_limpios}

        # Definir X (features) e y (targets)
        features = ['temp_max_lag1', 'temp_min_lag1', 'temp_media_lag1', 
                    'hum_media_lag1', 'viento_media_lag1', 'dia_del_anio']
        X_train = df_clean[features]

        # --- Entrenar 3 Modelos Separados ---
        
        # Modelo 1: Predicción de Temp Máxima
        y_max = df_clean['temp_max']
        model_max = LinearRegression()
        # model_max = RandomForestRegressor(n_estimators=50) # Opcional: Mejora
        model_max.fit(X_train, y_max)

        # Modelo 2: Predicción de Temp Mínima
        y_min = df_clean['temp_min']
        model_min = LinearRegression()
        model_min.fit(X_train, y_min)
        
        # Modelo 3: Predicción de Temp Media
        y_media = df_clean['temp_media']
        model_media = LinearRegression()
        model_media.fit(X_train, y_media)

        # --- Preparar datos para predecir "Mañana" ---
        
        # Necesitamos los datos de "Hoy" (última fila de df_daily) para predecir "Mañana"
        # Usamos df_daily.ffill() por si el script corre a las 00:01 y "hoy" aún no tiene datos
        today_features_row = df_daily.ffill().iloc[-1]
        
        # Calcular el 'dia_del_anio' de mañana
        dia_manana = (today_features_row.name + timedelta(days=1)).dayofyear

        X_to_predict = [[
            today_features_row['temp_max'],
            today_features_row['temp_min'],
            today_features_row['temp_media'],
            today_features_row['hum_media'],
            today_features_row['viento_media'],
            dia_manana
        ]]

        # --- Realizar las 3 Predicciones ---
        pred_max = model_max.predict(X_to_predict)[0]
        pred_min = model_min.predict(X_to_predict)[0]
        pred_media = model_media.predict(X_to_predict)[0]

        print(f"  [IA OK] Registros: {registros_limpios}. Pred Max: {pred_max:.1f}°C")
        return {
            "max": round(pred_max, 1), 
            "min": round(pred_min, 1), 
            "media": round(pred_media, 1),
            "registros": registros_limpios
        }

    except Exception as e:
        print(f"  [IA Error] Fallo al procesar IA: {e}")
        return {"max": None, "min": None, "media": None, "registros": 0}


# --- [REESTRUCTURADO] Ejecución Principal del Script ---
if __name__ == "__main__":
    
    os.makedirs(DIRECTORIO_DATOS, exist_ok=True)
    ciudades_a_procesar = leer_ciudades(FICHERO_CIUDADES) # devuelve lista de tuplas
    print(f"\n--- Iniciando proceso para {len(ciudades_a_procesar)} ciudades ---")
    
    datos_para_mapa = []
    all_predictions_data = {} # Estructura para el nuevo JSON de predicciones

    # Iteramos sobre la lista de tuplas
    for ciudad_data in ciudades_a_procesar: 
        
        # Desempaquetamos los datos de la ciudad
        nombre_ciudad_query, lat, lon = ciudad_data
        nombre_ciudad_limpio = nombre_ciudad_query.split(',')[0]
        
        print(f"\nProcesando: {nombre_ciudad_limpio}...")

        # Inicializamos la estructura de predicción para esta ciudad
        all_predictions_data[nombre_ciudad_limpio] = {
            "pred_owm": {"max": None, "min": None, "media": None},
            "pred_ia": {"max": None, "min": None, "media": None, "registros": 0}
        }
        
        # --- Parte 1: Recolección de Datos (Tu código) ---
        datos_actuales = obtener_datos_climaticos(ciudad_data, API_KEY) 
        
        if datos_actuales:
            resumen_ciudad = guardar_datos_ciudad(datos_actuales, nombre_ciudad_query)
            if resumen_ciudad:
                datos_para_mapa.append(resumen_ciudad)
        else:
            print(f"No se pudieron obtener datos actuales para {nombre_ciudad_limpio}.")
            
        print("Pausando 1.1 seg (API datos actuales)...")
        time.sleep(1.1) 
        
        
        # --- Parte 2: Generación de Predicción OWM (Nueva) ---
        if lat and lon:
            # Esta función ya imprime sus propios logs
            owm_preds = obtener_prediccion_owm(lat, lon, API_KEY)
            all_predictions_data[nombre_ciudad_limpio]["pred_owm"] = owm_preds
            
            print("Pausando 1.1 seg (API pronóstico)...")
            time.sleep(1.1) # Pausa para la segunda API (OneCall)
        else:
            print(f"  No hay lat/lon para {nombre_ciudad_limpio}. Saltando predicción OWM.")
            
            
        # --- Parte 3: Generación de Predicción IA (Nueva) ---
        csv_path = f"{DIRECTORIO_DATOS}/{nombre_ciudad_limpio.replace(' ', '_')}.csv"
        # Esta función ya imprime sus propios logs
        ia_preds = generar_prediccion_ia_ciudad(csv_path)
        all_predictions_data[nombre_ciudad_limpio]["pred_ia"] = ia_preds

            
    print("\n--- Proceso de registro y predicción completado ---")
    
    # --- Parte 4: Guardar Ficheros JSON ---
    
    # Guardamos el fichero JSON para el mapa (Tu código)
    try:
        with open(FICHERO_MAPA, 'w', encoding='utf-8') as f:
            json.dump(datos_para_mapa, f, indent=2, ensure_ascii=False)
        print(f"Resumen de mapa guardado con éxito en {FICHERO_MAPA}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON del mapa: {e}")

    # Guardamos el fichero JSON para las predicciones (Nuevo)
    try:
        with open(FICHERO_PREDICCIONES, 'w', encoding='utf-8') as f:
            json.dump(all_predictions_data, f, indent=2, ensure_ascii=False)
        print(f"Predicciones guardadas con éxito en {FICHERO_PREDICCIONES}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON de predicciones: {e}")

    print("\nScript finalizado.")