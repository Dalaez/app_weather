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

# --- MODIFICADO: Función con Lógica de Migración ---

def guardar_datos_ciudad(datos, nombre_ciudad_query): # Recibe "León,ES"
    """Extrae los datos relevantes y los guarda en el CSV específico de esa ciudad."""
    if not datos:
        print(f"No hay datos para guardar para {nombre_ciudad_query.split(',')[0]}.")
        return None

    try:
        nombre_ciudad_limpio = nombre_ciudad_query.split(',')[0]
        nombre_fichero = f"{DIRECTORIO_DATOS}/{nombre_ciudad_limpio.replace(' ', '_')}.csv"
        
        # 1. Extraemos los datos
        timestamp = datetime.now().isoformat() 
        ciudad_api = datos['name']
        descripcion_cielo = datos['weather'][0]['description']
        temperatura = datos['main']['temp']
        sensacion_termica = datos['main']['feels_like']
        temp_min = datos['main']['temp_min']
        temp_max = datos['main']['temp_max']
        humedad = datos['main']['humidity']
        velocidad_viento_kmh = datos['wind']['speed'] * 3.6
        latitud = datos['coord']['lat']
        longitud = datos['coord']['lon']
        
        # --- NUEVOS CAMPOS ---
        nubosidad = datos['clouds']['all'] 
        visibilidad = datos['visibility'] 
        lluvia_1h = datos.get('rain', {}).get('1h', 0.0)
        nieve_1h = datos.get('snow', {}).get('1h', 0.0)
        # --- FIN DE NUEVOS CAMPOS ---
        
        # 2. Definimos la cabecera NUEVA (15 columnas)
        cabecera_nueva = [
            'fecha_hora', 'ciudad', 'temperatura_c', 'sensacion_c',
            'temp_min_c', 'temp_max_c', 'humedad_porc', 'viento_kmh', 'descripcion',
            'lat', 'lon',
            'nubosidad_porc', 'visibilidad_m', 'lluvia_1h', 'nieve_1h'
        ]

        # 3. Preparamos la fila de datos NUEVA (15 columnas)
        fila_datos = [
            timestamp, ciudad_api, temperatura, sensacion_termica,
            temp_min, temp_max, humedad, velocidad_viento_kmh, descripcion_cielo,
            latitud, longitud,
            nubosidad, visibilidad, lluvia_1h, nieve_1h
        ]
        
        es_archivo_nuevo = not os.path.exists(nombre_fichero)
        
        # --- NUEVA LÓGICA DE MIGRACIÓN ---
        if not es_archivo_nuevo:
            try:
                # Leemos la cabecera del fichero existente para comprobar su longitud
                with open(nombre_fichero, 'r', encoding='utf-8') as f:
                    cabecera_existente = f.readline().strip().split(',')
                
                if len(cabecera_existente) != len(cabecera_nueva):
                    print(f"  [MIGRACIÓN] Detectado schema antiguo en {nombre_fichero}. Migrando a {len(cabecera_nueva)} columnas SIN PERDER DATOS...")
                    # El fichero es antiguo. Lo leemos con Pandas
                    df_antiguo = pd.read_csv(nombre_fichero)
                    
                    # Añadimos las columnas que faltan con valores por defecto
                    if 'nubosidad_porc' not in df_antiguo.columns:
                        df_antiguo['nubosidad_porc'] = 0
                    if 'visibilidad_m' not in df_antiguo.columns:
                        df_antiguo['visibilidad_m'] = 10000
                    if 'lluvia_1h' not in df_antiguo.columns:
                        df_antiguo['lluvia_1h'] = 0.0
                    if 'nieve_1h' not in df_antiguo.columns:
                        df_antiguo['nieve_1h'] = 0.0
                    
                    # Reordenamos y guardamos (SOBREESCRIBIMOS)
                    df_antiguo = df_antiguo[cabecera_nueva] # Asegura el orden correcto
                    df_antiguo.to_csv(nombre_fichero, index=False, header=True, encoding='utf-8')
                    print(f"  [MIGRACIÓN] ¡Fichero {nombre_fichero} migrado con éxito!")
                    
                    # El fichero ya está migrado y tiene cabecera.
                    es_archivo_nuevo = False # No es nuevo, y ya tiene la cabecera correcta
                    
            except Exception as e:
                print(f"  [MIGRACIÓN Error] Fallo al migrar {nombre_fichero}: {e}. Se borrará y recreará para evitar corrupción.")
                os.remove(nombre_fichero)
                es_archivo_nuevo = True
        # --- FIN DE LÓGICA DE MIGRACIÓN ---

        # 4. Escribimos la fila
        with open(nombre_fichero, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if es_archivo_nuevo:
                writer.writerow(cabecera_nueva) # Escribimos la nueva cabecera
            writer.writerow(fila_datos)
            
        print(f"Datos guardados (incluyendo nubes/lluvia) en {nombre_fichero}")
        
        # El resumen para el mapa no cambia
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
    Obtiene el pronóstico de 5 días de OWM.
    [CAMBIO] Devuelve una lista con los agregados de CADA día.
    """
    print("  Obteniendo pronóstico de 5 días de OWM (API 2.5/forecast)...")
    try:
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        r_forecast = requests.get(forecast_url)
        r_forecast.raise_for_status()
        forecast_data = r_forecast.json()

        # --- Nueva Lógica de Agregación ---
        daily_temps = {} # Clave: "YYYY-MM-DD", Valor: [lista de temps]
        today_date_str = datetime.now().strftime('%Y-%m-%d')

        for item in forecast_data['list']:
            date_str = item['dt_txt'].split(' ')[0]
            
            # Ignoramos los registros de "hoy"
            if date_str == today_date_str:
                continue
                
            if date_str not in daily_temps:
                daily_temps[date_str] = []
                
            daily_temps[date_str].append(item['main']['temp'])
        
        if not daily_temps:
            print("  [OWM Info] La API 2.5 no devolvió datos para días futuros.")
            return [] # Devolver lista vacía

        # --- Procesar los días agregados ---
        forecast_list = []
        for date_str, temps in daily_temps.items():
            owm_max = max(temps)
            owm_min = min(temps)
            owm_media = sum(temps) / len(temps)
            
            forecast_list.append({
                "date": date_str, # Importante para el gráfico
                "max": round(owm_max, 1),
                "min": round(owm_min, 1),
                "media": round(owm_media, 1)
            })
        
        # Ordenar por fecha y devolver la lista
        forecast_list.sort(key=lambda x: x['date'])
        print(f"  [OWM OK] Pronóstico de {len(forecast_list)} días calculado.")
        return forecast_list

    except Exception as e:
        print(f"  [OWM Error] Fallo al obtener pronóstico (API 2.5): {e}")
        return [] # Devolver lista vacía

def generar_prediccion_ia_ciudad(csv_path):
    """
    Carga el historial de una ciudad, entrena 3 modelos (max, min, media)
    y predice los valores de mañana.
    """
    print("  Generando predicción de IA...")
    
    # Valores por defecto si falla
    default_return = {"max": None, "min": None, "media": None, "registros": 0}

    try:
        # --- CORRECCIÓN: Ahora la lectura es segura gracias a la migración ---
        df = pd.read_csv(csv_path)
    
    except pd.errors.ParserError as e:
        # Este error solo debería ocurrir si el script falla a mitad de la migración
        print(f"  [IA Error] Fichero CSV AÚN corrupto: {e}. Omitiendo por ahora.")
        return default_return

    except FileNotFoundError:
        print(f"  [IA Error] No se encontró {csv_path}")
        return default_return

    try:
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        
        # Rellenar NaNs por si acaso (ej. datos antiguos migrados)
        df['humedad_porc'] = df['humedad_porc'].fillna(60)
        df['viento_kmh'] = df['viento_kmh'].fillna(0)
        
        # Convertir a numérico por si acaso
        df[['temperatura_c', 'humedad_porc', 'viento_kmh']] = df[['temperatura_c', 'humedad_porc', 'viento_kmh']].apply(pd.to_numeric, errors='coerce')
        
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
        
        y_max = df_clean['temp_max']
        model_max = LinearRegression()
        model_max.fit(X_train, y_max)

        y_min = df_clean['temp_min']
        model_min = LinearRegression()
        model_min.fit(X_train, y_min)
        
        y_media = df_clean['temp_media']
        model_media = LinearRegression()
        model_media.fit(X_train, y_media)

        # --- Preparar datos para predecir "Mañana" ---
        
        today_features_row = df_daily.ffill().iloc[-1]
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
        print(f"  [IA Error] Fallo al procesar IA (post-lectura): {e}")
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
            "pred_owm_5day": [],
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
            owm_preds_lista = obtener_prediccion_owm(lat, lon, API_KEY)
            all_predictions_data[nombre_ciudad_limpio]["pred_owm_5day"] = owm_preds_lista
            
            print("Pausando 1.1 seg (API pronóstico)...")
            time.sleep(1.1)
        else:
            print(f"  No hay lat/lon para {nombre_ciudad_limpio}. Saltando predicción OWM.")
            
            
        # --- Parte 3: Generación de Predicción IA (Nueva) ---
        csv_path = f"{DIRECTORIO_DATOS}/{nombre_ciudad_limpio.replace(' ', '_')}.csv"
        ia_preds = generar_prediccion_ia_ciudad(csv_path)
        all_predictions_data[nombre_ciudad_limpio]["pred_ia"] = ia_preds

            
    print("\n--- Proceso de registro y predicción completado ---")
    
    # --- Parte 4: Guardar Ficheros JSON ---
    
    try:
        with open(FICHERO_MAPA, 'w', encoding='utf-8') as f:
            json.dump(datos_para_mapa, f, indent=2, ensure_ascii=False)
        print(f"Resumen de mapa guardado con éxito en {FICHERO_MAPA}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON del mapa: {e}")

    try:
        with open(FICHERO_PREDICCIONES, 'w', encoding='utf-8') as f:
            json.dump(all_predictions_data, f, indent=2, ensure_ascii=False)
        print(f"Predicciones guardadas con éxito en {FICHERO_PREDICCIONES}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON de predicciones: {e}")

    print("\nScript finalizado.")