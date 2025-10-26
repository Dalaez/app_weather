import requests
import csv
import os
import time
import json                  
import pandas as pd          
from datetime import datetime
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Error: No se encontró la variable de entorno OPENWEATHER_API_KEY.")

DIRECTORIO_DATOS = "datos" 
FICHERO_CIUDADES = "ciudades.txt"
FICHERO_PREDICCIONES = "predicciones.json"
# [CORRECCIÓN] CIUDADES_A_PREDECIR ahora debe usar el nombre limpio ('León' no 'León,ES')
# porque así se llamará el fichero CSV
CIUDADES_A_PREDECIR = ["Madrid", "León", "London", "New York"] 
FICHERO_MAPA = "city_locations.json"
# --- FIN DE LA CONFIGURACIÓN ---


# --- FUNCIONES DE RECOLECCIÓN ---

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


# [CORRECCIÓN] Modificamos la función de guardado para usar el nombre limpio
def guardar_datos_ciudad(datos, nombre_ciudad_query): # Recibe "León,ES"
    """Extrae los datos relevantes y los guarda en el CSV específico de esa ciudad."""
    if not datos:
        print(f"No hay datos para guardar para {nombre_ciudad_query.split(',')[0]}.")
        return None

    try:
        # [CORRECCIÓN] Limpiamos el nombre ANTES de usarlo
        nombre_ciudad_limpio = nombre_ciudad_query.split(',')[0]

        # [CORRECCIÓN] Usamos el nombre limpio para el fichero
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
        velocidad_viento = datos['wind']['speed']
        latitud = datos['coord']['lat']
        longitud = datos['coord']['lon']
        
        # 2. Preparamos la fila de datos (sin cambios)
        fila_datos = [
            timestamp, ciudad_api, temperatura, sensacion_termica,
            temp_min, temp_max, humedad, velocidad_viento, descripcion_cielo,
            latitud, longitud
        ]
        
        # 3. Definimos la cabecera (sin cambios)
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
        
        # [CORRECCIÓN] Devolvemos el nombre limpio para el mapa
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

# --- SECCIÓN DE PREDICCIÓN (IA) (Sin cambios) ---
def preparar_datos_para_ia(df): # ... (idéntico)
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df = df.sort_values(by='fecha_hora')
    df['temp_max_manana'] = df['temp_max_c'].shift(-1)
    df = df.dropna()
    features = ['temp_max_c', 'temp_min_c', 'humedad_porc', 'viento_kmh']
    target = 'temp_max_manana'
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=features)
    X = df[features]
    y = df[target]
    ultima_fila_conocida = df[features].iloc[-1:]
    return X, y, ultima_fila_conocida

def entrenar_modelo_ia(X, y): # ... (idéntico)
    print("Entrenando modelo de IA...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    print(f"Modelo entrenado. Error (MSE): {error:.2f}")
    return model

def generar_predicciones(): # ... (idéntico)
    print("\n--- Iniciando Módulo de Predicción de IA ---")
    resultados_predicciones = {}
    for ciudad in CIUDADES_A_PREDECIR: # Usa los nombres limpios: 'Madrid', 'León'...
        print(f"\nProcesando predicción para: {ciudad}")
        # Busca el fichero con el nombre limpio: datos/León.csv
        nombre_fichero = f"{DIRECTORIO_DATOS}/{ciudad.replace(' ', '_')}.csv" 
        if not os.path.exists(nombre_fichero):
            print(f"No se encontró historial para {ciudad}. Saltando.")
            continue
        try:
            historial_df = pd.read_csv(nombre_fichero)
            if len(historial_df) < 10:
                print(f"Historial insuficiente para {ciudad}. Saltando.")
                continue
            X, y, ultima_fila_conocida = preparar_datos_para_ia(historial_df)
            if X.empty:
                print(f"No hay datos limpios suficientes para entrenar en {ciudad}.")
                continue
            modelo = entrenar_modelo_ia(X, y)
            prediccion_para_manana = modelo.predict(ultima_fila_conocida)
            temp_predicha = prediccion_para_manana[0]
            print(f"Predicción de Tº Max para mañana en {ciudad}: {temp_predicha:.2f}°C")
            resultados_predicciones[ciudad] = {
                'prediccion_temp_max': round(temp_predicha, 2),
                'fecha_prediccion': datetime.now().isoformat(),
                'historial_usado': len(historial_df)
            }
        except Exception as e:
            print(f"Error al procesar la predicción para {ciudad}: {e}")
            
    try:
        with open(FICHERO_PREDICCIONES, 'w', encoding='utf-8') as f:
            json.dump(resultados_predicciones, f, indent=4)
        print(f"\nPredicciones guardadas con éxito en {FICHERO_PREDICCIONES}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON de predicciones: {e}")


# --- [CORRECCIÓN] Ejecución Principal del Script ---
if __name__ == "__main__":
    
    # --- Parte 1: Recolección de Datos ---
    os.makedirs(DIRECTORIO_DATOS, exist_ok=True)
    ciudades_a_procesar = leer_ciudades(FICHERO_CIUDADES) # devuelve lista de tuplas
    print(f"Iniciando proceso de registro para {len(ciudades_a_procesar)} ciudades.")
    
    datos_para_mapa = []

    # Iteramos sobre la lista de tuplas
    for ciudad_data in ciudades_a_procesar: 
        # Pasamos la tupla completa a obtener_datos_climaticos
        datos_actuales = obtener_datos_climaticos(ciudad_data, API_KEY) 
        
        if datos_actuales:
            # [CORRECCIÓN] Pasamos SOLO el nombre_query ('León,ES') a guardar_datos_ciudad
            nombre_ciudad_query = ciudad_data[0] 
            resumen_ciudad = guardar_datos_ciudad(datos_actuales, nombre_ciudad_query)
            if resumen_ciudad:
                datos_para_mapa.append(resumen_ciudad)
        else:
            # Mostramos el nombre limpio en el error
            print(f"No se pudieron obtener datos para {ciudad_data[0].split(',')[0]}. Saltando.") 
            
        print("Pausando 1.1 segundos para respetar el límite de la API...")
        time.sleep(1.1) 
            
    print("Proceso de registro completado.")
    
    # Guardamos el fichero JSON para el mapa (sin cambios)
    try:
        with open(FICHERO_MAPA, 'w', encoding='utf-8') as f:
            json.dump(datos_para_mapa, f)
        print(f"Resumen de mapa guardado con éxito en {FICHERO_MAPA}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON del mapa: {e}")

    # --- Parte 2: Generación de Predicciones ---
    # La función generar_predicciones ya usa los nombres limpios, no necesita cambios
    generar_predicciones() 
    
    print("Script finalizado.")