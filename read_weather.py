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

# --- CONFIGURACIÓN (Sin cambios) ---
API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Error: No se encontró la variable de entorno OPENWEATHER_API_KEY.")

DIRECTORIO_DATOS = "datos" 
FICHERO_CIUDADES = "ciudades.txt"
FICHERO_PREDICCIONES = "predicciones.json"
CIUDADES_A_PREDECIR = ["Madrid", "León", "London", "New York"] 

# [CAMBIO] Nuevo fichero de resumen para el mapa
FICHERO_MAPA = "city_locations.json"
# --- FIN DE LA CONFIGURACIÓN ---


# --- FUNCIONES DE RECOLECCIÓN ---

def obtener_datos_climaticos(ciudad, api_key):
    # ... (Esta función es idéntica, no la cambies) ...
    print(f"Obteniendo datos para: {ciudad}...")
    URL = f"https://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={api_key}&units=metric&lang=es"
    try:
        respuesta = requests.get(URL)
        if respuesta.status_code == 200:
            print("¡Datos obtenidos con éxito!")
            return respuesta.json()
        # ... (resto de la función idéntica) ...
        elif respuesta.status_code == 404:
            print(f"Error 404: Ciudad '{ciudad}' no encontrada en la API.")
            return None
        elif respuesta.status_code == 401:
            print("Error 401: API Key no válida o no activada.")
            return None
        else:
            print(f"Error en la API: {respuesta.status_code} para {ciudad}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión para {ciudad}: {e}")
        return None


# [CAMBIO] Modificamos la función de guardado
def guardar_datos_ciudad(datos, nombre_ciudad):
    """Extrae los datos relevantes y los guarda en el CSV específico de esa ciudad."""
    if not datos:
        print(f"No hay datos para guardar para {nombre_ciudad}.")
        return None # [CAMBIO] Devolvemos None si no hay datos

    try:
        nombre_fichero = f"{DIRECTORIO_DATOS}/{nombre_ciudad.replace(' ', '_')}.csv"
        
        # 1. Extraemos los datos que nos interesan
        timestamp = datetime.now().isoformat() 
        ciudad_api = datos['name']
        descripcion_cielo = datos['weather'][0]['description']
        temperatura = datos['main']['temp']
        sensacion_termica = datos['main']['feels_like']
        temp_min = datos['main']['temp_min']
        temp_max = datos['main']['temp_max']
        humedad = datos['main']['humidity']
        velocidad_viento = datos['wind']['speed']
        # [CAMBIO] Extraemos latitud y longitud
        latitud = datos['coord']['lat']
        longitud = datos['coord']['lon']
        
        # 2. Preparamos la fila de datos
        fila_datos = [
            timestamp, ciudad_api, temperatura, sensacion_termica,
            temp_min, temp_max, humedad, velocidad_viento, descripcion_cielo,
            latitud, longitud  # [CAMBIO] Añadimos los nuevos datos
        ]
        
        # 3. Definimos la cabecera
        cabecera = [
            'fecha_hora', 'ciudad', 'temperatura_c', 'sensacion_c',
            'temp_min_c', 'temp_max_c', 'humedad_porc', 'viento_kmh', 'descripcion',
            'lat', 'lon'  # [CAMBIO] Añadimos las nuevas cabeceras
        ]
        
        es_archivo_nuevo = not os.path.exists(nombre_fichero)
        
        with open(nombre_fichero, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if es_archivo_nuevo:
                writer.writerow(cabecera)
            writer.writerow(fila_datos)
            
        print(f"Datos guardados correctamente en {nombre_fichero}")
        
        # [CAMBIO] Devolvemos un resumen para el mapa
        return {
            "ciudad": nombre_ciudad,
            "lat": latitud,
            "lon": longitud,
            "temp": temperatura
        }
        
    except KeyError as e:
        print(f"Error: No se pudo encontrar la clave {e} en la respuesta JSON para {nombre_ciudad}.")
        return None
    except Exception as e:
        print(f"Error al guardar los datos para {nombre_ciudad}: {e}")
        return None

def leer_ciudades(fichero):
    # ... (Esta función es idéntica, no la cambies) ...
    if not os.path.exists(fichero):
        print(f"Error: No se encuentra el fichero {fichero}. Creando uno de ejemplo con 'Madrid'.")
        with open(fichero, 'w', encoding='utf-8') as f:
            f.write("Madrid\n")
        return ["Madrid"]
        
    with open(fichero, 'r', encoding='utf-8') as f:
        ciudades = [linea.strip() for linea in f if linea.strip()] 
    print(f"Se han cargado {len(ciudades)} ciudades de {fichero}.")
    return ciudades

# --- SECCIÓN DE PREDICCIÓN (IA) (Sin cambios) ---

def preparar_datos_para_ia(df):
    # ... (Esta función es idéntica, no la cambies) ...
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

def entrenar_modelo_ia(X, y):
    # ... (Esta función es idéntica, no la cambies) ...
    print("Entrenando modelo de IA...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)
    print(f"Modelo entrenado. Error (MSE): {error:.2f}")
    return model

def generar_predicciones():
    # ... (Esta función es idéntica, no la cambies) ...
    print("\n--- Iniciando Módulo de Predicción de IA ---")
    resultados_predicciones = {}
    for ciudad in CIUDADES_A_PREDECIR:
        print(f"\nProcesando predicción para: {ciudad}")
        nombre_fichero = f"{DIRECTORIO_DATOS}/{ciudad.replace(' ', '_')}.csv"
        if not os.path.exists(nombre_fichero):
            print(f"No se encontró historial para {ciudad}. Saltando.")
            continue
        try:
            historial_df = pd.read_csv(nombre_fichero)
            if len(historial_df) < 10:
                print(f"Historial insuficiente para {ciudad} (necesarios 10, hay {len(historial_df)}). Saltando.")
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


# --- [CAMBIO] Ejecución Principal del Script ---
if __name__ == "__main__":
    
    # --- Parte 1: Recolección de Datos ---
    os.makedirs(DIRECTORIO_DATOS, exist_ok=True)
    ciudades_a_procesar = leer_ciudades(FICHERO_CIUDADES)
    print(f"Iniciando proceso de registro para {len(ciudades_a_procesar)} ciudades.")
    
    # [CAMBIO] Lista para guardar el resumen del mapa
    datos_para_mapa = []

    for ciudad in ciudades_a_procesar:
        datos_actuales = obtener_datos_climaticos(ciudad, API_KEY)
        
        if datos_actuales:
            # [CAMBIO] Guardamos el resumen que devuelve la función
            resumen_ciudad = guardar_datos_ciudad(datos_actuales, ciudad)
            if resumen_ciudad:
                datos_para_mapa.append(resumen_ciudad)
        else:
            print(f"No se pudieron obtener datos para {ciudad}. Saltando.")
            
        print("Pausando 1.1 segundos para respetar el límite de la API...")
        time.sleep(1.1) 
            
    print("Proceso de registro completado.")
    
    # [CAMBIO] Guardamos el nuevo fichero JSON para el mapa
    try:
        with open(FICHERO_MAPA, 'w', encoding='utf-8') as f:
            json.dump(datos_para_mapa, f)
        print(f"Resumen de mapa guardado con éxito en {FICHERO_MAPA}")
    except Exception as e:
        print(f"Error al guardar el fichero JSON del mapa: {e}")

    # --- Parte 2: Generación de Predicciones ---
    generar_predicciones()
    
    print("Script finalizado.")