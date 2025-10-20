import requests
import csv         # Importamos la librería para manejar archivos CSV
import os          # Para comprobar si el archivo ya existe
from datetime import datetime # Para obtener la fecha y hora actual

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Reemplaza esto con tu clave de API real
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("Error: No se encontró la variable de entorno OPENWEATHER_API_KEY.")
# Puedes añadir más ciudades, pero empecemos con una
CIUDAD = "Madrid" 
NOMBRE_ARCHIVO = "weather_data.csv"
# --- FIN DE LA CONFIGURACIÓN ---

def obtener_datos_climaticos(ciudad, api_key):
    """Se conecta a la API y devuelve los datos de la ciudad."""
    print(f"Obteniendo datos para: {ciudad}...")
    URL = f"https://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={api_key}&units=metric&lang=es"
    
    try:
        respuesta = requests.get(URL)
        if respuesta.status_code == 200:
            print("¡Datos obtenidos con éxito!")
            return respuesta.json() # Devuelve el diccionario de datos
        elif respuesta.status_code == 401:
            print("Error 401: API Key no válida o no activada.")
            return None
        else:
            print(f"Error en la API: {respuesta.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None

def procesar_y_guardar_datos(datos, nombre_archivo):
    """Extrae los datos relevantes y los guarda en el CSV."""
    if not datos:
        print("No hay datos para guardar.")
        return

    try:
        # 1. Extraemos los datos que nos interesan del JSON
        # Usamos .isoformat() para un formato de fecha estándar
        timestamp = datetime.now().isoformat() 
        ciudad_nombre = datos['name']
        descripcion_cielo = datos['weather'][0]['description']
        temperatura = datos['main']['temp']
        sensacion_termica = datos['main']['feels_like']
        temp_min = datos['main']['temp_min']
        temp_max = datos['main']['temp_max']
        humedad = datos['main']['humidity']
        velocidad_viento = datos['wind']['speed']
        
        # 2. Preparamos la fila de datos
        # El orden debe coincidir con la cabecera
        fila_datos = [
            timestamp, ciudad_nombre, temperatura, sensacion_termica,
            temp_min, temp_max, humedad, velocidad_viento, descripcion_cielo
        ]
        
        # 3. Definimos la cabecera
        cabecera = [
            'fecha_hora', 'ciudad', 'temperatura_c', 'sensacion_c',
            'temp_min_c', 'temp_max_c', 'humedad_porc', 'viento_kmh', 'descripcion'
        ]
        
        # 4. Comprobamos si el archivo es nuevo para escribir la cabecera
        es_archivo_nuevo = not os.path.exists(nombre_archivo)
        
        # 5. Abrimos el archivo en modo 'a' (append/añadir)
        # newline='' es importante para que la librería csv funcione bien en Windows
        with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if es_archivo_nuevo:
                writer.writerow(cabecera) # Escribimos la cabecera solo si es nuevo
            
            writer.writerow(fila_datos) # Escribimos la fila de datos
            
        print(f"Datos guardados correctamente en {nombre_archivo}")
        
    except KeyError as e:
        print(f"Error: No se pudo encontrar la clave {e} en la respuesta JSON.")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")

# --- Ejecución Principal del Script ---
if __name__ == "__main__":
    datos_actuales = obtener_datos_climaticos(CIUDAD, API_KEY)
    
    # Solo intentamos guardar si obtuvimos datos
    if datos_actuales:
        procesar_y_guardar_datos(datos_actuales, NOMBRE_ARCHIVO)