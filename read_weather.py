import requests
import csv
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
# La API Key se lee desde las variables de entorno de GitHub
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("Error: No se encontró la variable de entorno OPENWEATHER_API_KEY.")

# Directorio donde guardaremos todos los CSV
DIRECTORIO_DATOS = "datos" 
# Fichero que contiene nuestra lista de ciudades
FICHERO_CIUDADES = "ciudades.txt" 
# --- FIN DE LA CONFIGURACIÓN ---


def obtener_datos_climaticos(ciudad, api_key):
    """Se conecta a la API y devuelve los datos de la ciudad."""
    print(f"Obteniendo datos para: {ciudad}...")
    URL = f"https://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={api_key}&units=metric&lang=es"
    
    try:
        respuesta = requests.get(URL)
        if respuesta.status_code == 200:
            print("¡Datos obtenidos con éxito!")
            return respuesta.json()
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

def guardar_datos_ciudad(datos, nombre_ciudad):
    """Extrae los datos relevantes y los guarda en el CSV específico de esa ciudad."""
    if not datos:
        print(f"No hay datos para guardar para {nombre_ciudad}.")
        return

    try:
        # 1. Definimos la ruta del fichero
        # Limpiamos el nombre de la ciudad para que sea un nombre de fichero seguro (aunque los de la lista ya lo son)
        nombre_fichero = f"{DIRECTORIO_DATOS}/{nombre_ciudad.replace(' ', '_')}.csv"
        
        # 2. Extraemos los datos que nos interesan
        timestamp = datetime.now().isoformat() 
        ciudad_api = datos['name'] # El nombre canónico que devuelve la API
        descripcion_cielo = datos['weather'][0]['description']
        temperatura = datos['main']['temp']
        sensacion_termica = datos['main']['feels_like']
        temp_min = datos['main']['temp_min']
        temp_max = datos['main']['temp_max']
        humedad = datos['main']['humidity']
        velocidad_viento = datos['wind']['speed']
        
        # 3. Preparamos la fila de datos
        fila_datos = [
            timestamp, ciudad_api, temperatura, sensacion_termica,
            temp_min, temp_max, humedad, velocidad_viento, descripcion_cielo
        ]
        
        # 4. Definimos la cabecera (será la misma para todos los ficheros)
        cabecera = [
            'fecha_hora', 'ciudad', 'temperatura_c', 'sensacion_c',
            'temp_min_c', 'temp_max_c', 'humedad_porc', 'viento_kmh', 'descripcion'
        ]
        
        # 5. Comprobamos si el archivo es nuevo para escribir la cabecera
        es_archivo_nuevo = not os.path.exists(nombre_fichero)
        
        # 6. Abrimos el archivo en modo 'a' (append/añadir)
        with open(nombre_fichero, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if es_archivo_nuevo:
                writer.writerow(cabecera) # Escribimos la cabecera solo si es nuevo
            
            writer.writerow(fila_datos) # Escribimos la fila de datos
            
        print(f"Datos guardados correctamente en {nombre_fichero}")
        
    except KeyError as e:
        print(f"Error: No se pudo encontrar la clave {e} en la respuesta JSON para {nombre_ciudad}.")
    except Exception as e:
        print(f"Error al guardar los datos para {nombre_ciudad}: {e}")

def leer_ciudades(fichero):
    """Lee el fichero de ciudades y devuelve una lista limpia."""
    if not os.path.exists(fichero):
        print(f"Error: No se encuentra el fichero {fichero}. Creando uno de ejemplo con 'Madrid'.")
        with open(fichero, 'w', encoding='utf-8') as f:
            f.write("Madrid\n")
        return ["Madrid"]
        
    with open(fichero, 'r', encoding='utf-8') as f:
        # Leemos las líneas, quitamos espacios en blanco y saltos de línea
        ciudades = [linea.strip() for linea in f if linea.strip()] 
    print(f"Se han cargado {len(ciudades)} ciudades de {fichero}.")
    return ciudades

# --- Ejecución Principal del Script ---
if __name__ == "__main__":
    # 1. Asegurarnos de que el directorio de datos existe
    os.makedirs(DIRECTORIO_DATOS, exist_ok=True)
    
    # 2. Leer la lista de ciudades
    ciudades_a_procesar = leer_ciudades(FICHERO_CIUDADES)
    
    print(f"Iniciando proceso de registro para {len(ciudades_a_procesar)} ciudades.")
    
    # 3. Iterar por cada ciudad y procesarla
    for ciudad in ciudades_a_procesar:
        datos_actuales = obtener_datos_climaticos(ciudad, API_KEY)
        
        if datos_actuales:
            # Usamos el nombre de la lista (ej. "León") para el nombre del fichero
            guardar_datos_ciudad(datos_actuales, ciudad) 
        else:
            print(f"No se pudieron obtener datos para {ciudad}. Saltando.")
            
    print("Proceso de registro completado.")