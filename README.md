# Weather Service (`apps/weather`)

Dashboard meteorológico interactivo y motor de recolección de telemetría global con predicción determinista mediante Machine Learning (Scikit-Learn).

## Arquitectura del Módulo

- **Frontend (`index.html`):** Aplicación de página única construida con JavaScript Vanilla, Tailwind CSS por CDN, PapaParse.js para lectura eficiente de CSVs en streaming, Chart.js para visualización de series temporales multivariables y Leaflet.js para el mapa geográfico.
- **Backend Batch & Machine Learning (`read_weather.py`):** Script en Python 3.10 que consulta la API de OpenWeatherMap v2.5 (`/weather` y `/forecast`), actualiza el dataset histórico en `datos/` y entrena modelos de Regresión Lineal para predecir las temperaturas de las próximas 24 horas.
- **Automatización CI/CD (`.github/workflows/actualizar-clima.yml`):** Tarea programada en GitHub Actions que se ejecuta 3 veces al día (06:00, 14:00 y 22:00 UTC) utilizando el secreto `OPENWEATHER_API_KEY`.
- **Despliegue y Hosting:**
  - Repositorio upstream: [`https://github.com/Dalaez/app_weather.git`](https://github.com/Dalaez/app_weather.git)
  - Hosting en vivo: [`https://datalaria-app-weather.netlify.app/`](https://datalaria-app-weather.netlify.app/)
  - Proxy en el portal Datalaria: Enrutado vía `static/_redirects` (`/apps/weather/* -> https://datalaria-app-weather.netlify.app/:splat 200!`).

## Ejecución Local

### Recolector y Motor Predictivo (Python)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar clave de API
export OPENWEATHER_API_KEY="tu_clave_de_openweather"  # En Linux/macOS
$env:OPENWEATHER_API_KEY="tu_clave_de_openweather"    # En PowerShell

# 3. Ejecutar recolección y predicción
python read_weather.py
```

### Visualización Frontend

Abre `index.html` directamente en tu navegador o levanta un servidor estático local:

```bash
npx serve .
# o con Python:
python -m http.server 8080
```
