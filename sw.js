// sw.js - Service Worker para Weather Intelligence v3.3.2 (INV-WEATHER-10, INV-WEATHER-22)
const CACHE_NAME = 'weather-v3.3.2';
const SHELL_ASSETS = [
    './',
    'index.html',
    'embed.html',
    'manifest.json',
    'icon.svg',
    'icon-192.png',
    'icon-512.png',
    'icon-maskable-192.png',
    'icon-maskable-512.png',
    'apple-touch-icon.png',
    'datalaria_logo_transp.png'
];

// 1. Install - Pre-cache Shell de forma tolerante a fallos
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return Promise.allSettled(
                SHELL_ASSETS.map((asset) => cache.add(new Request(asset, { cache: 'reload' })).catch(() => {}))
            );
        }).then(() => self.skipWaiting())
    );
});

// 2. Activate - Limpiar versiones obsoletas inmediatamente
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.map((key) => {
                    if (key !== CACHE_NAME && key.startsWith('weather-')) {
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

// 3. Fetch Handler con estrategias diferenciadas (INV-WEATHER-10)
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Solo gestionar peticiones GET
    if (event.request.method !== 'GET') return;

    // REGLA ESTRICTA INV-WEATHER-10: NUNCA cachear archivos CSV planos
    if (url.pathname.endsWith('.csv') || url.pathname.includes('/datos/')) {
        return; // Petición directa a red
    }

    const scopePath = new URL(self.registration.scope).pathname;
    const isAppShellNav = url.origin === self.location.origin && (
        url.pathname === scopePath ||
        url.pathname === scopePath + 'index.html' ||
        url.pathname === '/' ||
        url.pathname === '/apps/weather/'
    );

    // A. Shell / Navegación Start URL / Assets locales (Stale-While-Revalidate)
    if (
        isAppShellNav ||
        url.pathname.endsWith('.html') ||
        url.pathname.endsWith('.svg') ||
        url.pathname.endsWith('.png') ||
        (url.pathname.endsWith('.json') && url.pathname.includes('manifest'))
    ) {
        event.respondWith(
            caches.match(event.request, { ignoreSearch: isAppShellNav || event.request.mode === 'navigate' }).then((cached) => {
                const networkFetch = fetch(event.request).then((response) => {
                    if (response && response.status === 200) {
                        const clone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                    }
                    return response;
                }).catch(() => cached || (event.request.mode === 'navigate' ? (caches.match('./') || caches.match('index.html')) : undefined));

                return cached || networkFetch;
            }).catch(() => caches.match('./') || caches.match('index.html') || fetch(event.request))
        );
        return;
    }

    // B. JSON de Datos (derived/**, global_summary, predicciones) -> Stale-While-Revalidate
    if (
        (url.pathname.endsWith('.json') || url.pathname.includes('/derived/')) &&
        (url.origin === self.location.origin || url.hostname.includes('datalaria-app-weather.netlify.app') || url.pathname.includes('/apps/weather/'))
    ) {
        event.respondWith(
            caches.open(CACHE_NAME).then((cache) => {
                return cache.match(event.request).then((cached) => {
                    const fetchPromise = fetch(event.request).then((networkResponse) => {
                        if (networkResponse && networkResponse.status === 200) {
                            cache.put(event.request, networkResponse.clone());
                        }
                        return networkResponse;
                    }).catch(() => cached);

                    return cached || fetchPromise;
                });
            }).catch(() => fetch(event.request))
        );
        return;
    }

    // C. Teselas del mapa Leaflet -> Network-first con fallback a caché
    if (url.hostname.includes('tile.openstreetmap.org')) {
        event.respondWith(
            fetch(event.request).then((networkResponse) => {
                if (networkResponse && networkResponse.status === 200) {
                    const clone = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                }
                return networkResponse;
            }).catch(() => caches.match(event.request))
        );
        return;
    }

    // Nota: Las librerías CDN (Choices, Leaflet, Chart.js, Feather) no se interceptan;
    // se delegan a la caché nativa HTTP del navegador para evitar ERR_CACHE_READ_FAILURE.
});
