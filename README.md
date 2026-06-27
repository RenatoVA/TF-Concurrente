# TF-Concurrente — CC65 Programación Concurrente y Distribuida (UPC)

Cargador de datos concurrente + entrenamiento paralelo de regresión lineal en Go puro (stdlib únicamente).

## Requisitos

- Go 1.22 o superior
- Dataset CSV en `data/yellow_tripdata_2015-01.csv` (~2GB, no incluido en el repo)

## Estructura del proyecto

```
cmd/taxi/          CLI principal (subcomandos: load, train, benchmark, stats)
internal/loader/   Parte A: carga concurrente productor–consumidor
internal/model/    Parte B: regresión lineal + SGD paralelo
internal/stats/    Agregados estadísticos para análisis exploratorio
data/              CSV de entrada (gitignored) + salidas generadas
```

## Compilar

```bash
go build -o taxi ./cmd/taxi
```

## Subcomandos

### `load` — Parte A: carga y validación concurrente

```bash
./taxi load --file data/yellow_tripdata_2015-01.csv --workers 4
```

Imprime reporte con total leídas, válidas, descartes por regla (R1..R9), throughput y tiempo.

### `train` — Partes A + B: carga + entrenamiento + evaluación

```bash
./taxi train \
  --file data/yellow_tripdata_2015-01.csv \
  --workers 4 \
  --epochs 10 \
  --lr 0.01 \
  --batch 1024 \
  --seed 42 \
  --output data/model.json
```

Guarda el modelo en `data/model.json` para uso de la API (entregable 2).

### `benchmark` — Tabla de speedup

```bash
# Con todas las filas (tarda varios minutos)
./taxi benchmark --file data/yellow_tripdata_2015-01.csv

# Con límite de filas para pruebas rápidas
./taxi benchmark --file data/yellow_tripdata_2015-01.csv --limit 500000 --epochs 3
```

Ejecuta carga + entrenamiento con workers = 1, 2, 4, 8 y genera tabla de speedup.

### `stats` — Análisis exploratorio

```bash
./taxi stats --file data/yellow_tripdata_2015-01.csv --outdir data
```

Exporta a `data/`:
- `trips_por_hora.csv`
- `trips_por_dia_semana.csv`
- `histograma_duracion.csv` (bins de 2 min)
- `histograma_distancia.csv` (bins de 1 mi)
- `velocidad_media_por_hora.csv`
- `top_celdas_pickup.csv` (grilla 0.01°, top 50 celdas)

## Tests con race detector

```bash
# Ejecutar todos los tests con el detector de race conditions
go test -race ./...

# Tests verbose con detalles
go test -race -v ./internal/loader/...
go test -race -v ./internal/model/...
```

## Arquitectura de concurrencia

### Parte A — Cargador (loader)

Patrón productor–consumidor con fan-out/fan-in:

```
Reader goroutine  ──batchCh──▶  Worker 0  ──┐
  bufio.Scanner                  Worker 1  ──┤──outCh──▶  Collector ──▶ []Trip
  lotes de 10,000                Worker N  ──┘
  cap(batchCh) = 2×W            cap(outCh) = W×1024
```

- **Contadores atómicos** (`sync/atomic`): uno por regla R1-R9, compartidos sin mutex.
- **Cierre coordinado**: una goroutine espera el WaitGroup de workers y cierra `outCh`; evita el race de cierre múltiple.

### Parte B — SGD paralelo (model)

```
Coordinator ──paramCh[i]──▶ Worker_i ──┐
                                        ├── gradCh ──▶ Aggregator
Coordinator ◀──epochResultCh ──────────┘
```

- **El agregador es la ÚNICA goroutine que escribe θ** (patrón actor).
- Los workers reciben una **copia independiente** de θ por canal al inicio de cada época.
- La barrera de época usa **mensajes `kindDone` en `gradCh`**: cada worker envía `kindDone` como último mensaje de la época, garantizando ordering correcto (Go preserva el orden de envío en un canal).
- `go test -race ./...` pasa limpio.

## Modelo

- **Target**: `DurationMin` (duración del viaje en minutos)
- **Features** (8 + bias): trip_distance, sin/cos hora, isWeekend, pickup_lat, pickup_lon, isAirport, passenger_count
- **Estandarización**: z-score ajustada solo sobre el train set (split 80/20)
- **Salida**: `data/model.json` con pesos, medias, desviaciones y metadatos de entrenamiento

---

## PC4 · Despliegue distribuido

### Topología (estrella / parameter-server)

```
                    ┌──────────────┐
              ┌────▶│    node1     │
              │TCP  │ (worker TCP) │
              │     └──────────────┘
┌─────────────┤
│  API/coord  │────▶│    node2     │
│  :8080      │     │ (worker TCP) │
└──────┬──────┘     └──────────────┘
       │       ────▶│    node3     │
       │            │ (worker TCP) │
  ┌────▼────┐       └──────────────┘
  │  Redis  │
  │  Mongo  │
  └─────────┘
```

- **API** (`taxi api`): coordinador + agregador. Es la única goroutine que escribe θ. Expone REST en `:8080`.
- **Nodos** (`taxi node`): servidores TCP que calculan `AccumulateGradient` sobre su partición y devuelven el gradiente al coordinador.
- **MongoDB**: persiste modelos, training runs, predicciones y los 6 insights de EDA.
- **Redis**: cachea `model:active` (lectura en cada `/predict`), predicciones individuales (TTL 5 min) y métricas del cluster.
- **Dataset CSV**: montado como volumen read-only (`./data:/data:ro`) solo en la API; nunca en Mongo.

### Levantar el cluster

```bash
# Con docker compose (recomendado)
bash scripts/run.sh
# o directamente:
docker compose up --build
```

### Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/train` | Carga CSV → entrena distribuido → guarda en Mongo + Redis |
| `GET`  | `/predict` | Predice desde modelo en Redis (< 100 ms) |
| `GET`  | `/model` | Metadata + métricas del modelo activo (Mongo) |
| `GET`  | `/metrics` | Métricas del cluster (Redis): nodos, duración, p50/p95 |
| `GET`  | `/insights/{name}` | EDA dataset por nombre (Mongo) |
| `GET`  | `/healthz` | Liveness probe |

### Ejemplos de uso

```bash
# Entrenar con 3 nodos, 100 épocas, 1 millón de filas
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"file":"/data/yellow_tripdata_2015-01.csv","limit":1000000,"epochs":100,"lr":0.05}'

# Predecir duración de un viaje
curl "http://localhost:8080/predict?trip_distance=2.5&hour=18&day_of_week=3&pickup_lat=40.75&pickup_lon=-73.98&rate_code=1&passenger_count=1"

# Ver modelo activo
curl http://localhost:8080/model

# Ver insights de EDA
curl http://localhost:8080/insights/trips_por_hora

# Health check
curl http://localhost:8080/healthz

# Cargar insights en MongoDB (con cluster corriendo)
bash scripts/seed_insights.sh

# Benchmark de latencia (p95 < 100 ms)
bash scripts/bench_predict.sh
```

### Qué persiste dónde

| Dato | Almacenamiento |
|------|----------------|
| Modelo entrenado (pesos, métricas) | MongoDB colección `models` |
| Historial de training runs | MongoDB colección `training_runs` |
| Predicciones individuales | MongoDB colección `predictions` (async) |
| Insights EDA (6 datasets) | MongoDB colección `insights` |
| Modelo activo (para predicción rápida) | Redis clave `model:active` |
| Cache de predicciones | Redis `pred:cache:<hash>` (TTL 5 min) |
| Métricas del cluster | Redis `cluster:metrics` |
| Dataset CSV (~1.9 GB) | Volumen local `./data/` (nunca en DB) |

### Escalar nodos

Para agregar un cuarto nodo: añadir `node4` en `docker-compose.yml` con el mismo patrón y agregar `node4:9100` a la variable `NODES` del servicio `api`.

### Tests

```bash
# Tests unitarios (sin docker)
go test ./...

# Test de equivalencia distribuido==local
go test -v ./internal/cluster/...

# Con race detector (requiere CGO habilitado en Linux/macOS)
go test -race ./...
```
