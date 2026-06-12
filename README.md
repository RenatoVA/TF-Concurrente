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
