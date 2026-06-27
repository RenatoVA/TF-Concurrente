#!/usr/bin/env bash
# bench_predict.sh — mide latencia de GET /predict (objetivo: p95 < 100 ms)
set -e

HOST="${BENCH_HOST:-localhost}"
PORT="${BENCH_PORT:-8080}"
N="${BENCH_N:-100}"
BASE_URL="http://${HOST}:${PORT}/predict"

PARAMS=(
  "trip_distance=2.5&hour=18&day_of_week=3&pickup_lat=40.75&pickup_lon=-73.98&rate_code=1&passenger_count=1"
  "trip_distance=1.0&hour=8&day_of_week=1&pickup_lat=40.70&pickup_lon=-74.00&rate_code=1&passenger_count=2"
  "trip_distance=5.3&hour=22&day_of_week=5&pickup_lat=40.76&pickup_lon=-73.97&rate_code=2&passenger_count=1"
  "trip_distance=0.8&hour=12&day_of_week=0&pickup_lat=40.72&pickup_lon=-73.99&rate_code=1&passenger_count=3"
)

echo "[bench] Enviando $N requests a ${BASE_URL}..."
latencies=()

for i in $(seq 1 "$N"); do
  params="${PARAMS[$((i % ${#PARAMS[@]}))]}"
  ms=$(curl -s -o /dev/null -w "%{time_total}" "${BASE_URL}?${params}" | awk '{printf "%.0f", $1 * 1000}')
  latencies+=("$ms")
done

# Sort and compute percentiles
IFS=$'\n' sorted=($(sort -n <<<"${latencies[*]}")); unset IFS

count=${#sorted[@]}
p50_idx=$(( count / 2 ))
p95_idx=$(( count * 95 / 100 ))

p50="${sorted[$p50_idx]}"
p95="${sorted[$p95_idx]}"
min="${sorted[0]}"
max="${sorted[$((count-1))]}"

echo ""
echo "=== Benchmark /predict (N=${count}) ==="
printf "  min:  %d ms\n" "$min"
printf "  p50:  %d ms\n" "$p50"
printf "  p95:  %d ms\n" "$p95"
printf "  max:  %d ms\n" "$max"

if [ "$p95" -lt 100 ]; then
  echo "  [OK] p95 < 100 ms"
else
  echo "  [FAIL] p95 >= 100 ms"
  exit 1
fi
