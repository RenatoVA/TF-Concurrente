#!/usr/bin/env bash
# seed_insights.sh — carga los 6 datasets de EDA en MongoDB
# Requiere que el contenedor api esté corriendo con acceso al dataset.
set -e

echo "[seed] Ejecutando taxi seed-insights dentro del contenedor api..."
docker compose exec api taxi seed-insights \
  --file /data/yellow_tripdata_2015-01.csv \
  --mongo mongodb://mongo:27017

echo "[seed] Insights cargados en MongoDB."
echo "[seed] Verificando con GET /insights/trips_por_hora..."
curl -s http://localhost:8080/insights/trips_por_hora | head -c 200
echo ""
