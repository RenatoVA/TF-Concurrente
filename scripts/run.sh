#!/usr/bin/env bash
# run.sh — levanta el cluster completo con docker compose
set -e
cd "$(dirname "$0")/.."
echo "[run] Construyendo imagen y levantando servicios..."
docker compose up --build
