#!/usr/bin/env bash
# Uso en el servidor:  ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Falta .env  ->  cp .env.example .env  y edita la contrasena"; exit 1; }
git pull --ff-only
docker compose up -d --build
docker compose ps
