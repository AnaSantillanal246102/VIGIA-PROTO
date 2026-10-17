#!/usr/bin/env bash
# Uso en el servidor:  ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "Falta .env  ->  cp .env.example .env  y edita la contrasena"; exit 1; }
[ -f deploy/panel.htpasswd ] && [ -s deploy/panel.htpasswd ] || { echo "Falta el login  ->  ./deploy/crear_usuario.sh operador1"; exit 1; }
git pull --ff-only
docker compose up -d --build
# nginx monta archivos sueltos: si git o crear_usuario.sh los reemplazan, solo los ve al recrearlo.
docker compose up -d --force-recreate --no-deps web
docker compose ps
