#!/usr/bin/env bash
# Crea o cambia un usuario del login del panel. Pide la contrasena dos veces.
# Uso, desde la raiz del repo:  ./deploy/crear_usuario.sh operador1
# Usa un nombre de rol (operador1), no uno personal: se ve en el panel.
set -euo pipefail
cd "$(dirname "$0")"
usuario="${1:?Uso: ./deploy/crear_usuario.sh <usuario>}"
[[ "$usuario" =~ ^[A-Za-z0-9._-]{1,32}$ ]] || { echo "Usuario invalido: solo letras, numeros, punto, guion y guion bajo"; exit 1; }
[ -d panel.htpasswd ] && { echo "deploy/panel.htpasswd es una carpeta (Docker la creo al faltar el archivo): borrala con  sudo rmdir deploy/panel.htpasswd"; exit 1; }

hash="$(openssl passwd -apr1)"
touch panel.htpasswd
awk -F: -v u="$usuario" '$1 != u' panel.htpasswd > panel.htpasswd.tmp
printf '%s:%s\n' "$usuario" "$hash" >> panel.htpasswd.tmp
mv panel.htpasswd.tmp panel.htpasswd
chmod 644 panel.htpasswd   # nginx lo lee con su propio usuario dentro del contenedor

echo "Listo: $usuario en deploy/panel.htpasswd"
echo "Si nginx ya esta corriendo:  docker compose up -d --force-recreate --no-deps web"
