# VIGIA — Mejoramiento inteligente de video para videovigilancia urbana (prototipo)

```
APIs / Video ─► [pipeline] ─① JSON─► [modules/enhancement] ─② JSON─► [modules/plates] ─③ JSON─► [panel]
                 Team 1                    Team 2                        Team 3                  Team 1
```
Cada etapa copia todo lo anterior y agrega su bloque. Formato exacto en `contracts/`.

## Estructura y responsables

| Carpeta | Qué contiene | Equipo | Rama |
|---|---|---|---|
| `contracts/` | Esquemas Pydantic + validador + ejemplos (**compartido, no editar sin avisar**) | Todos | `main` |
| `pipeline/` | `ingest.py`: video/imágenes/stream → JPG + JSON etapa 1 | Team 1 | `p1-pipeline` |
| `modules/enhancement.py` | Mejora de resolución (línea base bicúbica; aquí se enchufa Real-ESRGAN) | Team 2 | `p2-mejora` |
| `modules/plates.py` | YOLO de placas + OCR → JSON etapa 3 | Team 3 | `p3-placas` |
| `panel/` | Interfaz web estática (original vs. mejorado, placas, métricas) | Team 1 | `p1-panel` |
| `api/` | FastAPI de solo lectura que sirve JSON al panel | Backend | `p5-backend` |
| `db/` | Esquema PostgreSQL (frames, detections, audit_log) | Backend | `p5-backend` |
| `deploy/`, `docker-compose.yml`, `deploy.sh` | Despliegue en LinuxONE y en laptops | Backend | `p5-backend` |
| `demo_data/` | Datos **sintéticos** para que el panel funcione desde el primer minuto | Todos | `main` |
| `data/` | Resultados reales (no se sube a git) | — | — |
| `notebooks/` | Notebooks de Colab | Team 2, 3 | sus ramas |

## Probar el panel (cualquier máquina con Docker)
```bash
cp .env.example .env        # cambia POSTGRES_PASSWORD
docker compose up -d --build
# abrir http://localhost  (con WEB_PORT=80)
```
Por defecto muestra `demo_data/` (sintético). Para ver resultados reales: `HOST_DATA_DIR=./data` en `.env`.

## Correr los módulos (desde la raíz del repo; `pip install -r contracts/requirements.txt opencv-python-headless`)
```bash
python -m pipeline.ingest --input clip.mp4 --camera 1 --every 5 --max-frames 200   # etapa 1
python -m modules.enhancement --in-dir data/json --root .                          # etapa 2
python -m modules.plates --weights best.pt --in-dir data/json --root .             # etapa 3 (necesita ultralytics, easyocr)
python contracts/validate.py 3 data/json                                           # validar
```

## Reglas del equipo
- Nada de `.env`, llaves, videos ni pesos `.pt` en git (el `.gitignore` ya los excluye).
- Una rama por persona; solo el integrador fusiona a `main` (con Pull Request).
- Las cajas `bbox_xyxy` van **en píxeles de la imagen original**.
- Imagen mejorada = apoyo, nunca evidencia.
- Las imágenes Docker se construyen **en el servidor** (s390x), nunca en una laptop.
