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
| `pipeline/` | `live.py` (camaras en vivo), `sources.py`, `selector.py`, `discover.py`, `ingest.py` (archivos) → JPG + JSON etapa 1 | Team 1 | `p1-pipeline` |
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

## Entrada de video en vivo (camaras publicas, HLS, RTSP)
```bash
pip install -r requirements.txt
python -m pipeline.discover --limit 10 --verify          # arma config/sources.local.json con camaras publicas
python -m pipeline.live --config config/sources.local.json --duration 300 --enhance bicubic
# en Colab/GPU, ademas con placas:  pip install -r requirements-gpu.txt
python -m pipeline.live --config config/sources.local.json --duration 300 --enhance bicubic --plates best.pt
```
Sin `--enhance/--plates` solo captura (sirve en el servidor LinuxONE). Detalle de fuentes y limites: `docs/fuentes_video.md`.

## Enviar resultados al panel desde Colab (GPU) -> servidor
1. En el servidor: `INGEST_TOKEN` en `.env` (`openssl rand -hex 24`), `HOST_DATA_DIR=./data`, `docker compose up -d --build`.
2. En Colab: `os.environ["VIGIA_TOKEN"] = userdata.get("VIGIA_TOKEN")` (Secretos de Colab) y:
```python
from modules.plates import PlateModels
from modules.publish import procesar_imagen
models = PlateModels("/content/drive/MyDrive/vigia/best.pt")
frame, vista = procesar_imagen("foto.jpg", models, root=".", camera=1, enhance="bicubic",
                               push_url="http://IP_DEL_SERVIDOR:PUERTO")
```
`vista` trae los rectangulos verdes (solo para mirar; el original no se toca). En vivo: `python -m pipeline.live ... --plates best.pt --push http://IP:PUERTO`.
La API valida contrato, sha256 del original y rutas antes de guardar. Sobre HTTP la clave viaja sin cifrar (prototipo).

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
