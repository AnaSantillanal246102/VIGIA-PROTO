"""API minima (solo lectura): sirve al panel los JSON de etapa 3 y el resumen.
Pendiente (hoja de ruta): conectar PostgreSQL y registrar cada consulta en audit_log."""
import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException

DATA = Path(os.getenv("DATA_DIR", "/data"))
SAFE = re.compile(r"^[A-Za-z0-9_-]+$")      # evita rutas tipo ../../etc/passwd

app = FastAPI(title="VIGIA API")


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok", "data_dir_exists": DATA.exists()}


@app.get("/api/summary")
def summary():
    p = DATA / "summary" / "metricas.json"
    if not p.exists():
        raise HTTPException(404, "Aun no hay summary/metricas.json")
    return _load(p)


@app.get("/api/frames")
def frames():
    out = []
    for p in sorted((DATA / "json").rglob("*.json")):
        d = _load(p)
        if "analysis" in d:                  # solo fotogramas ya procesados (etapa 3)
            out.append(d)
    return sorted(out, key=lambda d: d["timestamp_utc"])


@app.get("/api/frames/{camera}/{name}")
def frame(camera: str, name: str):
    if not (SAFE.match(camera) and SAFE.match(name)):
        raise HTTPException(400, "nombre invalido")
    p = DATA / "json" / camera / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "no existe")
    return _load(p)
