"""
API del panel.
  GET  /api/frames, /api/summary, /api/frames/{cam}/{id}   lectura (la usa el panel)
  POST /api/ingest                                          recibe resultados del nodo con GPU

La ingesta esta DESHABILITADA si no existe INGEST_TOKEN. Ademas de la clave valida:
  contrato completo (Stage3Frame), sha256 del original, rutas derivadas del frame_id (no se
  confia en rutas enviadas por el cliente), tipo real de archivo (firma JPEG/PNG) y tamanos.
OJO: sobre HTTP simple la clave viaja sin cifrar. Aceptable para el prototipo; para produccion, TLS.
"""
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import ValidationError

from contracts.schemas import Stage3Frame

DATA = Path(os.getenv("DATA_DIR", "/data"))
TOKEN = os.getenv("INGEST_TOKEN", "")
SAFE = re.compile(r"^[A-Za-z0-9_-]+$")                     # evita rutas tipo ../../etc/passwd
FRAME_ID = re.compile(r"^(cam\d{2,})_\d{8}T\d{6}Z_(\d{6})$")
MAX_IMG = 12 * 1024 * 1024
MAX_JSON = 512 * 1024

app = FastAPI(title="VIGIA API")


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _atomic(path: Path, data: bytes) -> None:
    """Escribe a temporal y renombra: el panel nunca ve un archivo a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _frames(limit: Optional[int] = None) -> list:
    out = []
    for p in sorted((DATA / "json").rglob("*.json")):
        try:
            d = _load(p)
        except Exception:                      # archivo danado: se omite, no tumba el panel
            continue
        if "analysis" in d:                    # solo fotogramas ya procesados (etapa 3)
            out.append(d)
    out.sort(key=lambda d: d["timestamp_utc"])
    return out[-limit:] if limit else out


def _summary(frames: list) -> dict:
    plates = [d["plate"] for f in frames for d in f["analysis"]["detections"] if d.get("plate")]
    r_o = [p for p in plates if p.get("text_original")]
    r_e = [p for p in plates if p.get("text_enhanced")]
    avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    lat = []
    for f in frames:
        partes = [f["enhancement"].get("processing_ms"), f["analysis"].get("detect_ms"),
                  f["analysis"].get("ocr_ms")]
        partes = [x for x in partes if x is not None]
        if partes:
            lat.append(sum(partes))
    per_camera = None
    fichero = DATA / "summary" / "metricas.json"
    if fichero.exists():                       # la evaluacion humana por camara se conserva
        try:
            per_camera = _load(fichero).get("per_camera")
        except Exception:
            pass
    if per_camera is None:
        per_camera = [{"camera_id": c, "usable_without_replacement": None, "note": None}
                      for c in sorted({f["camera_id"] for f in frames})]
    return {
        "schema_version": "1.0", "total_frames": len(frames), "plates_detected": len(plates),
        "plates_read_original": len(r_o), "plates_read_enhanced": len(r_e),
        "avg_confidence_original": avg([p["confidence_original"] for p in r_o]),
        "avg_confidence_enhanced": avg([p["confidence_enhanced"] for p in r_e]),
        "avg_latency_ms_per_frame": avg(lat), "per_camera": per_camera,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "data_dir_exists": DATA.exists(), "ingest_enabled": bool(TOKEN)}


@app.get("/api/summary")
def summary():
    frames = _frames()
    if frames:                                 # se calcula sobre lo que hay: siempre al dia
        return _summary(frames)
    fichero = DATA / "summary" / "metricas.json"
    if fichero.exists():
        return _load(fichero)
    raise HTTPException(404, "Aun no hay resultados")


@app.get("/api/frames")
def frames(limit: int = 500):
    return _frames(max(1, min(limit, 5000)))


@app.get("/api/frames/{camera}/{name}")
def frame(camera: str, name: str):
    if not (SAFE.match(camera) and SAFE.match(name)):
        raise HTTPException(400, "nombre invalido")
    p = DATA / "json" / camera / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "no existe")
    return _load(p)


@app.post("/api/ingest")
async def ingest(frame: str = Form(...), original: UploadFile = File(...),
                 enhanced: Optional[UploadFile] = File(None),
                 authorization: Optional[str] = Header(None)):
    if not TOKEN:
        raise HTTPException(503, "Ingesta deshabilitada: falta INGEST_TOKEN en el servidor")
    if not hmac.compare_digest(authorization or "", f"Bearer {TOKEN}"):
        raise HTTPException(401, "token invalido")
    if len(frame) > MAX_JSON:
        raise HTTPException(413, "JSON demasiado grande")
    try:
        f = Stage3Frame.model_validate_json(frame)
    except ValidationError as e:
        err = e.errors()[0]
        raise HTTPException(422, f"El JSON no cumple el contrato ({'.'.join(map(str, err['loc']))}: {err['msg']})")

    tag, idx = FRAME_ID.match(f.frame_id).groups()
    if f.image_path != f"data/orig/{tag}/{idx}.jpg":
        raise HTTPException(400, "image_path no corresponde al frame_id")
    orig = await original.read(MAX_IMG + 1)
    if len(orig) > MAX_IMG:
        raise HTTPException(413, "original demasiado grande")
    if not orig.startswith(b"\xff\xd8\xff"):
        raise HTTPException(400, "el original debe ser JPEG")
    if hashlib.sha256(orig).hexdigest() != f.sha256:
        raise HTTPException(400, "el sha256 no coincide con el archivo original")

    enh = None
    ruta_enh = f.enhancement.enhanced_path
    if ruta_enh is not None:
        if ruta_enh != f"data/enh/{tag}/{idx}.png":
            raise HTTPException(400, "enhanced_path no corresponde al frame_id")
        if enhanced is None:
            raise HTTPException(400, "el JSON declara imagen mejorada pero no se envio el archivo")
        enh = await enhanced.read(MAX_IMG + 1)
        if len(enh) > MAX_IMG:
            raise HTTPException(413, "imagen mejorada demasiado grande")
        if not enh.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(400, "la imagen mejorada debe ser PNG")
    elif enhanced is not None:
        raise HTTPException(400, "se envio imagen mejorada que el JSON no declara")

    _atomic(DATA / "orig" / tag / f"{idx}.jpg", orig)             # imagenes primero...
    if enh is not None:
        _atomic(DATA / "enh" / tag / f"{idx}.png", enh)
    _atomic(DATA / "json" / tag / f"{idx}.json",                  # ...y el JSON al final
            json.dumps(f.model_dump(by_alias=True), indent=2, ensure_ascii=False).encode("utf-8"))
    return {"ok": True, "frame_id": f.frame_id,
            "plates": sum(1 for d in f.analysis.detections if d.plate)}
