"""
Publicar resultados en el panel: detectar placas, dibujar el rectangulo VERDE y enviar.

  from modules.plates import PlateModels
  from modules.publish import procesar_imagen
  models = PlateModels("/content/drive/MyDrive/vigia/best.pt")
  frame, vista = procesar_imagen("foto.jpg", models, root=".", camera=1, enhance="bicubic",
                                 push_url="http://IP_DEL_SERVIDOR:PUERTO")   # token en VIGIA_TOKEN

La imagen anotada (vista) es SOLO para mirar: se guarda aparte en data/preview/ y el original
nunca se modifica. El panel dibuja las cajas por su cuenta desde el JSON.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from pipeline.ingest import atomic_write_json, write_stage1
from pipeline.live import next_index, with_skipped_enhancement
from pipeline.sources import UA

log = logging.getLogger("vigia.publish")
VERDE = (0, 255, 0)   # BGR


def mejor_lectura(plate: dict):
    """(texto, confianza) de la lectura con mayor confianza entre original y mejorada."""
    c = [(plate.get("text_enhanced"), plate.get("confidence_enhanced") or 0.0),
         (plate.get("text_original"), plate.get("confidence_original") or 0.0)]
    c = [x for x in c if x[0]]
    return max(c, key=lambda x: x[1]) if c else (None, 0.0)


def annotate(img: np.ndarray, detections: list, color=VERDE) -> np.ndarray:
    """Copia de la imagen con un rectangulo verde por placa y su texto leido."""
    out = img.copy()
    grosor = max(2, round(max(out.shape[:2]) / 250))
    escala = max(0.5, max(out.shape[:2]) / 900)
    for d in detections:
        p = d.get("plate") or {}
        x1, y1, x2, y2 = map(int, p.get("bbox_xyxy") or d["bbox_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, grosor)
        texto, conf = mejor_lectura(p)
        etiqueta = f"{texto} {conf:.0%}" if texto else "no legible"
        (tw, th), base = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        ty = y1 - 6 if y1 - th - base - 6 >= 0 else y2 + th + 6           # dentro de la imagen
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(out, etiqueta, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor)
    return out


# ---------- envio al servidor ----------
def _multipart(fields: dict, files: dict):
    b = uuid.uuid4().hex
    partes = []
    for k, v in fields.items():
        partes.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'.encode() + v.encode("utf-8") + b"\r\n")
    for k, (nombre, datos, ctype) in files.items():
        partes.append((f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{nombre}"\r\n'
                       f'Content-Type: {ctype}\r\n\r\n').encode() + datos + b"\r\n")
    partes.append(f"--{b}--\r\n".encode())
    return b"".join(partes), f"multipart/form-data; boundary={b}"


def enviar(frame: dict, root, base_url: str, token: str = None, timeout: float = 30, reintentos: int = 2) -> dict:
    """Sube JSON etapa 3 + imagen original (+ mejorada) a POST {base_url}/api/ingest."""
    token = token or os.environ.get("VIGIA_TOKEN")
    if not token:
        raise RuntimeError("Falta la clave: define la variable de entorno VIGIA_TOKEN")
    root = Path(root)
    files = {"original": (Path(frame["image_path"]).name, (root / frame["image_path"]).read_bytes(), "image/jpeg")}
    ruta_enh = frame["enhancement"].get("enhanced_path")
    if ruta_enh:
        files["enhanced"] = (Path(ruta_enh).name, (root / ruta_enh).read_bytes(), "image/png")
    cuerpo, ctype = _multipart({"frame": json.dumps(frame, ensure_ascii=False)}, files)
    req = urllib.request.Request(base_url.rstrip("/") + "/api/ingest", data=cuerpo, method="POST",
                                 headers={"Content-Type": ctype, "Authorization": f"Bearer {token}", "User-Agent": UA})
    ultimo = None
    for intento in range(reintentos + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detalle = e.read().decode("utf-8", "replace")[:300]
            if e.code < 500:                                  # 4xx: reintentar no sirve
                raise RuntimeError(f"El servidor rechazo el fotograma ({e.code}): {detalle}")
            ultimo = f"{e.code}: {detalle}"
        except urllib.error.URLError as e:
            ultimo = e
        time.sleep(2 ** intento)
    raise RuntimeError(f"No se pudo enviar al servidor: {ultimo}")


def make_push_step(url: str, root, token: str = None):
    """Paso para pipeline.live: envia cada fotograma procesado; si falla, queda guardado en local."""
    def step(frame: dict) -> dict:
        try:
            r = enviar(frame, root, url, token)
            log.info("enviado al panel: %s (%d placa/s)", r["frame_id"], r["plates"])
        except Exception as e:
            log.warning("no se pudo enviar %s al panel: %s (queda guardado en local)", frame["frame_id"], e)
        return frame
    return step


# ---------- funcion de un solo paso ----------
def procesar_imagen(imagen, models, root=".", camera: int = 1, enhance: str = None, scale: int = 4,
                    detect_on: str = "original", push_url: str = None, source_name: str = "colab",
                    source_type: str = "file"):
    """imagen (ruta o array BGR) -> detecta placas, lee matriculas, guarda JSON y vista con
    rectangulos verdes y, si hay push_url, lo envia al panel. Devuelve (frame_json, vista_BGR)."""
    from modules.plates import process_frame
    root = Path(root)
    if isinstance(imagen, (str, Path)):
        img = cv2.imread(str(imagen))
        if img is None:
            raise FileNotFoundError(imagen)
        source_name = Path(imagen).name
    else:
        img = imagen
    idx = next_index(root, camera)
    frame = write_stage1(img, root, camera, idx, datetime.now(timezone.utc).replace(microsecond=0),
                         source_type, source_name, None)
    if enhance:
        from modules.enhancement import enhance_frame
        frame = enhance_frame(frame, root, enhance, scale)
    else:
        frame = with_skipped_enhancement(frame)
    frame = process_frame(frame, models, root, detect_on)
    tag = f"cam{camera:02d}"
    atomic_write_json(root / f"data/json/{tag}/{idx:06d}.json", frame)

    vista = annotate(img, frame["analysis"]["detections"])
    (root / f"data/preview/{tag}").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / f"data/preview/{tag}/{idx:06d}.jpg"), vista)
    if push_url:
        enviar(frame, root, push_url)
    return frame, vista
