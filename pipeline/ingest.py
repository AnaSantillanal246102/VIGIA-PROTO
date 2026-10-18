"""
Pipeline (Team 1): video / imagenes / stream (archivo) -> JPG + JSON de ETAPA 1.

  python -m pipeline.ingest --input clip.mp4 --camera 1 --every 5 --max-frames 200
  python -m pipeline.ingest --input carpeta_con_jpgs/ --camera 2

Para fuentes EN VIVO (APIs de camaras, HLS, RTSP) usar  python -m pipeline.live.
Escribe  data/orig/camNN/000123.jpg  y  data/json/camNN/000123.json
OJO: el "original" es el fotograma decodificado guardado sin ningun otro procesamiento.
Para evidencia real hay que conservar tambien el video de origen.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from contracts.schemas import Stage1Frame

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic_write_json(path: Path, data: dict) -> None:
    """Escribe a un temporal y renombra: el panel/API nunca leen un JSON a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_stage1(img, root, cam_num: int, idx: int, ts: datetime,
                 source_type: str, source_name: str, fps=None) -> dict:
    """Guarda el fotograma original (JPG) y su JSON de etapa 1 validado. Devuelve el dict."""
    root = Path(root)
    tag, cam_id = f"cam{cam_num:02d}", f"cam_{cam_num:02d}"
    rel = f"data/orig/{tag}/{idx:06d}.jpg"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(root / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(f"No se pudo escribir {root / rel}")
    h, w = img.shape[:2]
    frame = Stage1Frame.model_validate({
        "frame_id": f"{tag}_{ts:%Y%m%dT%H%M%SZ}_{idx:06d}",
        "camera_id": cam_id,
        "source": {"type": source_type, "name": source_name},
        "timestamp_utc": f"{ts:%Y-%m-%dT%H:%M:%SZ}",
        "frame_index": idx, "width": w, "height": h,
        "fps_source": fps, "image_path": rel,
        "sha256": sha256_file(root / rel),
    }).model_dump(by_alias=True)
    atomic_write_json(root / f"data/json/{tag}/{idx:06d}.json", frame)
    return frame


def iter_frames(src: str):
    """Rinde (indice, imagen BGR, fps) desde una carpeta de imagenes o un video."""
    p = Path(src)
    if p.is_dir():
        for i, f in enumerate(sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT)):
            img = cv2.imread(str(f))
            if img is not None:
                yield i, img, None
        return
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield i, img, fps
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--camera", type=int, required=True, help="numero de camara (1 -> cam01)")
    ap.add_argument("--source-type", default="file", choices=["file", "api", "hls", "rtsp"])
    ap.add_argument("--source-name", default=None)
    ap.add_argument("--every", type=int, default=1, help="tomar 1 de cada N fotogramas")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = sin limite")
    ap.add_argument("--root", default=".")
    ap.add_argument("--start-time", default=None, help="ISO UTC, ej. 2026-10-16T14:00:00Z (por defecto: ahora)")
    a = ap.parse_args()

    t0 = (datetime.strptime(a.start_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
          if a.start_time else datetime.now(timezone.utc).replace(microsecond=0))
    n = 0
    for i, img, fps in iter_frames(a.input):
        if i % a.every:
            continue
        ts = t0 + timedelta(seconds=(i / fps) if fps else i)
        write_stage1(img, a.root, a.camera, i, ts, a.source_type,
                     a.source_name or Path(a.input).name, fps)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    print(f"{n} fotogramas ingestados en data/orig/cam{a.camera:02d} y data/json/cam{a.camera:02d}")


if __name__ == "__main__":
    main()
