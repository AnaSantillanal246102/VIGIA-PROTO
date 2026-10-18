"""
Entrada en vivo (Team 1): camaras -> seleccion de fotogramas -> JSON etapa 1 -> (mejora) -> (placas).

  python -m pipeline.live --config config/sources.json --duration 300
  python -m pipeline.live --config config/sources.json --duration 300 --enhance bicubic
  python -m pipeline.live --config config/sources.json --duration 300 --enhance bicubic --plates best.pt

Una hebra por camara lee y selecciona; una hebra trabajadora guarda y encadena los modulos.
La cola es acotada: si el procesamiento no da abasto se descartan los fotogramas MAS VIEJOS
(en tiempo real importa lo reciente). Ctrl-C, --duration o --max-frames detienen limpiamente.
Sin --enhance/--plates solo captura (corre bien en el servidor LinuxONE, sin GPU).
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from pipeline.ingest import atomic_write_json, write_stage1
from pipeline.selector import FrameSelector
from pipeline.sources import build_source

log = logging.getLogger("vigia.live")

STAGE1_TYPE = {"snapshot": "api", "hls": "hls", "rtsp": "rtsp", "file": "file"}


@dataclass
class Job:
    cam: int
    idx: int
    ts: datetime
    img: np.ndarray
    fps: Optional[float]
    source_type: str
    source_name: str


def load_config(path) -> List[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["sources"] if isinstance(raw, dict) else raw
    out, seen = [], set()
    for c in items:
        if not c.get("enabled", True):
            continue
        for k in ("camera", "type", "url"):
            if k not in c:
                raise ValueError(f"Fuente sin '{k}': {c}")
        if c["type"] not in STAGE1_TYPE:
            raise ValueError(f"type invalido '{c['type']}' (usa: {', '.join(STAGE1_TYPE)})")
        if c["camera"] in seen:
            raise ValueError(f"Numero de camara repetido: {c['camera']}")
        seen.add(c["camera"])
        c.setdefault("name", f"cam{c['camera']:02d}")
        out.append(c)
    if not out:
        raise ValueError("No hay fuentes habilitadas en la configuracion")
    return out


def next_index(root: Path, cam: int) -> int:
    d = root / "data/json" / f"cam{cam:02d}"
    nums = [int(p.stem) for p in d.glob("*.json") if p.stem.isdigit()] if d.exists() else []
    return max(nums) + 1 if nums else 0


def prune(root: Path, cam: int, keep: int) -> None:
    """Deja solo los ultimos `keep` fotogramas de la camara (proteccion de disco)."""
    tag = f"cam{cam:02d}"
    files = sorted((root / "data/json" / tag).glob("*.json"))
    for f in files[:-keep] if len(files) > keep else []:
        for p in (f, root / "data/orig" / tag / (f.stem + ".jpg"), root / "data/enh" / tag / (f.stem + ".png")):
            p.unlink(missing_ok=True)


def with_skipped_enhancement(frame: dict) -> dict:
    """Etapa 1 -> etapa 2 'sin mejora', para poder correr placas sin el Modulo 1."""
    d = dict(frame)
    d["enhancement"] = {"model": "ninguno", "scale": None, "enhanced_path": None, "width": None,
                        "height": None, "processing_ms": None,
                        "skipped_reason": "modulo 1 no ejecutado", "is_evidence": False}
    return d


def _put_drop_oldest(q: queue.Queue, job: Job, glob: Counter, lock: threading.Lock):
    while True:
        try:
            q.put_nowait(job)
            return
        except queue.Full:
            try:
                q.get_nowait()
                with lock:
                    glob["descartados_por_cola_llena"] += 1
            except queue.Empty:
                pass


def producer(cfg, root, q, stop, stats, glob, lock):
    cam, st = cfg["camera"], stats[cfg["camera"]]
    sel = FrameSelector(cfg.get("min_sharpness", 10.0), cfg.get("min_change", 1.5),
                        cfg.get("force_every_s", 60.0))
    try:
        src = build_source(cfg)
        idx = next_index(root, cam)
        for img, fps in src.frames(stop):
            ok, reason = sel.accept(img, time.monotonic())
            with lock:
                st["leidos"] += 1
                st["aceptados" if ok else f"rechazo_{reason}"] += 1
            if not ok:
                continue
            job = Job(cam, idx, datetime.now(timezone.utc).replace(microsecond=0), img, fps,
                      STAGE1_TYPE[cfg["type"]], cfg["name"])
            idx += 1
            _put_drop_oldest(q, job, glob, lock)
    except Exception as e:
        log.error("[cam%02d] la fuente se detuvo: %s", cam, e)
        with lock:
            st["errores_fuente"] += 1


def consumer(q, root, steps, stop, producers_done, stats, glob, lock, keep_last, max_frames):
    total = 0
    while not (producers_done.is_set() and q.empty()):
        try:
            job = q.get(timeout=0.5)
        except queue.Empty:
            continue
        st, t0 = stats[job.cam], time.perf_counter()
        try:
            frame = write_stage1(job.img, root, job.cam, job.idx, job.ts, job.source_type,
                                 job.source_name, job.fps)
            if steps:
                for step in steps:
                    frame = step(frame)
                atomic_write_json(root / f"data/json/cam{job.cam:02d}/{job.idx:06d}.json", frame)
            dets = frame.get("analysis", {}).get("detections", [])
            read = sum(1 for d in dets if (d.get("plate") or {}).get("text_enhanced")
                       or (d.get("plate") or {}).get("text_original"))
            log.info("cam%02d #%06d listo en %.0f ms%s", job.cam, job.idx,
                     (time.perf_counter() - t0) * 1000,
                     f" | placas detectadas: {len(dets)}, legibles: {read}" if "analysis" in frame else "")
            with lock:
                st["procesados"] += 1
            if keep_last:
                prune(root, job.cam, keep_last)
        except Exception as e:
            log.error("cam%02d #%06d fallo al procesar: %s", job.cam, job.idx, e)
            with lock:
                st["errores_proceso"] += 1
        total += 1
        if max_frames and total >= max_frames:
            stop.set()
            break                       # limite exacto: lo que quede en la cola se ignora


def run(configs: List[dict], root, steps: Optional[List[Callable[[dict], dict]]] = None,
        duration: float = 0, max_frames: int = 0, keep_last: int = 0, queue_size: int = 16):
    root = Path(root)
    stats = {c["camera"]: Counter() for c in configs}
    glob, lock = Counter(), threading.Lock()
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    stop, producers_done = threading.Event(), threading.Event()

    prods = [threading.Thread(target=producer, args=(c, root, q, stop, stats, glob, lock),
                              daemon=True, name=f"cam{c['camera']:02d}") for c in configs]
    cons = threading.Thread(target=consumer, args=(q, root, steps or [], stop, producers_done,
                                                   stats, glob, lock, keep_last, max_frames), daemon=True)
    cons.start()
    for t in prods:
        t.start()
    log.info("Capturando de %d fuente(s). Ctrl-C para detener.", len(prods))
    try:
        stop.wait(duration if duration else None)
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario")
    stop.set()
    for t in prods:
        t.join(timeout=10)
    producers_done.set()
    cons.join(timeout=120)

    log.info("---- resumen ----")
    for cam, st in stats.items():
        log.info("cam%02d: %s", cam, dict(st))
    if glob:
        log.info("global: %s", dict(glob))
    return stats


def build_steps(args, root: Path):
    steps = []
    if args.enhance:
        from modules.enhancement import METHODS, enhance_frame
        if args.enhance not in METHODS:
            raise SystemExit(f"--enhance '{args.enhance}' no existe. Disponibles: {list(METHODS)}")
        steps.append(lambda f: enhance_frame(f, root, args.enhance, args.scale))
    if args.plates:
        from modules.plates import PlateModels, process_frame
        try:
            models = PlateModels(args.plates)
        except ImportError as e:
            raise SystemExit(f"Falta una libreria ({e}). Instala:  pip install ultralytics easyocr")
        steps.append(lambda f: process_frame(f if "enhancement" in f else with_skipped_enhancement(f),
                                             models, root, args.detect_on))
    if args.push:
        if not args.plates:
            raise SystemExit("--push envia resultados de etapa 3: usa tambien --plates")
        from modules.publish import make_push_step
        steps.append(make_push_step(args.push, root))       # clave en la variable VIGIA_TOKEN
    return steps


def main():
    ap = argparse.ArgumentParser(description="Captura en vivo -> pipeline VIGIA")
    ap.add_argument("--config", default="config/sources.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--duration", type=float, default=0, help="segundos (0 = hasta Ctrl-C)")
    ap.add_argument("--max-frames", type=int, default=0, help="detenerse tras N fotogramas procesados")
    ap.add_argument("--keep-last", type=int, default=0, help="conservar solo los ultimos N por camara (0 = todos)")
    ap.add_argument("--queue-size", type=int, default=16)
    ap.add_argument("--enhance", default=None, help="metodo de mejora, ej. bicubic")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--plates", default=None, help="ruta a best.pt para detectar placas")
    ap.add_argument("--push", default=None, help="URL del panel (http://IP:PUERTO); clave en VIGIA_TOKEN")
    ap.add_argument("--detect-on", choices=["original", "both"], default="original")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(load_config(a.config), a.root, build_steps(a, Path(a.root)), a.duration, a.max_frames,
        a.keep_last, a.queue_size)


if __name__ == "__main__":
    main()
