"""
Fuentes de video. Cada fuente expone  frames(stop_event) -> genera (imagen_BGR, fps|None).

  snapshot : URL que devuelve un JPG/PNG (lo mas comun en camaras publicas de transito).
             Se consulta cada interval_s segundos.
  hls/rtsp : flujo de video; se lee continuamente y se toma 1 fotograma cada interval_s.
  file     : archivo de video, simulando tiempo real (para ensayos y demos sin internet).

Claves de API: NUNCA en el JSON de configuracion. Usar ${NOMBRE_VARIABLE} en url/headers.
"""
import logging
import os
import re
import time
import urllib.request

import cv2
import numpy as np

log = logging.getLogger("vigia.sources")

UA = "VIGIA-prototipo/0.1 (proyecto academico de investigacion)"
MIN_SNAPSHOT_INTERVAL_S = 2.0      # piso de cortesia: no saturar servidores publicos
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(s: str) -> str:
    def rep(m):
        v = os.environ.get(m.group(1))
        if v is None:
            raise KeyError(f"Falta la variable de entorno {m.group(1)}")
        return v
    return _ENV.sub(rep, s)


class SnapshotSource:
    def __init__(self, name, url, interval_s=10.0, headers=None, timeout=10.0,
                 max_bytes=5_000_000, **_):
        self.name = name
        self.url = expand_env(url)
        self.interval = max(float(interval_s), MIN_SNAPSHOT_INTERVAL_S)
        self.headers = {k: expand_env(v) for k, v in (headers or {}).items()}
        self.timeout, self.max_bytes = timeout, max_bytes

    def fetch_once(self) -> np.ndarray:
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, **self.headers})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            ctype = r.headers.get_content_type()
            if not ctype.startswith("image/"):
                raise ValueError(f"tipo de contenido inesperado: {ctype}")
            data = r.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("imagen demasiado grande")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("no se pudo decodificar la imagen")
        return img

    def frames(self, stop):
        fails = 0
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                img, fails = self.fetch_once(), 0
            except Exception as e:                      # red caida, 401/404, HTML en vez de imagen...
                img, fails = None, fails + 1
                log.warning("[%s] %s (fallo %d)", self.name, e, fails)
            if img is not None:
                yield img, None
            wait = self.interval if fails == 0 else min(60.0, self.interval * 2 ** min(fails, 5))
            stop.wait(max(0.0, wait - (time.monotonic() - t0)))


class StreamSource:
    def __init__(self, name, url, interval_s=5.0, **_):
        self.name, self.url, self.interval = name, expand_env(url), float(interval_s)

    def frames(self, stop):
        fails = 0
        while not stop.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                fails += 1
                log.warning("[%s] no se pudo abrir el flujo (fallo %d)", self.name, fails)
                cap.release()
                stop.wait(min(60.0, 2.0 ** min(fails, 6)))
                continue
            fails = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            fps = fps if fps and fps < 120 else None
            last = 0.0
            try:
                while not stop.is_set():
                    if not cap.grab():                  # flujo cortado -> reconectar
                        break
                    now = time.monotonic()
                    if now - last >= self.interval:     # decodificar solo cuando toca
                        ok, img = cap.retrieve()
                        if ok:
                            last = now
                            yield img, fps
            finally:
                cap.release()
            log.warning("[%s] flujo interrumpido; reconectando", self.name)
            stop.wait(2.0)


class FileSource:
    def __init__(self, name, url, interval_s=1.0, speed=1.0, loop=False, **_):
        self.name, self.url = name, url
        self.interval, self.speed, self.loop = float(interval_s), float(speed), bool(loop)

    def frames(self, stop):
        while not stop.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                raise FileNotFoundError(f"No se pudo abrir {self.url}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step, i = max(1, round(self.interval * fps)), 0
            try:
                while not stop.is_set():
                    ok, img = cap.read()
                    if not ok:
                        break
                    if i % step == 0:
                        yield img, fps
                        if self.speed > 0:              # speed=0 -> lo mas rapido posible
                            stop.wait(self.interval / self.speed)
                    i += 1
            finally:
                cap.release()
            if not self.loop:
                return


def build_source(cfg: dict):
    t = cfg["type"]
    if t == "snapshot":
        return SnapshotSource(**cfg)
    if t in ("hls", "rtsp"):
        return StreamSource(**cfg)
    if t == "file":
        return FileSource(**cfg)
    raise ValueError(f"tipo de fuente desconocido: {t}")
