"""Seleccion de fotogramas: decide cuales vale la pena procesar."""
import cv2
import numpy as np


class FrameSelector:
    """
    Rechaza fotogramas:
      - "borrosa"    : varianza del Laplaciano < min_sharpness (imagen sin detalle / fuera de foco)
      - "sin_cambio" : casi identico al ultimo aceptado (camaras tipo snapshot repiten la imagen)
                       salvo que pasen force_every_s segundos (latido para no quedarse mudo).
    Poner min_sharpness=0 o min_change=0 desactiva el criterio. Escenas nocturnas o muy
    uniformes tienen poca varianza: ajustar por camara en la configuracion.
    """

    def __init__(self, min_sharpness: float = 10.0, min_change: float = 1.5,
                 force_every_s: float = 60.0):
        self.min_sharpness = float(min_sharpness)
        self.min_change = float(min_change)
        self.force_every_s = float(force_every_s)
        self._last_thumb = None
        self._last_accept = float("-inf")

    def accept(self, img: np.ndarray, now: float):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.min_sharpness > 0 and cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_sharpness:
            return False, "borrosa"
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._last_thumb is not None and self.min_change > 0:
            diff = float(np.abs(thumb - self._last_thumb).mean())
            if diff < self.min_change and (now - self._last_accept) < self.force_every_s:
                return False, "sin_cambio"
        self._last_thumb, self._last_accept = thumb, now
        return True, "ok"
