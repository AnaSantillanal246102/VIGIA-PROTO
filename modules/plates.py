"""
Modulo 2 (Team 3): deteccion y lectura de matriculas.

  models = PlateModels("runs/detect/placas/weights/best.pt")     # ingesta del modelo
  s3 = process_frame(stage2_dict, models, root=".")              # Stage2 -> Stage3

Entrada : JSON etapa 2 (contracts.schemas.Stage2Frame)
Salida  : JSON etapa 3 (contracts.schemas.Stage3Frame), validado antes de devolverlo.

Estrategia:
  1. YOLO (tu modelo entrenado) detecta placas en la imagen ORIGINAL.
  2. Se recorta cada placa de la original y de la mejorada (mismo lugar, escalado).
  3. OCR sobre ambos recortes -> text_original vs text_enhanced (la cifra de la demo).
  4. Todas las cajas se guardan en pixeles de la imagen ORIGINAL.

Uso desde terminal (desde la raiz del repo):
  python -m modules.plates --weights best.pt --in-dir data/json --root .
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from contracts.schemas import Stage2Frame, Stage3Frame

# ---- Parametros ajustables -------------------------------------------------
MIN_PLATE_LEN = 5        # largo minimo de una lectura valida (placas MX: 6-7 caracteres)
MAX_PLATE_LEN = 8
MIN_OCR_CONF = 0.30      # por debajo de esto la placa cuenta como "no legible"
OCR_MIN_HEIGHT = 64      # el recorte se amplia a esta altura antes del OCR
BOX_PAD = 0.08           # margen (fraccion del ancho/alto) alrededor de la caja
NMS_IOU = 0.5
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

Box = Tuple[float, float, float, float]            # x1, y1, x2, y2
DetBox = Tuple[float, float, float, float, float]  # + confianza


# ---------------------------------------------------------------------------
# 1) INGESTA DEL MODELO
# ---------------------------------------------------------------------------
class PlateModels:
    """Carga el detector YOLO entrenado y el OCR una sola vez (reutilizar entre fotogramas)."""

    def __init__(self, weights: str = "runs/detect/placas/weights/best.pt",
                 device: Optional[str] = None, conf: float = 0.25,
                 imgsz: int = 640, imgsz_enhanced: int = 1280,
                 ocr_langs: Tuple[str, ...] = ("en",)):
        from ultralytics import YOLO   # imports pesados aqui, no al importar el modulo
        import easyocr
        import torch

        w = Path(weights)
        if not w.exists():
            raise FileNotFoundError(
                f"No existe {w}. Tras modelo.train(...), los pesos quedan en "
                f"runs/detect/<name>/weights/best.pt (si repites el nombre, Ultralytics crea "
                f"placas2, placas3...). Revisa la carpeta runs/detect/.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conf, self.imgsz, self.imgsz_enh = conf, imgsz, imgsz_enhanced
        self.yolo = YOLO(str(w))
        self.reader = easyocr.Reader(list(ocr_langs), gpu=self.device.startswith("cuda"))
        self.detector_name = f"yolov8n_placas:{w.name}"
        self.ocr_name = "easyocr"
        # calentamiento: la primera inferencia es lenta y distorsionaria los ms medidos
        self.yolo.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz,
                          device=self.device, verbose=False)

    def detect(self, img: np.ndarray, imgsz: int) -> List[DetBox]:
        """Devuelve [(x1,y1,x2,y2,conf)] en pixeles de la imagen recibida."""
        r = self.yolo.predict(img, imgsz=imgsz, conf=self.conf,
                              device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        return [(float(a), float(b), float(c), float(d), float(p))
                for (a, b, c, d), p in zip(xyxy, conf)]

    def read(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """OCR de un recorte de placa -> (texto crudo, confianza media 0-1)."""
        if crop is None or crop.size == 0:
            return None, 0.0
        h = crop.shape[0]
        if h < OCR_MIN_HEIGHT:
            f = OCR_MIN_HEIGHT / h
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        res = self.reader.readtext(gray, allowlist=ALLOWLIST, detail=1, paragraph=False)
        if not res:
            return None, 0.0
        res.sort(key=lambda r: min(p[0] for p in r[0]))      # izquierda -> derecha
        text = "".join(r[1] for r in res)
        return text, float(np.mean([r[2] for r in res]))


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def normalize_plate(text: Optional[str]) -> Optional[str]:
    """Mayusculas, solo A-Z y 0-9, y largo plausible. Si no cumple -> None."""
    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    return t if MIN_PLATE_LEN <= len(t) <= MAX_PLATE_LEN else None


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(boxes: List[DetBox], thr: float = NMS_IOU) -> List[DetBox]:
    keep: List[DetBox] = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(_iou(b[:4], k[:4]) < thr for k in keep):
            keep.append(b)
    return keep


def _clamp(b: Box, W: int, H: int) -> Optional[Box]:
    x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])
    return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def _crop(img: np.ndarray, box_orig: Box, sx: float, sy: float) -> np.ndarray:
    """Recorta la caja (en coords originales) sobre una imagen a escala sx, sy, con margen."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_orig[0] * sx, box_orig[1] * sy, box_orig[2] * sx, box_orig[3] * sy
    px, py = (x2 - x1) * BOX_PAD, (y2 - y1) * BOX_PAD
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return img[y1:y2, x1:x2]


def _read_clean(models, crop) -> Tuple[Optional[str], float]:
    text, conf = models.read(crop)
    text = normalize_plate(text)
    if text is None or conf < MIN_OCR_CONF:
        return None, 0.0
    return text, round(conf, 4)


# ---------------------------------------------------------------------------
# 2) FUNCION QUE ACOMPANA EL FOTOGRAMA CON LA MATRICULA
# ---------------------------------------------------------------------------
def process_frame(frame: dict, models: PlateModels, root: str | Path = ".",
                  detect_on: str = "original") -> dict:
    """
    frame     : dict con el JSON de etapa 2.
    detect_on : "original" (recomendado: es la escala con la que se entreno el modelo)
                "both"     (ademas detecta en la mejorada y fusiona; recupera placas que
                            en la original son demasiado pequenas. Evaluarlo con metricas.)
    Devuelve  : dict de etapa 3 (validado contra Stage3Frame).
    """
    s2 = Stage2Frame.model_validate(frame)          # falla de inmediato si la entrada rompe el contrato
    root = Path(root)

    orig = cv2.imread(str(root / s2.image_path))
    if orig is None:
        raise FileNotFoundError(f"No se pudo leer {root / s2.image_path}")
    H, W = orig.shape[:2]
    if (W, H) != (s2.width, s2.height):
        raise ValueError(f"{s2.frame_id}: la imagen mide {W}x{H} pero el JSON dice "
                         f"{s2.width}x{s2.height}")

    enh = None
    if s2.enhancement.enhanced_path:
        enh = cv2.imread(str(root / s2.enhancement.enhanced_path))
        if enh is None:
            raise FileNotFoundError(f"No se pudo leer {root / s2.enhancement.enhanced_path}")
    esx = enh.shape[1] / W if enh is not None else 1.0
    esy = enh.shape[0] / H if enh is not None else 1.0

    # --- deteccion ---
    t0 = time.perf_counter()
    boxes = list(models.detect(orig, models.imgsz))
    if detect_on == "both" and enh is not None:
        # cajas de la mejorada -> pixeles ORIGINALES (regla del contrato)
        boxes += [(x1 / esx, y1 / esy, x2 / esx, y2 / esy, c)
                  for x1, y1, x2, y2, c in models.detect(enh, models.imgsz_enh)]
    boxes = _nms(boxes)
    detect_ms = (time.perf_counter() - t0) * 1000

    # --- OCR sobre original y mejorada ---
    t1 = time.perf_counter()
    cam_tag, _, idx = s2.frame_id.split("_")
    detections = []
    n = 0
    for x1, y1, x2, y2, conf in boxes:
        b = _clamp((x1, y1, x2, y2), W, H)
        if b is None:
            continue
        n += 1
        t_o, c_o = _read_clean(models, _crop(orig, b, 1.0, 1.0))
        t_e, c_e = (None, 0.0)
        if enh is not None:
            t_e, c_e = _read_clean(models, _crop(enh, b, esx, esy))
        improved = t_e is not None and (t_o is None or (t_e != t_o and c_e > c_o))
        box = [round(v, 1) for v in b]
        detections.append({
            "detection_id": f"{cam_tag}_{idx}_d{n}",
            "class": "plate_only",          # el modelo detecta placas, no vehiculos completos
            "bbox_xyxy": box,
            "confidence": round(conf, 4),
            "track_id": None,
            "plate": {
                "bbox_xyxy": box,
                "text_original": t_o, "confidence_original": c_o,
                "text_enhanced": t_e, "confidence_enhanced": c_e,
                "readable_improved": improved,
            },
        })
    ocr_ms = (time.perf_counter() - t1) * 1000

    out = s2.model_dump(by_alias=True)
    out["analysis"] = {
        "detector": models.detector_name,
        "ocr": models.ocr_name,
        "detect_ms": round(detect_ms, 1),
        "ocr_ms": round(ocr_ms, 1),
        "detections": detections,
    }
    return Stage3Frame.model_validate(out).model_dump(by_alias=True)   # valida salida


def process_json(in_path: Path, out_path: Path, models, root=".", detect_on="original"):
    data = json.loads(in_path.read_text(encoding="utf-8"))
    res = process_frame(data, models, root, detect_on)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# CLI: procesa una carpeta de JSON de etapa 2
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Etapa 2 -> Etapa 3 (matriculas)")
    ap.add_argument("--weights", default="runs/detect/placas/weights/best.pt")
    ap.add_argument("--in-dir", default="data/json")
    ap.add_argument("--out-dir", default=None, help="por defecto = in-dir (se completa en sitio)")
    ap.add_argument("--root", default=".", help="carpeta base de las rutas image_path")
    ap.add_argument("--detect-on", choices=["original", "both"], default="original")
    ap.add_argument("--force", action="store_true", help="reprocesar JSON que ya tienen 'analysis'")
    a = ap.parse_args()

    in_dir = Path(a.in_dir)
    out_dir = Path(a.out_dir) if a.out_dir else in_dir
    models = PlateModels(a.weights)
    ok = skipped = failed = 0
    for f in sorted(in_dir.rglob("*.json")):
        try:
            if not a.force and "analysis" in json.loads(f.read_text(encoding="utf-8")):
                skipped += 1
                continue
            process_json(f, out_dir / f.relative_to(in_dir), models, a.root, a.detect_on)
            ok += 1
        except Exception as e:       # un fotograma malo no detiene el lote
            failed += 1
            print(f"FALLA {f}: {e}")
    print(f"listo: {ok} procesados, {skipped} omitidos, {failed} con error")


if __name__ == "__main__":
    main()
