"""Genera demo_data/: 2 fotogramas SINTETICOS (no son resultados del modelo) para que el
panel y el docker-compose muestren algo desde el primer minuto.
Ejecutar desde la raiz del repo:  python -m scripts.make_demo_data"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from contracts.schemas import Stage3Frame, Summary

OUT = Path("demo_data")
FONT = cv2.FONT_HERSHEY_SIMPLEX
rng = np.random.default_rng(7)


def build(idx, car, plate, text, color, blur):
    big = np.full((1440, 2560, 3), (95, 100, 105), np.uint8)
    cv2.rectangle(big, (0, 1000), (2560, 1440), (70, 70, 72), -1)
    x1, y1, x2, y2 = [v * 4 for v in car]
    cv2.rectangle(big, (x1, y1), (x2, y2), color, -1)
    px1, py1, px2, py2 = [v * 4 for v in plate]
    cv2.rectangle(big, (px1, py1), (px2, py2), (245, 245, 245), -1)
    cv2.rectangle(big, (px1, py1), (px2, py2), (20, 20, 20), 4)
    (tw, th), _ = cv2.getTextSize(text, FONT, 1, 3)
    s = min((px2 - px1 - 30) / tw, (py2 - py1 - 30) / th)
    cv2.putText(big, text, (px1 + 15, py2 - (py2 - py1 - int(th * s)) // 2), FONT, s, (20, 20, 20), int(3 * s) + 1)
    cv2.putText(big, "DEMO SINTETICO", (40, 80), FONT, 2.2, (0, 220, 255), 6)
    orig = cv2.resize(big, (640, 360), interpolation=cv2.INTER_AREA)
    k = blur * 2 + 1
    orig = cv2.GaussianBlur(orig, (k, k), blur)
    orig = np.clip(orig.astype(int) + rng.normal(0, 6, orig.shape), 0, 255).astype(np.uint8)
    (OUT / "orig/cam01").mkdir(parents=True, exist_ok=True)
    (OUT / "enh/cam01").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"orig/cam01/{idx:06d}.jpg"), orig, [cv2.IMWRITE_JPEG_QUALITY, 40])
    cv2.imwrite(str(OUT / f"enh/cam01/{idx:06d}.png"), big)
    return hashlib.sha256((OUT / f"orig/cam01/{idx:06d}.jpg").read_bytes()).hexdigest()


def frame(idx, ts, car, plate, text, color, blur, t_o, c_o, t_e, c_e, improved):
    sha = build(idx, car, plate, text, color, blur)
    d = {
        "frame_id": f"cam01_{ts.replace('-', '').replace(':', '')}_{idx:06d}",
        "camera_id": "cam_01", "source": {"type": "file", "name": "sintetico_demo"},
        "timestamp_utc": ts, "frame_index": idx, "width": 640, "height": 360, "fps_source": None,
        "image_path": f"data/orig/cam01/{idx:06d}.jpg", "sha256": sha,
        "enhancement": {"model": "demo_sintetico", "scale": 4, "enhanced_path": f"data/enh/cam01/{idx:06d}.png",
                        "width": 2560, "height": 1440, "processing_ms": None, "skipped_reason": None, "is_evidence": False},
        "analysis": {"detector": "demo_sintetico", "ocr": "demo_sintetico", "detect_ms": None, "ocr_ms": None,
                     "detections": [{"detection_id": f"cam01_{idx:06d}_d1", "class": "vehicle",
                                     "bbox_xyxy": list(map(float, car)), "confidence": 0.9, "track_id": None,
                                     "plate": {"bbox_xyxy": list(map(float, plate)), "text_original": t_o,
                                               "confidence_original": c_o, "text_enhanced": t_e,
                                               "confidence_enhanced": c_e, "readable_improved": improved}}]},
    }
    m = Stage3Frame.model_validate(d)
    (OUT / "json/cam01").mkdir(parents=True, exist_ok=True)
    (OUT / f"json/cam01/{idx:06d}.json").write_text(json.dumps(m.model_dump(by_alias=True), indent=2), encoding="utf-8")


frame(1, "2026-10-16T12:00:00Z", (120, 80, 310, 200), (180, 170, 250, 195), "ABC1234", (40, 60, 160), 3, None, 0.0, "ABC1234", 0.78, True)
frame(2, "2026-10-16T12:00:05Z", (330, 120, 520, 240), (385, 205, 455, 230), "XYZ9876", (50, 130, 60), 1, "XYZ9876", 0.55, "XYZ9876", 0.88, False)

(OUT / "summary").mkdir(exist_ok=True)
s = Summary.model_validate({
    "total_frames": 2, "plates_detected": 2, "plates_read_original": 1, "plates_read_enhanced": 2,
    "avg_confidence_original": 0.55, "avg_confidence_enhanced": 0.83, "avg_latency_ms_per_frame": None,
    "per_camera": [{"camera_id": "cam_01", "usable_without_replacement": None, "note": "dato sintetico de demostracion"}]})
(OUT / "summary/metricas.json").write_text(json.dumps(s.model_dump(), indent=2), encoding="utf-8")
print("demo_data/ generado")
