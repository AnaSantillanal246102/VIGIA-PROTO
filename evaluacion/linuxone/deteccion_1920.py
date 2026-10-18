"""
Detección de placas en IBM LinuxONE (s390x) SIN OpenCV, ultralytics ni torchvision.
Usa el modelo YOLO exportado a TorchScript (recomendado) o a ONNX.

Uso:
    python deteccion_s390x.py best.torchscript fotos/
"""

import glob
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

MODELO = sys.argv[1] if len(sys.argv) > 1 else "best.torchscript"
CARPETA = sys.argv[2] if len(sys.argv) > 2 else "fotos"
CONF_MIN = 0.5
IOU_MAX = 0.5
TAM = 1920
SALIDA = "resultados_1920"
os.makedirs(SALIDA, exist_ok=True)

print("Arquitectura:", platform.machine())
print("PyTorch:", torch.__version__)
t0 = time.perf_counter()
if MODELO.endswith(".onnx"):
    import onnx
    from onnx2torch import convert
    modelo = convert(onnx.load(MODELO)).eval()
else:
    modelo = torch.jit.load(MODELO, map_location="cpu").eval()
print(f"Modelo cargado en {(time.perf_counter() - t0) * 1000:.0f} ms")


def preparar(img):
    w, h = img.size
    r = min(TAM / w, TAM / h)
    nw, nh = round(w * r), round(h * r)
    dx, dy = (TAM - nw) // 2, (TAM - nh) // 2
    lienzo = Image.new("RGB", (TAM, TAM), (114, 114, 114))
    lienzo.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    x = torch.from_numpy(np.asarray(lienzo, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x, r, dx, dy


def nms(cajas, puntos, umbral):
    orden = puntos.argsort()[::-1]
    quedan = []
    while orden.size:
        i = orden[0]
        quedan.append(i)
        resto = orden[1:]
        x1 = np.maximum(cajas[i, 0], cajas[resto, 0])
        y1 = np.maximum(cajas[i, 1], cajas[resto, 1])
        x2 = np.minimum(cajas[i, 2], cajas[resto, 2])
        y2 = np.minimum(cajas[i, 3], cajas[resto, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (cajas[i, 2] - cajas[i, 0]) * (cajas[i, 3] - cajas[i, 1])
        area_r = (cajas[resto, 2] - cajas[resto, 0]) * (cajas[resto, 3] - cajas[resto, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        orden = resto[iou < umbral]
    return quedan


def detectar(img):
    x, r, dx, dy = preparar(img)
    with torch.no_grad():
        sal = modelo(x)
    if isinstance(sal, (list, tuple)):
        sal = sal[0]
    p = sal[0].numpy()
    if p.shape[0] > p.shape[1]:
        p = p.T
    conf = p[4:].max(0)
    m = conf >= CONF_MIN
    cx, cy, w, h, conf = p[0][m], p[1][m], p[2][m], p[3][m], conf[m]
    if conf.size == 0:
        return []
    cajas = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    cajas[:, [0, 2]] -= dx
    cajas[:, [1, 3]] -= dy
    cajas /= r
    cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(0, img.width)
    cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(0, img.height)
    return [([round(v, 1) for v in cajas[i].tolist()], round(float(conf[i]), 3)) for i in nms(cajas, conf, IOU_MAX)]


fotos = sorted(glob.glob(os.path.join(CARPETA, "*.jpg")) + glob.glob(os.path.join(CARPETA, "*.png")))
if not fotos:
    raise SystemExit(f"No hay fotos en {CARPETA}")

detectar(Image.open(fotos[0]).convert("RGB"))  # calentamiento, no se mide

tiempos, total, con_placa = [], 0, 0
with open(f"{SALIDA}/detecciones.jsonl", "w") as fh:
    for ruta in fotos:
        img = Image.open(ruta).convert("RGB")
        t = time.perf_counter()
        dets = detectar(img)
        ms = (time.perf_counter() - t) * 1000
        tiempos.append(ms)
        total += len(dets)
        con_placa += bool(dets)
        dibujo = ImageDraw.Draw(img)
        for caja, c in dets:
            dibujo.rectangle(caja, outline=(0, 255, 0), width=4)
            dibujo.text((caja[0], max(caja[1] - 14, 0)), f"placa {c:.2f}", fill=(0, 255, 0))
        img.save(f"{SALIDA}/{os.path.splitext(os.path.basename(ruta))[0]}_det.jpg")
        fh.write(json.dumps({"foto": os.path.basename(ruta), "ms": round(ms, 1),
                             "detecciones": [{"caja": c, "confianza": s} for c, s in dets]}) + "\n")
        print(f"{os.path.basename(ruta)}: {len(dets)} placa(s) en {ms:.0f} ms")

print("\n===== RESUMEN (IBM LinuxONE) =====")
print("Arquitectura:", platform.machine())
print("Fotos analizadas:", len(fotos))
print("Fotos con placa:", con_placa)
print("Placas detectadas:", total)
print(f"Tiempo promedio de detección: {np.mean(tiempos):.0f} ms por foto")
print(f"Resultados en {SALIDA}/")
