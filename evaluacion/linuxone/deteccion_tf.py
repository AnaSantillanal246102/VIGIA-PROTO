"""
Detección de placas en IBM LinuxONE con TensorFlow (para aprovechar ibm-zdnn-plugin / acelerador Telum).
Usa el modelo YOLO exportado con: YOLO('best.pt').export(format='saved_model', imgsz=1920)

Uso:
    python deteccion_tf.py best_saved_model fotos_prueba
"""

import glob
import json
import os
import platform
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

MODELO = sys.argv[1] if len(sys.argv) > 1 else "best_saved_model"
CARPETA = sys.argv[2] if len(sys.argv) > 2 else "fotos_prueba"
CONF_MIN = 0.5
IOU_MAX = 0.5
TAM = 1920
SALIDA = "resultados_tf"
os.makedirs(SALIDA, exist_ok=True)

print("Arquitectura:", platform.machine())
print("TensorFlow:", tf.__version__)
print("Dispositivos:", tf.config.list_physical_devices())
t0 = time.perf_counter()
cargado = tf.saved_model.load(MODELO)
funcion = cargado.signatures["serving_default"]
nombre_entrada = list(funcion.structured_input_signature[1].keys())[0]
print(f"Modelo cargado en {(time.perf_counter() - t0) * 1000:.0f} ms (entrada: {nombre_entrada})")


def preparar(img):
    w, h = img.size
    r = min(TAM / w, TAM / h)
    nw, nh = round(w * r), round(h * r)
    dx, dy = (TAM - nw) // 2, (TAM - nh) // 2
    lienzo = Image.new("RGB", (TAM, TAM), (114, 114, 114))
    lienzo.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    x = (np.asarray(lienzo, dtype=np.float32) / 255.0)[None]  # NHWC
    return tf.constant(x), r, dx, dy


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
    sal = funcion(**{nombre_entrada: x})
    p = list(sal.values())[0].numpy()[0]
    if p.shape[0] > p.shape[1]:
        p = p.T
    if p[:4].max() <= 1.5:  # algunas exportaciones de TF dan coordenadas normalizadas (0 a 1)
        p[:4] *= TAM
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

print("\n===== RESUMEN (IBM LinuxONE + TensorFlow) =====")
print("Arquitectura:", platform.machine())
print("Fotos analizadas:", len(fotos))
print("Fotos con placa:", con_placa)
print("Placas detectadas:", total)
print(f"Tiempo promedio de detección: {np.mean(tiempos):.0f} ms por foto")
print(f"Resultados en {SALIDA}/")
