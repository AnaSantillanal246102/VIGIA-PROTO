"""
Mide cuánto recupera la mejora de resolución la lectura de placas.

Para cada par HR/LR (de capturar_live.py):
  1. Busca placas en la imagen HR y las lee -> esa lectura es la "respuesta de referencia".
  2. Recorta la misma placa en LR y la lee de 3 formas:
       - LR tal cual (agrandada sin mejorar, solo para que el OCR la pueda procesar)
       - LR mejorada con bicúbica x4
       - LR mejorada con IA (FSRCNN x4, modelo ya entrenado), si está disponible
  3. Compara cada lectura contra la de HR.

Uso (desde la raíz del repo):
    python evaluacion/comparar_resolucion.py
"""

import csv
import glob
import os
import re
from difflib import SequenceMatcher

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

CARPETA = "dataset_puentes"
MODELO_YOLO = "best.pt"
MODELO_SR = "FSRCNN_x4.pb"
ESCALA = 4
CONF_MIN = 0.5
SALIDA = "evaluacion/resultados"
LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

os.makedirs(SALIDA, exist_ok=True)
yolo = YOLO(MODELO_YOLO)
lector = easyocr.Reader(["en"], gpu=False)

sr = None
if os.path.exists(MODELO_SR):
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(MODELO_SR)
        sr.setModel("fsrcnn", ESCALA)
        print("Usando mejora con IA: FSRCNN x4")
    except AttributeError:
        print("Falta opencv-contrib-python; solo se comparará con bicúbica.")
        sr = None
else:
    print(f"No se encontró {MODELO_SR}; solo se comparará con bicúbica.")


def leer(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = lector.readtext(gris, allowlist=LETRAS)
    if not res:
        return ""
    alturas = [max(p[1] for p in r[0]) - min(p[1] for p in r[0]) for r in res]
    grandes = [r for r, h in zip(res, alturas) if h >= 0.6 * max(alturas)]
    grandes.sort(key=lambda r: min(p[0] for p in r[0]))
    return re.sub(r"[^A-Z0-9]", "", "".join(r[1] for r in grandes).upper())


def parecido(a, b):
    return SequenceMatcher(None, a, b).ratio() if a or b else 0.0


filas = []
pares = sorted(glob.glob(f"{CARPETA}/HR/*.png"))
print(f"Pares encontrados: {len(pares)}")

for ruta_hr in pares:
    nombre = os.path.basename(ruta_hr)
    hr = cv2.imread(ruta_hr)
    lr = cv2.imread(f"{CARPETA}/LR/{nombre}")
    if hr is None or lr is None:
        continue
    det = yolo.predict(hr, conf=CONF_MIN, verbose=False)[0]
    for caja in det.boxes.xyxy.tolist():
        x1, y1, x2, y2 = (int(v) for v in caja)
        rec_hr = hr[y1:y2, x1:x2]
        texto_hr = leer(rec_hr)
        if len(texto_hr) < 4:
            continue
        lx1, ly1, lx2, ly2 = x1 // ESCALA, y1 // ESCALA, max(x2 // ESCALA, x1 // ESCALA + 1), max(y2 // ESCALA, y1 // ESCALA + 1)
        rec_lr = lr[ly1:ly2, lx1:lx2]
        if rec_lr.size == 0:
            continue
        tam = (rec_hr.shape[1], rec_hr.shape[0])
        lr_sin = cv2.resize(rec_lr, tam, interpolation=cv2.INTER_NEAREST)
        lr_bic = cv2.resize(rec_lr, tam, interpolation=cv2.INTER_CUBIC)
        fila = {"imagen": nombre, "texto_hr": texto_hr,
                "texto_lr": leer(lr_sin), "texto_bicubica": leer(lr_bic)}
        if sr is not None:
            fila["texto_ia"] = leer(cv2.resize(sr.upsample(rec_lr), tam))
        filas.append(fila)

if not filas:
    raise SystemExit("No se encontraron placas legibles en HR. Prueba con más pares o con CONF_MIN más bajo.")

metodos = ["texto_lr", "texto_bicubica"] + (["texto_ia"] if sr is not None else [])
nombres = {"texto_lr": "Baja resolución", "texto_bicubica": "Mejora bicúbica", "texto_ia": "Mejora con IA (FSRCNN)"}

print(f"\nPlacas comparadas (legibles en HR): {len(filas)}\n")
print(f"{'Método':<26}{'Lectura exacta':>16}{'Similitud':>12}")
resumen = []
for m in metodos:
    exacta = np.mean([f[m] == f["texto_hr"] for f in filas]) * 100
    sim = np.mean([parecido(f[m], f["texto_hr"]) for f in filas]) * 100
    resumen.append({"metodo": nombres[m], "lectura_exacta_pct": round(exacta, 1), "similitud_pct": round(sim, 1)})
    print(f"{nombres[m]:<26}{exacta:>15.1f}%{sim:>11.1f}%")

with open(f"{SALIDA}/comparacion_resolucion_detalle.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=filas[0].keys())
    w.writeheader()
    w.writerows(filas)
with open(f"{SALIDA}/comparacion_resolucion_resumen.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=resumen[0].keys())
    w.writeheader()
    w.writerows(resumen)
print(f"\nGuardado en {SALIDA}/comparacion_resolucion_*.csv")
