import json, glob, csv, statistics as st

archivos = sorted(glob.glob("data/json/**/*.json", recursive=True))
filas, det_ms, ocr_ms, enh_ms, fotos_con = [], [], [], [], 0

for f in archivos:
    d = json.load(open(f))
    a, e = d.get("analysis") or {}, d.get("enhancement") or {}
    if e.get("processing_ms") is not None: enh_ms.append(e["processing_ms"])
    if a.get("detect_ms") is not None: det_ms.append(a["detect_ms"])
    if a.get("ocr_ms") is not None: ocr_ms.append(a["ocr_ms"])
    dets = a.get("detections") or []
    if dets: fotos_con += 1
    for x in dets:
        p = x.get("plate") or {}
        b = x["bbox_xyxy"]; w, h = b[2] - b[0], b[3] - b[1]
        filas.append({"foto": f.split("/")[-1], "confianza": x.get("confidence"),
                      "ancho": round(w), "alto": round(h),
                      "proporcion": round(w / h, 2) if h else None,
                      "texto_original": p.get("text_original"),
                      "texto_mejorado": p.get("text_enhanced"),
                      "mejoro": p.get("readable_improved")})

print("Fotos analizadas:", len(archivos))
print("Fotos con al menos una detección:", fotos_con)
print("Detecciones totales:", len(filas))
print("Confianza >= 0.5:", sum(1 for r in filas if (r["confianza"] or 0) >= 0.5))
print("Forma de placa (proporción 2 a 6):", sum(1 for r in filas if r["proporcion"] and 2 <= r["proporcion"] <= 6))
print("Con texto leído (original):", sum(1 for r in filas if r["texto_original"]))
print("Con texto leído (mejorado):", sum(1 for r in filas if r["texto_mejorado"]))
print("La mejora ayudó en:", sum(1 for r in filas if r["mejoro"]))
if enh_ms: print("Tiempo mejora (promedio ms):", round(st.mean(enh_ms), 1))
if det_ms: print("Tiempo detección (promedio ms):", round(st.mean(det_ms), 1))
if ocr_ms: print("Tiempo lectura (promedio ms):", round(st.mean(ocr_ms), 1))

if filas:
    with open("evaluacion/resultados/detecciones.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=filas[0].keys()); w.writeheader(); w.writerows(filas)
    print("\nPrimeras 15 detecciones:")
    for r in filas[:15]: print(r)
