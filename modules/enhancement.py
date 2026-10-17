"""
Modulo 1 (Team 2): mejora de resolucion.  ETAPA 1 -> ETAPA 2.

  python -m modules.enhancement --in-dir data/json --root . --method bicubic

Metodos disponibles: "bicubic" (LINEA BASE: solo amplia, no inventa detalle).
Para agregar Real-ESRGAN u otro: escribir una funcion  f(img_bgr, scale) -> img_bgr
y registrarla en METHODS. Todo lo demas (rutas, JSON, validacion) ya esta resuelto.
La linea base sirve para dos cosas: que el flujo completo corra desde hoy, y como
comparacion honesta (la mejora de la IA debe superar a un simple bicubico).
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from contracts.schemas import Stage1Frame, Stage2Frame


def _bicubic(img, scale):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


METHODS = {"bicubic": _bicubic}      # "realesrgan": mi_funcion_realesrgan, ...


def enhance_frame(frame: dict, root=".", method="bicubic", scale=4) -> dict:
    s1 = Stage1Frame.model_validate({k: v for k, v in frame.items() if k != "enhancement"})
    root = Path(root)
    img = cv2.imread(str(root / s1.image_path))
    if img is None:
        raise FileNotFoundError(root / s1.image_path)

    t = time.perf_counter()
    out = METHODS[method](img, scale)
    ms = (time.perf_counter() - t) * 1000

    rel = Path(s1.image_path.replace("data/orig/", "data/enh/", 1)).with_suffix(".png")
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / rel), out)

    d = s1.model_dump(by_alias=True)
    d["enhancement"] = {
        "model": f"{method}_x{scale}", "scale": scale, "enhanced_path": rel.as_posix(),
        "width": out.shape[1], "height": out.shape[0], "processing_ms": round(ms, 1),
        "skipped_reason": None, "is_evidence": False,
    }
    return Stage2Frame.model_validate(d).model_dump(by_alias=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--method", default="bicubic", choices=list(METHODS))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ok = skip = bad = 0
    for f in sorted(Path(a.in_dir).rglob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if "enhancement" in d and not a.force:
                skip += 1
                continue
            f.write_text(json.dumps(enhance_frame(d, a.root, a.method, a.scale), indent=2), encoding="utf-8")
            ok += 1
        except Exception as e:
            bad += 1
            print(f"FALLA {f}: {e}")
    print(f"listo: {ok} procesados, {skip} omitidos, {bad} con error")


if __name__ == "__main__":
    main()
