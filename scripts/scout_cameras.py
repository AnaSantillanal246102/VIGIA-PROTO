"""
scripts/scout_cameras.py
Explora los catalogos de Caltrans, baja UNA imagen por camara, mide resolucion /
nitidez / brillo y genera hojas de contacto (JPG) para elegir a ojo las camaras
con el angulo que te sirve. Escribe tambien scout.csv con el ranking.

Uso:
  python -m scripts.scout_cameras --districts 3 4 7 8 11 12 --per-district 80 --out data/scout
Luego abre data/scout/sheet_01.jpg ... y anota los IDs (ej. d07-123).
"""
from __future__ import annotations
import argparse, csv, io, logging, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.ingest import _get, _session, parse_caltrans

log = logging.getLogger("scout")
TILE_W, TILE_H, COLS, ROWS = 360, 220, 4, 4


def analyze(raw: bytes) -> dict:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    sharp, bright = float(lap.var()), float(g.mean())
    w, h = img.size
    score = sharp * min(w, 1920) / 640 if 35 < bright < 225 else 0.0   # descarta noche/sobreexpuesta
    return {"img": img, "w": w, "h": h, "sharp": round(sharp, 1), "bright": round(bright, 1), "score": round(score, 1)}


def make_sheet(items: list[dict], path: Path) -> None:
    sheet = Image.new("RGB", (COLS * TILE_W, ROWS * (TILE_H + 28)), "white")
    d = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        x, y = (i % COLS) * TILE_W, (i // COLS) * (TILE_H + 28)
        t = it["img"].copy(); t.thumbnail((TILE_W, TILE_H))
        sheet.paste(t, (x, y + 28))
        d.text((x + 4, y + 2), f'{it["id"]}  {it["w"]}x{it["h"]}  s={it["sharp"]}', fill="black")
        d.text((x + 4, y + 14), f'{it["route"]} {str(it["name"])[:40]}', fill="gray")
    sheet.save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--districts", nargs="+", type=int, default=[7])
    ap.add_argument("--per-district", type=int, default=80)
    ap.add_argument("--min-width", type=int, default=0)
    ap.add_argument("--out", default="data/scout")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sess, items = _session(), []
    for d in a.districts:
        url = f"https://cwwp2.dot.ca.gov/data/d{d}/cctv/cctvStatusD{d:02d}.json"
        try:
            cams = parse_caltrans(_get(sess, url).json(), a.per_district)
        except Exception as e:
            log.warning("d%02d: %s", d, e); continue
        log.info("d%02d: %d camaras", d, len(cams))
        for c in cams:
            try:
                r = analyze(_get(sess, c["url"]).content)
            except Exception as e:
                log.debug("%s: %s", c["url"], e); continue
            if r["w"] < a.min_width:
                continue
            items.append({**r, "id": f'd{d:02d}-{c["index"]}', "district": d, "index": c["index"],
                          "route": c["route"], "name": c["name"], "lat": c["lat"], "lon": c["lon"]})
            time.sleep(0.3)   # no saturar el servidor
    items.sort(key=lambda x: x["score"], reverse=True)
    with open(out / "scout.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "district", "index", "route", "name", "lat", "lon", "width", "height", "sharp", "bright", "score"])
        for it in items:
            wr.writerow([it["id"], it["district"], it["index"], it["route"], it["name"], it["lat"], it["lon"],
                         it["w"], it["h"], it["sharp"], it["bright"], it["score"]])
    per = COLS * ROWS
    for k in range(0, len(items), per):
        make_sheet(items[k:k + per], out / f"sheet_{k // per + 1:02d}.jpg")
    log.info("%d camaras evaluadas -> %s (abre las sheet_*.jpg)", len(items), out)


if __name__ == "__main__":
    main()
