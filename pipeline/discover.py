"""
Genera config/sources.local.json a partir de un catalogo publico de camaras.

  python -m pipeline.discover --provider caltrans --limit 10 --verify
  python -m pipeline.discover --provider caltrans --match "San Diego" --limit 5

Proveedores: caltrans (Caltrans CCTV, California; sin clave segun su documentacion).
--verify descarga 1 imagen de cada camara, descarta las que no responden y anota la resolucion
(util para elegir camaras de BAJA resolucion). Lee las condiciones de uso del proveedor antes
de usar los datos.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from pipeline.sources import UA, SnapshotSource

CALTRANS = "https://gisdata.dot.ca.gov/arcgis/rest/services/CHhighway/CCTV/FeatureServer/0/query"


def fetch_caltrans(n: int) -> list:
    q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "returnGeometry": "false",
                                "f": "json", "resultRecordCount": str(n)})
    req = urllib.request.Request(f"{CALTRANS}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data.get("features", [])


def build_entries(features, limit, match=None, prefer_stream=False, start_camera=1, interval_s=10):
    """Funcion pura (facil de probar): atributos ArcGIS -> entradas de configuracion."""
    out, cam = [], start_camera
    for f in features:
        a = {k.lower(): v for k, v in (f.get("attributes") or {}).items()}
        name = str(a.get("locationname") or "sin nombre")
        img = (a.get("currentimageurl") or "").strip()
        stream = (a.get("streamingvideourl") or "").strip()
        if match and match.lower() not in name.lower():
            continue
        if prefer_stream and stream:
            e = {"type": "hls", "url": stream, "interval_s": 5}
        elif img:
            e = {"type": "snapshot", "url": img, "interval_s": interval_s}
        elif stream:
            e = {"type": "hls", "url": stream, "interval_s": 5}
        else:
            continue
        out.append({"enabled": True, "camera": cam, "name": f"caltrans: {name}"[:80], **e})
        cam += 1
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="caltrans", choices=["caltrans"])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--match", default=None, help="filtrar por texto en el nombre de la ubicacion")
    ap.add_argument("--prefer-stream", action="store_true")
    ap.add_argument("--start-camera", type=int, default=1)
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="config/sources.local.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(f"{out} ya existe (usa --force para sobrescribir)")
    try:
        feats = fetch_caltrans(2000)
    except Exception as e:
        sys.exit(f"No se pudo consultar el catalogo ({e}). Revisa tu conexion o que el servicio siga activo.")
    entries = build_entries(feats, a.limit * (5 if a.verify else 1), a.match, a.prefer_stream,
                            a.start_camera, a.interval)
    if a.verify:
        ok = []
        for e in entries:
            if e["type"] != "snapshot":
                ok.append(e)
                continue
            try:
                img = SnapshotSource(e["name"], e["url"]).fetch_once()
                e["resolucion_observada"] = f"{img.shape[1]}x{img.shape[0]}"
                ok.append(e)
                print(f"OK    cam{e['camera']:02d} {e['resolucion_observada']:>9}  {e['name']}")
            except Exception as ex:
                print(f"FALLA cam{e['camera']:02d} {e['name']}: {ex}")
            if len(ok) >= a.limit:
                break
        entries = ok[:a.limit]
        for i, e in enumerate(entries, a.start_camera):       # renumerar tras descartar
            e["camera"] = i
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sources": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(entries)} camaras escritas en {out}")


if __name__ == "__main__":
    main()
