#!/usr/bin/env python3
"""
Valida JSON contra el contrato.

Uso:
  python validate.py 1 data/json/cam01/000123.json     # un archivo, etapa 1
  python validate.py 3 data/json/                       # todos los .json de una carpeta
  python validate.py summary data/summary/metricas.json

Sale con codigo 1 si algo falla (sirve en scripts y CI).
Desde Python:
  from contracts.schemas import Stage3Frame
  Stage3Frame.model_validate(mi_dict)     # lanza ValidationError si rompe el contrato
  Stage3Frame.model_validate(mi_dict).model_dump(by_alias=True)   # para guardar (usa "class")
"""
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from schemas import STAGES


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in STAGES:
        print(__doc__)
        return 2
    model = STAGES[sys.argv[1]]
    target = Path(sys.argv[2])
    files = sorted(target.rglob("*.json")) if target.is_dir() else [target]
    if not files:
        print("No se encontraron archivos .json")
        return 2

    bad = 0
    for f in files:
        try:
            model.model_validate(json.loads(f.read_text(encoding="utf-8")))
            print(f"OK    {f}")
        except (ValidationError, json.JSONDecodeError) as e:
            bad += 1
            print(f"FALLA {f}\n{e}\n")
    print(f"\n{len(files) - bad}/{len(files)} validos")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
