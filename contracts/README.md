# contracts/ — Contratos de datos (v1.0)

```
Pipeline (T1) → Stage1 → Módulo 1 (T2) → Stage2 → Módulo 2 (T3) → Stage3 → Panel (T1)
```

Cada etapa **copia todo lo anterior sin cambios** y agrega su bloque
(`enhancement`, luego `analysis`). Ejemplos reales de formato en `examples/`.

## Instalar y validar
```bash
pip install -r contracts/requirements.txt
cd contracts
python validate.py 1 examples/stage1_input.json
python validate.py 2 examples/stage2_enhanced.json
python validate.py 3 examples/stage3_output.json
python validate.py summary examples/summary.json
```

## Usar en tu código
```python
from contracts.schemas import Stage2Frame
frame = Stage2Frame.model_validate(datos_dict)           # falla si rompes el contrato
json_str = frame.model_dump_json(by_alias=True, indent=2)  # by_alias → escribe "class"
```

## Reglas (el validador las revisa)
- `bbox_xyxy = [x1,y1,x2,y2]` en **píxeles de la imagen ORIGINAL**. Si detectas sobre la imagen
  mejorada (×4), divide entre `scale`. El validador rechaza cajas fuera de `width × height`.
- Desconocido = `null`. Confianzas 0.0–1.0. Fechas UTC con `Z`.
- `enhancement.is_evidence` siempre `false`.
- Si `enhanced_path` es `null`, `skipped_reason` es obligatorio.
- No se permiten campos nuevos sin avisar: se sube `schema_version`.
