"""
Contratos de datos del proyecto VIGIA  (schema_version 1.0)

Flujo:  Pipeline (Team 1) -> Stage1Frame
        Modulo 1  (Team 2) -> Stage2Frame  (= Stage1 + enhancement)
        Modulo 2  (Team 3) -> Stage3Frame  (= Stage2 + analysis)
        Panel     (Team 1) -> lee Stage3Frame y Summary

Reglas:
  * Cada etapa conserva TODO lo anterior sin modificarlo y agrega su bloque.
  * bbox_xyxy = [x1, y1, x2, y2] en PIXELES DE LA IMAGEN ORIGINAL.
  * Valores desconocidos = null (nunca "" ni "N/A").
  * Confianzas entre 0.0 y 1.0. Fechas en UTC (ISO 8601, terminan en Z).
  * Se prohiben campos no definidos (extra="forbid"): si necesitas uno nuevo,
    avisa al equipo y se sube la version del esquema.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

FRAME_ID_RE = r"^cam\d{2,}_\d{8}T\d{6}Z_\d{6}$"      # cam01_20261016T143210Z_000123
UTC_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"     # 2026-10-16T14:32:10Z
SHA256_RE = r"^[0-9a-f]{64}$"

Confidence = Field(ge=0.0, le=1.0)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Etapa 1: Pipeline -> Modulo 1 ----------
class Source(Strict):
    type: Literal["api", "hls", "rtsp", "file"]
    name: str


class Stage1Frame(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    frame_id: str = Field(pattern=FRAME_ID_RE)
    camera_id: str
    source: Source
    timestamp_utc: str = Field(pattern=UTC_RE)
    frame_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps_source: Optional[float] = Field(default=None, gt=0)
    image_path: str
    sha256: str = Field(pattern=SHA256_RE)


# ---------- Etapa 2: Modulo 1 (mejora) ----------
class Enhancement(Strict):
    model: str
    scale: Optional[int] = Field(default=None, ge=1)
    enhanced_path: Optional[str] = None
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    processing_ms: Optional[float] = Field(default=None, ge=0)
    skipped_reason: Optional[str] = None
    is_evidence: Literal[False] = False   # la imagen mejorada NUNCA es evidencia

    @model_validator(mode="after")
    def _coherencia(self):
        if self.enhanced_path is None and not self.skipped_reason:
            raise ValueError("Si enhanced_path es null, skipped_reason es obligatorio")
        if self.enhanced_path is not None and (self.width is None or self.height is None):
            raise ValueError("Si hay enhanced_path, width y height son obligatorios")
        return self


class Stage2Frame(Stage1Frame):
    enhancement: Enhancement


# ---------- Etapa 3: Modulo 2 (deteccion + matriculas) ----------
BBox = List[float]  # [x1, y1, x2, y2] en pixeles de la imagen ORIGINAL


def _check_bbox(b: BBox, name: str = "bbox_xyxy"):
    if len(b) != 4:
        raise ValueError(f"{name} debe tener 4 valores [x1,y1,x2,y2]")
    x1, y1, x2, y2 = b
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"{name} invalido: se espera x2>x1 y y2>y1, llego {b}")


class Plate(Strict):
    bbox_xyxy: BBox
    text_original: Optional[str] = None
    confidence_original: float = Confidence
    text_enhanced: Optional[str] = None
    confidence_enhanced: float = Confidence
    readable_improved: bool

    @model_validator(mode="after")
    def _bbox(self):
        _check_bbox(self.bbox_xyxy, "plate.bbox_xyxy")
        return self


class Detection(Strict):
    detection_id: str
    cls: Literal["vehicle", "person", "plate_only", "other"] = Field(alias="class")
    bbox_xyxy: BBox
    confidence: float = Confidence
    track_id: Optional[int] = None
    plate: Optional[Plate] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _bbox(self):
        _check_bbox(self.bbox_xyxy)
        return self


class Analysis(Strict):
    detector: str
    ocr: Optional[str] = None
    detect_ms: Optional[float] = Field(default=None, ge=0)
    ocr_ms: Optional[float] = Field(default=None, ge=0)
    detections: List[Detection]


class Stage3Frame(Stage2Frame):
    analysis: Analysis

    @model_validator(mode="after")
    def _bbox_dentro_de_imagen_original(self):
        """Atrapa el error tipico: cajas calculadas sobre la imagen x4 sin reescalar."""
        W, H = self.width, self.height
        for d in self.analysis.detections:
            boxes = [("bbox_xyxy", d.bbox_xyxy)]
            if d.plate:
                boxes.append(("plate.bbox_xyxy", d.plate.bbox_xyxy))
            for name, (x1, y1, x2, y2) in boxes:
                if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                    raise ValueError(
                        f"{d.detection_id}.{name} sale de la imagen original ({W}x{H}). "
                        f"Caja={[x1, y1, x2, y2]}. Probable falta de reescalar desde la imagen mejorada."
                    )
        return self


# ---------- Resumen global para el panel ----------
class CameraSummary(Strict):
    camera_id: str
    usable_without_replacement: Optional[bool] = None
    note: Optional[str] = None


class Summary(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    total_frames: int = Field(ge=0)
    plates_detected: int = Field(ge=0)
    plates_read_original: int = Field(ge=0)
    plates_read_enhanced: int = Field(ge=0)
    avg_confidence_original: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_confidence_enhanced: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_latency_ms_per_frame: Optional[float] = Field(default=None, ge=0)
    per_camera: List[CameraSummary]

    @model_validator(mode="after")
    def _coherencia(self):
        for n in ("plates_read_original", "plates_read_enhanced"):
            if getattr(self, n) > self.plates_detected:
                raise ValueError(f"{n} no puede superar plates_detected")
        return self


STAGES = {"1": Stage1Frame, "2": Stage2Frame, "3": Stage3Frame, "summary": Summary}
