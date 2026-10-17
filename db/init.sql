CREATE TABLE IF NOT EXISTS frames (
  frame_id      TEXT PRIMARY KEY,
  camera_id     TEXT NOT NULL,
  ts_utc        TIMESTAMPTZ NOT NULL,
  image_path    TEXT NOT NULL,
  sha256        CHAR(64) NOT NULL,
  enhanced_path TEXT,
  raw           JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
  id                   BIGSERIAL PRIMARY KEY,
  frame_id             TEXT NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
  detection_id         TEXT NOT NULL,
  class                TEXT NOT NULL,
  confidence           REAL,
  plate_text_original  TEXT,
  plate_conf_original  REAL,
  plate_text_enhanced  TEXT,
  plate_conf_enhanced  REAL,
  readable_improved    BOOLEAN
);

-- Quien consulto que (trazabilidad). Aqui es donde LinuxONE aporta valor en la propuesta.
CREATE TABLE IF NOT EXISTS audit_log (
  id      BIGSERIAL PRIMARY KEY,
  ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor   TEXT NOT NULL,
  action  TEXT NOT NULL,
  target  TEXT,
  detail  JSONB
);
