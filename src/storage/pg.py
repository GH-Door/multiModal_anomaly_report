"""PostgreSQL CRUD for inspection reports with legacy compatibility."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

logger = logging.getLogger(__name__)

REPORT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspection_reports (
  id                     SERIAL PRIMARY KEY,
  dataset                VARCHAR(50),
  category               VARCHAR(100),
  line                   VARCHAR(50),
  ad_score               FLOAT,
  ad_decision            VARCHAR(20),
  is_anomaly_ad          BOOLEAN,
  has_defect             BOOLEAN,
  region                 TEXT,
  bbox                   JSONB,
  center                 JSONB,
  area_ratio             FLOAT,
  confidence             FLOAT,
  ingest_source_path     TEXT,
  image_path             TEXT,
  heatmap_path           TEXT,
  mask_path              TEXT,
  overlay_path           TEXT,
  similar_image_path     TEXT,
  ad_start_time          TIMESTAMP,
  ad_inference_duration  FLOAT,
  is_anomaly_llm         BOOLEAN,
  llm_report             JSONB,
  llm_summary            JSONB,
  llm_start_time         TIMESTAMP,
  llm_inference_duration FLOAT,
  applied_policy         JSONB DEFAULT '{}'::jsonb,
  created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CATEGORY_METADATA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS category_metadata (
  category                VARCHAR(100) PRIMARY KEY,
  dataset                 VARCHAR(50),
  line                    VARCHAR(50) DEFAULT 'line_1',
  t_low                   FLOAT DEFAULT 0.5,
  t_high                  FLOAT DEFAULT 0.8,
  review_band             FLOAT,
  reliability             VARCHAR(20),
  ad_weight               FLOAT,
  location_mode           VARCHAR(20),
  min_location_confidence FLOAT,
  use_bbox                BOOLEAN,
  created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

REPORT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_reports_category ON inspection_reports(category);
CREATE INDEX IF NOT EXISTS idx_reports_decision ON inspection_reports(ad_decision);
CREATE INDEX IF NOT EXISTS idx_reports_created ON inspection_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_reports_ingest_source_path ON inspection_reports(ingest_source_path);
"""

ALTER_REPORT_COLUMNS_SQL = [
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ad_decision VARCHAR(20);",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS has_defect BOOLEAN;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS region TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS bbox JSONB;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS center JSONB;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS area_ratio FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS confidence FLOAT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS ingest_source_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS overlay_path TEXT;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS applied_policy JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE inspection_reports ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
]

DEFAULT_POLICY = {"t_low": 0.5, "t_high": 0.8}

ALLOWED_COLUMNS = {
    "dataset",
    "category",
    "line",
    "ad_score",
    "ad_decision",
    "is_anomaly_ad",
    "has_defect",
    "region",
    "bbox",
    "center",
    "area_ratio",
    "confidence",
    "ingest_source_path",
    "image_path",
    "heatmap_path",
    "mask_path",
    "overlay_path",
    "similar_image_path",
    "ad_start_time",
    "ad_inference_duration",
    "is_anomaly_llm",
    "llm_report",
    "llm_summary",
    "llm_start_time",
    "llm_inference_duration",
    "applied_policy",
    "created_at",
}

JSON_COLUMNS = {"bbox", "center", "llm_report", "llm_summary", "applied_policy"}

COLUMN_ALIASES = {
    "is_anomaly_ad": "is_anomaly_ad",
    "is_anomaly_AD": "is_anomaly_ad",
    "is_anomaly_llm": "is_anomaly_llm",
    "is_anomaly_LLM": "is_anomaly_llm",
    "ad_start_time": "ad_start_time",
    "AD_start_time": "ad_start_time",
    "ad_inference_duration": "ad_inference_duration",
    "AD_inference_duration": "ad_inference_duration",
}


def connect(dsn: str, *, ensure_schema: bool = True) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL. Schema sync is optional for hot paths."""
    conn = psycopg2.connect(dsn)
    if ensure_schema:
        create_tables(conn)
    return conn


def connect_fast(dsn: str) -> psycopg2.extensions.connection:
    """Connect without running schema DDL on every request."""
    return connect(dsn, ensure_schema=False)


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create required tables and ensure additive schema compatibility."""
    with conn.cursor() as cur:
        cur.execute(REPORT_SCHEMA_SQL)
        cur.execute(CATEGORY_METADATA_SCHEMA_SQL)
        for ddl in ALTER_REPORT_COLUMNS_SQL:
            cur.execute(ddl)
        cur.execute(REPORT_INDEX_SQL)
    conn.commit()


def _normalize_column_name(raw_key: str) -> Optional[str]:
    col = COLUMN_ALIASES.get(raw_key, raw_key)
    if col in ALLOWED_COLUMNS:
        return col
    col_lower = COLUMN_ALIASES.get(raw_key.lower(), raw_key.lower())
    if col_lower in ALLOWED_COLUMNS:
        return col_lower
    return None


def _adapt_json_value(column: str, value: Any) -> Any:
    if column not in JSON_COLUMNS or value is None:
        return value
    if isinstance(value, (dict, list)):
        return Json(value)
    if isinstance(value, str):
        try:
            return Json(json.loads(value))
        except json.JSONDecodeError:
            return Json({"raw": value})
    return Json(value)


def _normalized_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw_key, raw_value in data.items():
        col = _normalize_column_name(str(raw_key))
        if col is None:
            continue
        out[col] = _adapt_json_value(col, raw_value)
    return out


def insert_report(conn: psycopg2.extensions.connection, data: Dict[str, Any]) -> int:
    """Insert a report row and return its id."""
    payload = _normalized_payload(data)
    if not payload:
        raise ValueError("No valid columns to insert into inspection_reports")

    columns = list(payload.keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    sql = f"INSERT INTO inspection_reports ({col_names}) VALUES ({placeholders}) RETURNING id"

    with conn.cursor() as cur:
        cur.execute(sql, [payload[c] for c in columns])
        report_id = cur.fetchone()[0]
    conn.commit()
    return int(report_id)


def update_report(conn: psycopg2.extensions.connection, report_id: int, data: Dict[str, Any]) -> None:
    """Update report row by id with partial data."""
    payload = _normalized_payload(data)
    if not payload:
        return

    columns = list(payload.keys())
    set_clause = ", ".join([f"{col} = %s" for col in columns])
    sql = f"UPDATE inspection_reports SET {set_clause} WHERE id = %s"

    with conn.cursor() as cur:
        cur.execute(sql, [payload[c] for c in columns] + [report_id])
    conn.commit()


def list_reports(conn: psycopg2.extensions.connection, limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent N reports."""
    sql = """
    SELECT * FROM inspection_reports
    ORDER BY created_at DESC NULLS LAST, id DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_report(conn: psycopg2.extensions.connection, report_id: int) -> Optional[Dict[str, Any]]:
    """Return a single report by id, or None."""
    sql = "SELECT * FROM inspection_reports WHERE id = %s"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (report_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def get_filtered_reports(
    conn: psycopg2.extensions.connection,
    *,
    category: str | None = None,
    decision: str | None = None,
    limit: int = 5000,
    offset: int = 0,
    include_full: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch reports with optional category/decision filters."""
    clauses = ["1=1"]
    params: List[Any] = []

    if category:
        clauses.append("category = %s")
        params.append(category)
    if decision:
        clauses.append("ad_decision = %s")
        params.append(decision)

    params.extend([limit, offset])
    select_sql = "*"
    if not include_full:
        select_sql = """
            id,
            dataset,
            category,
            line,
            ad_score,
            ad_decision,
            has_defect,
            region,
            area_ratio,
            confidence,
            image_path,
            heatmap_path,
            mask_path,
            overlay_path,
            ad_inference_duration,
            is_anomaly_llm,
            llm_report,
            llm_summary,
            llm_inference_duration,
            applied_policy,
            created_at
        """

    sql = f"""
    SELECT {select_sql} FROM inspection_reports
    WHERE {' AND '.join(clauses)}
    ORDER BY created_at DESC NULLS LAST, id DESC
    LIMIT %s OFFSET %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def count_filtered_reports(
    conn: psycopg2.extensions.connection,
    *,
    category: str | None = None,
    decision: str | None = None,
) -> int:
    """Count reports with the same filters used by get_filtered_reports."""
    clauses = ["1=1"]
    params: List[Any] = []

    if category:
        clauses.append("category = %s")
        params.append(category)
    if decision:
        clauses.append("ad_decision = %s")
        params.append(decision)

    sql = f"""
    SELECT COUNT(*) AS n
    FROM inspection_reports
    WHERE {' AND '.join(clauses)}
    """
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def has_ingest_source_path(conn: psycopg2.extensions.connection, source_path: str) -> bool:
    """Return True when a report with the same incoming source path already exists."""
    if not source_path:
        return False
    sql = "SELECT 1 FROM inspection_reports WHERE ingest_source_path = %s LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (source_path,))
        row = cur.fetchone()
    return bool(row)


def get_category_policy(conn: psycopg2.extensions.connection, category: str) -> Dict[str, Any]:
    """Load per-category threshold policy with defaults."""
    sql = "SELECT t_low, t_high FROM category_metadata WHERE category = %s"
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (category,))
            row = cur.fetchone()
        if row:
            policy = dict(row)
            return {
                "t_low": float(policy.get("t_low", DEFAULT_POLICY["t_low"])),
                "t_high": float(policy.get("t_high", DEFAULT_POLICY["t_high"])),
                "source": "category_metadata",
            }
    except Exception as exc:
        logger.warning("Failed to load category policy for %s: %s", category, exc)
    return {
        **dict(DEFAULT_POLICY),
        "source": "default",
    }
