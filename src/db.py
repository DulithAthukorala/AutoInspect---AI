import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path(os.getenv("DB_PATH", "data/autoinspect.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    created_at: str
    image_path: str
    image_sha256: str
    weights_path: str
    vehicle_weights: str
    thresholds_version: str
    response_json: Dict[str, Any]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                weights_path TEXT NOT NULL,
                vehicle_weights TEXT NOT NULL,
                thresholds_version TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at)")
        conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_case(
    *,
    case_id: str,
    image_path: str,
    image_sha256: str,
    weights_path: str,
    vehicle_weights: str,
    thresholds_version: str,
    response_json: Dict[str, Any],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO cases (
                case_id, created_at, image_path, image_sha256,
                weights_path, vehicle_weights, thresholds_version, response_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                now_iso(),
                image_path,
                image_sha256,
                weights_path,
                vehicle_weights,
                thresholds_version,
                json.dumps(response_json, ensure_ascii=False),
            ),
        )
        conn.commit()


def get_case(case_id: str) -> Optional[CaseRow]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            return None

        return CaseRow(
            case_id=row["case_id"],
            created_at=row["created_at"],
            image_path=row["image_path"],
            image_sha256=row["image_sha256"],
            weights_path=row["weights_path"],
            vehicle_weights=row["vehicle_weights"],
            thresholds_version=row["thresholds_version"],
            response_json=json.loads(row["response_json"]),
        )
