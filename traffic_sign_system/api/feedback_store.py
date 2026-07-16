"""Durable local store for human corrections and model feedback."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FeedbackStore:
    """Small SQLite-backed store with no extra runtime dependency.

    The store is intentionally local to the application. It keeps the feedback
    loop useful across browser refreshes and API restarts while leaving model
    training as an explicit export step.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _session(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._lock, self._session() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    history_id TEXT,
                    source TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    model TEXT,
                    predicted_class_id INTEGER NOT NULL,
                    predicted_class_name TEXT NOT NULL,
                    predicted_confidence REAL,
                    corrected_class_id INTEGER NOT NULL,
                    corrected_class_name TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    bbox_json TEXT,
                    status TEXT NOT NULL DEFAULT 'new'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    history_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    model TEXT,
                    predicted_class_id INTEGER NOT NULL,
                    predicted_class_name TEXT NOT NULL,
                    predicted_confidence REAL,
                    bbox_json TEXT,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    feedback_id TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_queue_created ON review_queue(created_at DESC)"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    @staticmethod
    def _csv_safe(value: Any) -> Any:
        if value is None:
            return ""
        if not isinstance(value, str):
            return value
        return "'" + value if value.startswith(("=", "+", "-", "@")) else value

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["bbox"] = json.loads(item.pop("bbox_json")) if item.get("bbox_json") else None
        return item

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = self._now()
        record = {
            "id": uuid.uuid4().hex,
            "created_at": now,
            "updated_at": now,
            "history_id": payload.get("history_id"),
            "source": payload["source"],
            "filename": payload["filename"],
            "model": payload.get("model"),
            "predicted_class_id": int(payload["predicted_class_id"]),
            "predicted_class_name": payload["predicted_class_name"],
            "predicted_confidence": payload.get("predicted_confidence"),
            "corrected_class_id": int(payload["corrected_class_id"]),
            "corrected_class_name": payload["corrected_class_name"],
            "verdict": payload["verdict"],
            "note": payload.get("note", "").strip(),
            "bbox_json": json.dumps(payload.get("bbox"), ensure_ascii=False) if payload.get("bbox") else None,
            "status": "new",
        }
        with self._lock, self._session() as connection:
            connection.execute(
                """
                INSERT INTO feedback (
                    id, created_at, updated_at, history_id, source, filename,
                    model, predicted_class_id, predicted_class_name,
                    predicted_confidence, corrected_class_id, corrected_class_name,
                    verdict, note, bbox_json, status
                ) VALUES (
                    :id, :created_at, :updated_at, :history_id, :source, :filename,
                    :model, :predicted_class_id, :predicted_class_name,
                    :predicted_confidence, :corrected_class_id, :corrected_class_name,
                    :verdict, :note, :bbox_json, :status
                )
                """,
                record,
            )
        record["bbox"] = payload.get("bbox")
        record.pop("bbox_json")
        return record

    def list(self, *, status: str | None = None, limit: int = 100, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._session() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM feedback {where}", params).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM feedback {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._row_to_dict(row) for row in rows], total

    def stats(self) -> dict[str, int]:
        with self._lock, self._session() as connection:
            rows = connection.execute(
                "SELECT status, verdict, COUNT(*) AS count FROM feedback GROUP BY status, verdict"
            ).fetchall()
        result = {"total": 0, "incorrect": 0, "correct": 0, "new": 0, "reviewed": 0, "exported": 0}
        for row in rows:
            count = int(row["count"])
            result["total"] += count
            result[row["verdict"]] = result.get(row["verdict"], 0) + count
            result[row["status"]] = result.get(row["status"], 0) + count
        return result

    def update_status(self, feedback_id: str, status: str) -> dict[str, Any] | None:
        now = self._now()
        with self._lock, self._session() as connection:
            cursor = connection.execute(
                "UPDATE feedback SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, feedback_id),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def delete(self, feedback_id: str) -> bool:
        with self._lock, self._session() as connection:
            cursor = connection.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        return cursor.rowcount > 0

    def enqueue_review(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Add a low-confidence result to the review queue exactly once."""
        now = self._now()
        record = {
            "id": uuid.uuid4().hex,
            "created_at": now,
            "updated_at": now,
            "history_id": payload["history_id"],
            "source": payload["source"],
            "filename": payload["filename"],
            "model": payload.get("model"),
            "predicted_class_id": int(payload["predicted_class_id"]),
            "predicted_class_name": payload["predicted_class_name"],
            "predicted_confidence": payload.get("predicted_confidence"),
            "bbox_json": json.dumps(payload.get("bbox"), ensure_ascii=False) if payload.get("bbox") else None,
            "reason": payload.get("reason", "low_confidence"),
            "status": "pending",
            "feedback_id": None,
        }
        with self._lock, self._session() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO review_queue (
                    id, created_at, updated_at, history_id, source, filename, model,
                    predicted_class_id, predicted_class_name, predicted_confidence,
                    bbox_json, reason, status, feedback_id
                ) VALUES (
                    :id, :created_at, :updated_at, :history_id, :source, :filename, :model,
                    :predicted_class_id, :predicted_class_name, :predicted_confidence,
                    :bbox_json, :reason, :status, :feedback_id
                )
                """,
                record,
            )
            row = connection.execute(
                "SELECT * FROM review_queue WHERE history_id = ?", (record["history_id"],)
            ).fetchone()
        return self._row_to_dict(row), cursor.rowcount > 0

    def list_review_queue(
        self, *, status: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._session() as connection:
            total = int(connection.execute(
                f"SELECT COUNT(*) FROM review_queue {where}", params
            ).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM review_queue {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [self._row_to_dict(row) for row in rows], total

    def review_queue_stats(self) -> dict[str, int]:
        with self._lock, self._session() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM review_queue GROUP BY status"
            ).fetchall()
        result = {"total": 0, "pending": 0, "reviewed": 0, "dismissed": 0}
        for row in rows:
            count = int(row["count"])
            result["total"] += count
            result[row["status"]] = count
        return result

    def get_review_queue_item(self, queue_id: str) -> dict[str, Any] | None:
        with self._lock, self._session() as connection:
            row = connection.execute(
                "SELECT * FROM review_queue WHERE id = ?", (queue_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def resolve_review_queue(self, queue_id: str, feedback_id: str) -> dict[str, Any] | None:
        now = self._now()
        with self._lock, self._session() as connection:
            cursor = connection.execute(
                """
                UPDATE review_queue
                SET status = 'reviewed', feedback_id = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (feedback_id, now, queue_id),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM review_queue WHERE id = ?", (queue_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def update_review_queue_status(self, queue_id: str, status: str) -> dict[str, Any] | None:
        now = self._now()
        with self._lock, self._session() as connection:
            cursor = connection.execute(
                "UPDATE review_queue SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, queue_id),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM review_queue WHERE id = ?", (queue_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def export_csv(self, *, status: str | None = None) -> str:
        items, _ = self.list(status=status, limit=100_000, offset=0)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "id", "created_at", "source", "filename", "model",
                "predicted_class_id", "predicted_class_name", "predicted_confidence",
                "corrected_class_id", "corrected_class_name", "verdict", "note", "status",
            ]
        )
        for item in items:
            writer.writerow([
                self._csv_safe(item["id"]), self._csv_safe(item["created_at"]),
                self._csv_safe(item["source"]), self._csv_safe(item["filename"]),
                self._csv_safe(item.get("model")), item["predicted_class_id"],
                self._csv_safe(item["predicted_class_name"]), item.get("predicted_confidence"),
                item["corrected_class_id"], self._csv_safe(item["corrected_class_name"]),
                self._csv_safe(item["verdict"]), self._csv_safe(item["note"]), self._csv_safe(item["status"]),
            ])
        return output.getvalue()
