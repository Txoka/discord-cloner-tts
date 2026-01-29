from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class AdminRecord:
    user_id: int
    role: str  # "admin" | "superadmin"


class AdminStore:
    def __init__(self, db_path: Path, master_superadmins: Iterable[int]):
        self._db_path = Path(db_path)
        self._master_superadmins = {int(x) for x in master_superadmins}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    user_id INTEGER PRIMARY KEY,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'superadmin')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def bootstrap_superadmins(self) -> None:
        if not self._master_superadmins:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for user_id in self._master_superadmins:
                conn.execute(
                    """
                    INSERT INTO admins (user_id, role)
                    VALUES (?, 'superadmin')
                    ON CONFLICT(user_id) DO UPDATE SET role='superadmin'
                    """,
                    (int(user_id),),
                )
            conn.commit()

    def is_superadmin(self, user_id: int) -> bool:
        user_id = int(user_id)
        if user_id in self._master_superadmins:
            return True
        rec = self.get_admin(user_id)
        return rec is not None and rec.role == "superadmin"

    def is_admin(self, user_id: int) -> bool:
        user_id = int(user_id)
        if self.is_superadmin(user_id):
            return True
        rec = self.get_admin(user_id)
        return rec is not None and rec.role == "admin"

    def get_admin(self, user_id: int) -> Optional[AdminRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, role FROM admins WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        if not row:
            return None
        return AdminRecord(user_id=int(row["user_id"]), role=str(row["role"]))

    def upsert_admin(self, user_id: int, role: str) -> None:
        if role not in {"admin", "superadmin"}:
            raise ValueError("role must be 'admin' or 'superadmin'")
        if int(user_id) in self._master_superadmins:
            role = "superadmin"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO admins (user_id, role)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET role=excluded.role
                """,
                (int(user_id), role),
            )
            conn.commit()

    def remove_admin(self, user_id: int) -> bool:
        user_id = int(user_id)
        if user_id in self._master_superadmins:
            return False
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM admins WHERE user_id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0
