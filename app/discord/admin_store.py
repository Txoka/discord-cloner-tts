from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, List


@dataclass(frozen=True)
class AdminRecord:
    user_id: int
    role: str  # "admin" | "superadmin"


class AdminStore:
    def __init__(self, db_path: Path, master_superadmins: Iterable[int], debug_guild_ids: Iterable[int]):
        self._db_path = Path(db_path)
        self._master_superadmins = {int(x) for x in master_superadmins}
        self._debug_guild_ids = {int(x) for x in debug_guild_ids}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            # New schema: multiple roles per user for future expansion.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admin_roles (
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, role)
                )
                """
            )
            # Legacy table migration (single role per user).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    user_id INTEGER PRIMARY KEY,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'superadmin')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            rows = conn.execute("SELECT user_id, role FROM admins").fetchall()
            for row in rows:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO admin_roles (user_id, role)
                    VALUES (?, ?)
                    """,
                    (int(row["user_id"]), str(row["role"])),
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS admin_guilds (
                    guild_id INTEGER PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def bootstrap_superadmins(self) -> None:
        if not self._master_superadmins:
            return
        with self._connect() as conn:
            for user_id in self._master_superadmins:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO admin_roles (user_id, role)
                    VALUES (?, 'superadmin')
                    """,
                    (int(user_id),),
                )
            conn.commit()

    def bootstrap_debug_guilds(self) -> None:
        if not self._debug_guild_ids:
            return
        with self._connect() as conn:
            for guild_id in self._debug_guild_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO admin_guilds (guild_id)
                    VALUES (?)
                    """,
                    (int(guild_id),),
                )
            conn.commit()

    def is_superadmin(self, user_id: int) -> bool:
        user_id = int(user_id)
        if user_id in self._master_superadmins:
            return True
        return self.has_role(user_id, "superadmin")

    def is_admin(self, user_id: int) -> bool:
        user_id = int(user_id)
        if self.is_superadmin(user_id):
            return True
        return self.has_role(user_id, "admin")

    def get_admin(self, user_id: int) -> Optional[AdminRecord]:
        roles = self.list_roles(user_id)
        if not roles:
            return None
        return AdminRecord(user_id=int(user_id), role=roles[0])

    def has_role(self, user_id: int, role: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM admin_roles WHERE user_id = ? AND role = ?",
                (int(user_id), role),
            ).fetchone()
        return row is not None

    def list_roles(self, user_id: int) -> Sequence[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role FROM admin_roles WHERE user_id = ? ORDER BY role",
                (int(user_id),),
            ).fetchall()
        return [str(r["role"]) for r in rows]

    def list_admins(self) -> List[AdminRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id, role FROM admin_roles ORDER BY user_id, role"
            ).fetchall()
        return [AdminRecord(user_id=int(r["user_id"]), role=str(r["role"])) for r in rows]

    def list_debug_guilds(self) -> List[int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT guild_id FROM admin_guilds ORDER BY guild_id").fetchall()
        return [int(r["guild_id"]) for r in rows]

    def add_debug_guild(self, guild_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO admin_guilds (guild_id) VALUES (?)",
                (int(guild_id),),
            )
            conn.commit()

    def remove_debug_guild(self, guild_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM admin_guilds WHERE guild_id = ?", (int(guild_id),))
            conn.commit()
            return cur.rowcount > 0

    def add_role(self, user_id: int, role: str) -> None:
        if int(user_id) in self._master_superadmins:
            role = "superadmin"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO admin_roles (user_id, role)
                VALUES (?, ?)
                """,
                (int(user_id), role),
            )
            conn.commit()

    def upsert_admin(self, user_id: int, role: str) -> None:
        if role not in {"admin", "superadmin"}:
            raise ValueError("role must be 'admin' or 'superadmin'")
        self.add_role(user_id, role)

    def remove_admin(self, user_id: int) -> bool:
        user_id = int(user_id)
        if user_id in self._master_superadmins:
            return False
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM admin_roles WHERE user_id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0
