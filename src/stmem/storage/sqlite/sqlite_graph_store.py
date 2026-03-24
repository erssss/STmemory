import logging
import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from stmem.storage.base import GraphStoreBase
from stmem.utils.utils import generate_snowflake_id, get_current_datetime, serialize_datetime

logger = logging.getLogger(__name__)


class SQLiteGraphStore(GraphStoreBase):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.db_path = self.config.get("database_path", "./data/stmem_graph.db")
        self.entities_table = self.config.get("entities_table", "graph_entities")
        self.relationships_table = self.config.get("relationships_table", "graph_relationships")
        self.max_hops = int(self.config.get("max_hops", 3))

        self.connection: sqlite3.Connection
        self._lock = threading.Lock()

        if self.db_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.entities_table} (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    run_id TEXT,
                    name TEXT NOT NULL,
                    entity_type TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            self.connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_scope_name
                ON {self.entities_table} (user_id, run_id, name)
                """
            )
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.relationships_table} (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    run_id TEXT,
                    source TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            self.connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_scope_edge
                ON {self.relationships_table} (user_id, run_id, source, relationship, destination)
                """
            )
            self.connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_relationships_scope_src
                ON {self.relationships_table} (user_id, run_id, source)
                """
            )
            self.connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_relationships_scope_dst
                ON {self.relationships_table} (user_id, run_id, destination)
                """
            )
            self.connection.commit()

    def _scope(self, filters: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        user_id = str(filters.get("user_id") or "")
        if not user_id:
            raise ValueError("filters.user_id is required for graph operations")
        run_id = filters.get("run_id")
        return user_id, (str(run_id) if run_id is not None and run_id != "" else None)

    def _extract_relations(self, text: str) -> List[Dict[str, str]]:
        if not text:
            return []

        relations: list[Dict[str, str]] = []

        latin_patterns = [
            r"(?P<src>[A-Za-z0-9_]+)\s+(?P<rel>knows|likes|loves|works_with|works-with|is|are)\s+(?P<dst>[A-Za-z0-9_]+)",
        ]
        for pat in latin_patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                src = m.group("src").strip()
                rel = m.group("rel").strip().lower().replace("-", "_")
                dst = m.group("dst").strip()
                if src and dst and rel:
                    relations.append({"source": src, "relationship": rel, "destination": dst})

        zh_patterns = [
            r"(?P<src>[\u4e00-\u9fff]{1,10})(?P<rel>认识|喜欢|讨厌|属于|是)(?P<dst>[\u4e00-\u9fff]{1,10})",
        ]
        for pat in zh_patterns:
            for m in re.finditer(pat, text):
                src = m.group("src").strip()
                rel = m.group("rel").strip()
                dst = m.group("dst").strip()
                if src and dst and rel:
                    relations.append({"source": src, "relationship": rel, "destination": dst})

        return relations

    def _upsert_entity(self, *, user_id: str, run_id: Optional[str], name: str) -> None:
        now = serialize_datetime(get_current_datetime())
        with self._lock:
            cursor = self.connection.execute(
                f"""
                SELECT id FROM {self.entities_table}
                WHERE user_id = ? AND (run_id IS ? OR run_id = ?) AND name = ?
                LIMIT 1
                """,
                (user_id, run_id, run_id, name),
            )
            row = cursor.fetchone()
            if row:
                self.connection.execute(
                    f"""
                    UPDATE {self.entities_table}
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (now, row[0]),
                )
            else:
                entity_id = generate_snowflake_id()
                self.connection.execute(
                    f"""
                    INSERT INTO {self.entities_table}
                    (id, user_id, run_id, name, entity_type, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (entity_id, user_id, run_id, name, None, now, now),
                )

    def _upsert_relationship(
        self, *, user_id: str, run_id: Optional[str], source: str, relationship: str, destination: str
    ) -> None:
        now = serialize_datetime(get_current_datetime())
        rel_id = generate_snowflake_id()
        with self._lock:
            self.connection.execute(
                f"""
                INSERT OR IGNORE INTO {self.relationships_table}
                (id, user_id, run_id, source, relationship, destination, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rel_id, user_id, run_id, source, relationship, destination, now, now),
            )
            self.connection.execute(
                f"""
                UPDATE {self.relationships_table}
                SET updated_at = ?
                WHERE user_id = ? AND (run_id IS ? OR run_id = ?) AND source = ? AND relationship = ? AND destination = ?
                """,
                (now, user_id, run_id, run_id, source, relationship, destination),
            )

    def add(self, data: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        user_id, run_id = self._scope(filters)
        extracted = self._extract_relations(data)
        if not extracted:
            return {"deleted_entities": [], "added_entities": []}

        for r in extracted:
            src = r["source"]
            dst = r["destination"]
            rel = r["relationship"]
            self._upsert_entity(user_id=user_id, run_id=run_id, name=src)
            self._upsert_entity(user_id=user_id, run_id=run_id, name=dst)
            self._upsert_relationship(
                user_id=user_id, run_id=run_id, source=src, relationship=rel, destination=dst
            )

        with self._lock:
            self.connection.commit()

        return {"deleted_entities": [], "added_entities": extracted}

    def search(self, query: str, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        user_id, run_id = self._scope(filters)
        query = (query or "").strip()
        if not query:
            return []

        candidates = set()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,10}", query):
            if token:
                candidates.add(token)

        params: list[Any] = [user_id, run_id, run_id]
        where_parts: list[str] = ["user_id = ? AND (run_id IS ? OR run_id = ?)"]

        if candidates:
            placeholders = ",".join(["?"] * len(candidates))
            where_parts.append(
                f"(source IN ({placeholders}) OR destination IN ({placeholders}))"
            )
            params.extend(list(candidates))
            params.extend(list(candidates))
        else:
            where_parts.append("(source LIKE ? OR destination LIKE ? OR relationship LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like, like])

        sql = (
            f"SELECT source, relationship, destination FROM {self.relationships_table} "
            f"WHERE {' AND '.join(where_parts)} "
            f"ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(int(limit))

        with self._lock:
            cursor = self.connection.execute(sql, params)
            rows = cursor.fetchall()

        return [
            {"source": row[0], "relationship": row[1], "destination": row[2]}
            for row in rows
        ]

    def delete_all(self, filters: Dict[str, Any]) -> None:
        user_id, run_id = self._scope(filters)
        with self._lock:
            self.connection.execute(
                f"DELETE FROM {self.relationships_table} WHERE user_id = ? AND (run_id IS ? OR run_id = ?)",
                (user_id, run_id, run_id),
            )
            self.connection.execute(
                f"DELETE FROM {self.entities_table} WHERE user_id = ? AND (run_id IS ? OR run_id = ?)",
                (user_id, run_id, run_id),
            )
            self.connection.commit()

    def get_all(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, str]]:
        user_id, run_id = self._scope(filters)
        with self._lock:
            cursor = self.connection.execute(
                f"""
                SELECT source, relationship, destination
                FROM {self.relationships_table}
                WHERE user_id = ? AND (run_id IS ? OR run_id = ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, run_id, run_id, int(limit)),
            )
            rows = cursor.fetchall()
        return [
            {"source": row[0], "relationship": row[1], "destination": row[2]}
            for row in rows
        ]

    def reset(self) -> None:
        with self._lock:
            self.connection.execute(f"DELETE FROM {self.relationships_table}")
            self.connection.execute(f"DELETE FROM {self.entities_table}")
            self.connection.commit()

    def get_statistics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if filters:
            user_id, run_id = self._scope(filters)
            where = "WHERE user_id = ? AND (run_id IS ? OR run_id = ?)"
            params = (user_id, run_id, run_id)
        else:
            where = ""
            params = ()

        with self._lock:
            entities_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.entities_table} {where}", params
            ).fetchone()[0]
            rels_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.relationships_table} {where}", params
            ).fetchone()[0]

        return {
            "entities": int(entities_count),
            "relationships": int(rels_count),
        }

    def get_unique_users(self) -> List[str]:
        with self._lock:
            cursor = self.connection.execute(
                f"SELECT DISTINCT user_id FROM {self.entities_table}"
            )
            rows = cursor.fetchall()
        return [str(row[0]) for row in rows if row and row[0]]

