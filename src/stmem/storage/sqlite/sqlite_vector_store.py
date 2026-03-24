"""
SQLite vector store implementation

This module provides a simple SQLite-based vector store for development and testing.
"""

import heapq
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from stmem.storage.base import VectorStoreBase, OutputData
from stmem.utils.utils import generate_snowflake_id

logger = logging.getLogger(__name__)


class SQLiteVectorStore(VectorStoreBase):
    """Simple SQLite-based vector store implementation."""

    def __init__(self, database_path: str = ":memory:", collection_name: str = "memories", **kwargs):
        """
        Initialize SQLite vector store.

        Args:
            database_path: Path to SQLite database file
            collection_name: Name of the collection/table
        """
        self.db_path = database_path
        self.collection_name = collection_name
        self.connection = None
        self._lock = threading.RLock()

        if database_path != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(database_path))
            if db_dir and not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"Created database directory: {db_dir}")
                except OSError as e:
                    logger.error(f"Failed to create database directory {db_dir}: {e}")
                    raise

        try:
            self.connection = sqlite3.connect(database_path, check_same_thread=False)
        except Exception as e:
            logger.error(f"Failed to connect to SQLite database at {database_path}: {e}")
            raise

        self.create_col()

        logger.info(f"SQLiteVectorStore initialized with db_path: {database_path}")

    def _fts_table_name(self, table_name: str) -> str:
        return f"{table_name}_fts"

    def _ensure_fts_table(self, table_name: str) -> bool:
        fts_table = self._fts_table_name(table_name)
        try:
            self.connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5(content)"
            )
            self.connection.commit()
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 not available, disabling full-text search: {e}")
            return False

    def _backfill_fts(self, table_name: str) -> None:
        fts_table = self._fts_table_name(table_name)
        with self._lock:
            try:
                cur = self.connection.cursor()
                cur.execute(
                    f"""
                    INSERT INTO {fts_table}(rowid, content)
                    SELECT m.id,
                           COALESCE(
                               json_extract(m.payload, '$.fulltext_content'),
                               json_extract(m.payload, '$.data'),
                               ''
                           )
                    FROM {table_name} m
                    LEFT JOIN {fts_table} f ON f.rowid = m.id
                    WHERE f.rowid IS NULL
                    """
                )
                self.connection.commit()
            except sqlite3.OperationalError:
                return

    def _upsert_fts_row(self, table_name: str, row_id: int, content: str) -> None:
        fts_table = self._fts_table_name(table_name)
        try:
            self.connection.execute(f"DELETE FROM {fts_table} WHERE rowid = ?", (row_id,))
            self.connection.execute(
                f"INSERT INTO {fts_table}(rowid, content) VALUES (?, ?)",
                (row_id, content or ""),
            )
        except sqlite3.OperationalError:
            return

    def _delete_fts_row(self, table_name: str, row_id: int) -> None:
        fts_table = self._fts_table_name(table_name)
        try:
            self.connection.execute(f"DELETE FROM {fts_table} WHERE rowid = ?", (row_id,))
        except sqlite3.OperationalError:
            return

    def create_col(self, name=None, vector_size=None, distance=None) -> None:
        """Create a new collection (table in SQLite)."""
        table_name = name or self.collection_name

        with self._lock:
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    vector TEXT,
                    payload TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            if self._ensure_fts_table(table_name):
                self._backfill_fts(table_name)
            self.connection.commit()

    def _build_filter_clause(
        self, filters: Optional[Dict[str, Any]], payload_column: str = "payload"
    ) -> tuple[str, list[Any]]:
        if not filters:
            return "", []

        top_level_keys = {
            "id",
            "user_id",
            "run_id",
            "actor_id",
            "hash",
            "category",
            "created_at",
            "created_at_ts",
            "updated_at",
            "updated_at_ts",
            "role",
            "data",
            "fulltext_content",
            "access_count",
        }

        conditions: list[str] = []
        params: list[Any] = []

        for raw_key, value in filters.items():
            raw = str(raw_key)
            op = None
            if raw.endswith("_gte"):
                op = ">="
                key = raw[:-4]
            elif raw.endswith("_lte"):
                op = "<="
                key = raw[:-4]
            else:
                key = raw
            if key.startswith("$."):
                primary_path = key
                alt_path = None
            elif key.startswith("metadata."):
                primary_path = f"$.{key}"
                alt_path = None
            else:
                primary_path = f"$.{key}"
                alt_path = None if key in top_level_keys else f"$.metadata.{key}"

            if value is None:
                if alt_path:
                    conditions.append(
                        f"(json_extract({payload_column}, ?) IS NULL OR json_extract({payload_column}, ?) IS NULL)"
                    )
                    params.extend([primary_path, alt_path])
                else:
                    conditions.append(f"json_extract({payload_column}, ?) IS NULL")
                    params.append(primary_path)
                continue

            if op is not None:
                conditions.append(
                    f"CAST(json_extract({payload_column}, ?) AS INTEGER) {op} ?"
                )
                params.extend([primary_path, value])
                continue

            if alt_path:
                conditions.append(
                    f"(json_extract({payload_column}, ?) = ? OR json_extract({payload_column}, ?) = ?)"
                )
                params.extend([primary_path, value, alt_path, value])
            else:
                conditions.append(f"json_extract({payload_column}, ?) = ?")
                params.extend([primary_path, value])

        if not conditions:
            return "", []

        return " WHERE " + " AND ".join(conditions), params

    def insert(self, vectors: List[List[float]], payloads=None, ids=None) -> List[int]:
        """
        Insert vectors into the collection.

        Args:
            vectors: List of vectors to insert
            payloads: List of payload dictionaries
            ids: Deprecated parameter (ignored), IDs are now generated using Snowflake algorithm

        Returns:
            List[int]: List of generated Snowflake IDs
        """
        if not vectors:
            return []

        if payloads is None:
            payloads = [{} for _ in vectors]

        if ids is not None:
            if len(ids) != len(vectors):
                raise ValueError("ids length must match vectors length")
            generated_ids = [int(x) for x in ids]
        else:
            generated_ids = [generate_snowflake_id() for _ in range(len(vectors))]

        with self._lock:
            for vector, payload, vector_id in zip(vectors, payloads, generated_ids):
                self.connection.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.collection_name}
                    (id, vector, payload) VALUES (?, ?, ?)
                    """,
                    (vector_id, json.dumps(vector), json.dumps(payload)),
                )
                content = ""
                if isinstance(payload, dict):
                    content = payload.get("fulltext_content") or payload.get("data") or ""
                self._upsert_fts_row(self.collection_name, int(vector_id), str(content))

            self.connection.commit()

        return generated_ids

    def _extract_query_vector(self, query: Any, vectors: Any) -> List[float]:
        if vectors is None:
            return query if isinstance(query, list) else []
        if isinstance(vectors, list):
            if not vectors:
                return query if isinstance(query, list) else []
            head = vectors[0]
            if isinstance(head, (int, float)):
                return vectors
            if isinstance(head, list):
                return head
        return query if isinstance(query, list) else []

    def _normalize_bm25_to_similarity(self, bm25_value: Optional[float]) -> float:
        if bm25_value is None:
            return 0.0
        try:
            val = float(bm25_value)
        except Exception:
            return 0.0
        if val <= 0:
            return 1.0
        return 1.0 / (1.0 + val)

    def search(
        self,
        query,
        vectors=None,
        limit: int = 5,
        filters=None,
        sparse_embedding=None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> List[OutputData]:
        results: list[OutputData] = []

        query_vector = self._extract_query_vector(query, vectors)
        if not query_vector:
            query_vector = [0.1] * 10

        fulltext_weight_raw = kwargs.get("fulltext_weight") or os.getenv(
            "SQLITE_FULLTEXT_WEIGHT", ""
        )
        try:
            fulltext_weight = float(fulltext_weight_raw) if str(fulltext_weight_raw).strip() else 0.3
        except Exception:
            fulltext_weight = 0.3
        fulltext_weight = max(0.0, min(1.0, fulltext_weight))

        vec_candidate_limit_raw = kwargs.get("vector_candidates") or os.getenv(
            "SQLITE_VECTOR_CANDIDATES", ""
        )
        try:
            vec_candidate_limit = int(vec_candidate_limit_raw) if str(vec_candidate_limit_raw).strip() else 0
        except Exception:
            vec_candidate_limit = 0
        if vec_candidate_limit <= 0:
            vec_candidate_limit = max(int(limit) * 10, 50)

        text_candidate_limit_raw = kwargs.get("fulltext_candidates") or os.getenv(
            "SQLITE_FULLTEXT_CANDIDATES", ""
        )
        try:
            text_candidate_limit = int(text_candidate_limit_raw) if str(text_candidate_limit_raw).strip() else 0
        except Exception:
            text_candidate_limit = 0
        if text_candidate_limit <= 0:
            text_candidate_limit = max(int(limit) * 10, 50)

        query_text = str(query or "").strip()
        fts_enabled = bool(query_text) and self._ensure_fts_table(self.collection_name)
        if fts_enabled:
            self._backfill_fts(self.collection_name)

        best_vector_heap: list[tuple[float, int, dict]] = []
        where_sql, query_params = self._build_filter_clause(filters)
        query_sql = f"SELECT id, vector, payload FROM {self.collection_name}{where_sql}"

        text_candidates: dict[int, tuple[float, list[float], dict]] = {}
        if fts_enabled:
            fts_table = self._fts_table_name(self.collection_name)
            fts_where = f" WHERE {fts_table} MATCH ?"
            fts_params: list[Any] = [query_text]
            if where_sql:
                fts_where += " AND " + where_sql[len(" WHERE ") :]
                fts_params.extend(query_params)
            fts_sql = (
                f"SELECT m.id, m.vector, m.payload, bm25({fts_table}) AS bm25_score "
                f"FROM {self.collection_name} m "
                f"JOIN {fts_table} ON {fts_table}.rowid = m.id"
                f"{fts_where} "
                f"ORDER BY bm25_score ASC LIMIT ?"
            )
            fts_params.append(int(text_candidate_limit))
            with self._lock:
                try:
                    cur = self.connection.execute(fts_sql, fts_params)
                    for row in cur.fetchall():
                        row_id, vector_str, payload_str, bm25_score = row
                        try:
                            vec = json.loads(vector_str) if vector_str else []
                        except Exception:
                            vec = []
                        try:
                            payload = json.loads(payload_str) if payload_str else {}
                        except Exception:
                            payload = {}
                        text_candidates[int(row_id)] = (
                            self._normalize_bm25_to_similarity(bm25_score),
                            vec,
                            payload,
                        )
                except sqlite3.OperationalError:
                    fts_enabled = False

        with self._lock:
            if query_params:
                cursor = self.connection.execute(query_sql, query_params)
            else:
                cursor = self.connection.execute(query_sql)

            for row in cursor.fetchall():
                vector_id, vector_str, payload_str = row
                try:
                    vector = json.loads(vector_str) if vector_str else []
                except Exception:
                    vector = []
                try:
                    payload = json.loads(payload_str) if payload_str else {}
                except Exception:
                    payload = {}
                similarity = self._cosine_similarity(query_vector, vector)
                vid = int(vector_id)

                if len(best_vector_heap) < int(vec_candidate_limit):
                    heapq.heappush(best_vector_heap, (similarity, vid, payload))
                else:
                    if similarity > best_vector_heap[0][0]:
                        heapq.heapreplace(best_vector_heap, (similarity, vid, payload))

        vector_candidates: dict[int, tuple[float, dict]] = {
            vid: (sim, payload) for sim, vid, payload in best_vector_heap
        }

        candidate_ids = set(vector_candidates.keys()) | set(text_candidates.keys())
        combined: list[tuple[float, int, dict]] = []

        for vid in candidate_ids:
            vec_entry = vector_candidates.get(vid)
            if vec_entry is not None:
                vec_sim = vec_entry[0]
                payload = vec_entry[1] or {}
            else:
                tc = text_candidates.get(vid)
                if tc is not None and tc[1]:
                    vec_sim = self._cosine_similarity(query_vector, tc[1])
                else:
                    vec_sim = 0.0
                payload = (tc[2] if tc is not None else {}) or {}

            text_sim = text_candidates.get(vid, (0.0, [], {}))[0]
            if fts_enabled:
                score = (1.0 - fulltext_weight) * float(vec_sim) + fulltext_weight * float(text_sim)
                quality = score
            else:
                score = float(vec_sim)
                quality = score

            if threshold is not None and quality < float(threshold):
                continue

            payload_out = payload.copy() if isinstance(payload, dict) else {}
            md = payload_out.get("metadata") if isinstance(payload_out.get("metadata"), dict) else {}
            md2 = dict(md)
            md2["_quality_score"] = quality
            md2["_vector_score"] = float(vec_sim)
            if fts_enabled:
                md2["_fulltext_score"] = float(text_sim)
                md2["_fulltext_weight"] = float(fulltext_weight)
            payload_out["metadata"] = md2
            combined.append((score, vid, payload_out))

        combined.sort(key=lambda x: x[0], reverse=True)
        for score, vid, payload in combined[: int(limit)]:
            results.append(OutputData(id=vid, score=float(score), payload=payload))

        return results

    def delete(self, vector_id: int) -> None:
        """Delete a vector by ID."""
        with self._lock:
            self.connection.execute(
                f"""
                DELETE FROM {self.collection_name} WHERE id = ?
                """,
                (vector_id,),
            )
            self._delete_fts_row(self.collection_name, int(vector_id))
            self.connection.commit()

    def update(self, vector_id: int, vector=None, payload=None) -> None:
        """Update a vector and its payload."""
        updates = []
        values = []

        if vector is not None:
            updates.append("vector = ?")
            values.append(json.dumps(vector))

        if payload is not None:
            updates.append("payload = ?")
            values.append(json.dumps(payload))

        if updates:
            values.append(vector_id)
            with self._lock:
                self.connection.execute(
                    f"""
                    UPDATE {self.collection_name}
                    SET {', '.join(updates)}
                    WHERE id = ?
                    """,
                    values,
                )
                if payload is not None and isinstance(payload, dict):
                    content = payload.get("fulltext_content") or payload.get("data") or ""
                    self._upsert_fts_row(self.collection_name, int(vector_id), str(content))
                self.connection.commit()

    def get(self, vector_id: int) -> Optional[OutputData]:
        """Retrieve a vector by ID."""
        with self._lock:
            cursor = self.connection.execute(
                f"""
                SELECT id, vector, payload FROM {self.collection_name} WHERE id = ?
                """,
                (vector_id,),
            )

            row = cursor.fetchone()
            if row:
                vector_id, vector_str, payload_str = row
                vector = json.loads(vector_str)
                payload = json.loads(payload_str)

                return OutputData(id=vector_id, score=1.0, payload=payload)

        return None

    def list_cols(self) -> List[str]:
        """List all collections (tables)."""
        with self._lock:
            cursor = self.connection.execute(
                """
                SELECT name FROM sqlite_master WHERE type='table'
                """
            )
            return [row[0] for row in cursor.fetchall()]

    def delete_col(self) -> None:
        """Delete the collection (table)."""
        with self._lock:
            self.connection.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
            self.connection.execute(
                f"DROP TABLE IF EXISTS {self._fts_table_name(self.collection_name)}"
            )
            self.connection.commit()

    def col_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        with self._lock:
            cursor = self.connection.execute(
                f"""
                SELECT COUNT(*) FROM {self.collection_name}
                """
            )
            count = cursor.fetchone()[0]

            return {"name": self.collection_name, "count": count, "db_path": self.db_path}

    def list(self, filters=None, limit=None, offset=None, order_by=None, order="desc") -> List[OutputData]:
        """List all memories with optional filtering, pagination and sorting."""
        query = f"SELECT id, vector, payload FROM {self.collection_name}"
        where_sql, query_params = self._build_filter_clause(filters)
        query += where_sql

        if order_by:
            order_upper = order.upper()
            if order_by in ["created_at", "updated_at"]:
                query += f" ORDER BY json_extract(payload, '$.{order_by}') {order_upper}"
            elif order_by == "id":
                query += f" ORDER BY id {order_upper}"

        if limit is not None:
            query += f" LIMIT {limit}"
        elif offset is not None:
            query += " LIMIT -1"
        if offset is not None:
            query += f" OFFSET {offset}"

        results = []
        with self._lock:
            if query_params:
                cursor = self.connection.execute(query, query_params)
            else:
                cursor = self.connection.execute(query)

            for row in cursor.fetchall():
                vector_id, vector_str, payload_str = row
                vector = json.loads(vector_str)
                payload = json.loads(payload_str)

                results.append(OutputData(id=vector_id, score=1.0, payload=payload))

        return results

    def count(self, filters=None) -> int:
        """Count all memories with optional filtering.

        Args:
            filters: Optional filters dictionary

        Returns:
            int: Total count of memories matching the filters
        """
        query = f"SELECT COUNT(*) FROM {self.collection_name}"
        where_sql, query_params = self._build_filter_clause(filters)
        query += where_sql

        with self._lock:
            if query_params:
                cursor = self.connection.execute(query, query_params)
            else:
                cursor = self.connection.execute(query)

            count = cursor.fetchone()[0]

        return count

    def reset(self) -> None:
        """Reset by deleting and recreating the collection."""
        self.delete_col()
        self.create_col()

    def get_statistics(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get statistics for the memories in SQLite."""
        query = f"SELECT id, payload, created_at FROM {self.collection_name}"
        query_params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"json_extract(payload, '$.{key}') = ?")
                query_params.append(value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        stats = {
            "total_memories": 0,
            "by_type": {},
            "avg_importance": 0.0,
            "top_accessed": [],
            "growth_trend": {},
            "age_distribution": {
                "< 1 day": 0,
                "1-7 days": 0,
                "7-30 days": 0,
                "> 30 days": 0,
            },
        }

        total_importance = 0.0
        importance_count = 0

        with self._lock:
            cursor = self.connection.execute(query, query_params)
            rows = cursor.fetchall()

            stats["total_memories"] = len(rows)
            if not rows:
                return stats

            from datetime import datetime

            now = datetime.now()

            memories_with_access = []

            for row in rows:
                row_id, payload_str, created_at_str = row
                payload = json.loads(payload_str)

                m_type = payload.get("category") or payload.get("type") or "unknown"
                stats["by_type"][m_type] = stats["by_type"].get(m_type, 0) + 1

                user_metadata = payload.get("metadata", {})
                importance = user_metadata.get("importance") or payload.get("importance")
                if importance is not None:
                    try:
                        total_importance += float(importance)
                        importance_count += 1
                    except (ValueError, TypeError):
                        pass

                access_count = user_metadata.get("access_count") or payload.get("access_count") or 0

                content = payload.get("data") or payload.get("content") or ""

                memories_with_access.append(
                    {"id": row_id, "content": content[:50], "access_count": int(access_count)}
                )

                if created_at_str:
                    date_part = created_at_str.split(" ")[0]
                    stats["growth_trend"][date_part] = stats["growth_trend"].get(date_part, 0) + 1

                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace(" ", "T"))
                        days_old = (now - created_at).days
                        if days_old < 1:
                            stats["age_distribution"]["< 1 day"] += 1
                        elif days_old < 7:
                            stats["age_distribution"]["1-7 days"] += 1
                        elif days_old < 30:
                            stats["age_distribution"]["7-30 days"] += 1
                        else:
                            stats["age_distribution"]["> 30 days"] += 1
                    except Exception:
                        pass

            if importance_count > 0:
                stats["avg_importance"] = round(total_importance / importance_count, 2)

            memories_with_access.sort(key=lambda x: x["access_count"], reverse=True)
            stats["top_accessed"] = memories_with_access[:10]

        return stats

    def get_unique_users(self) -> List[str]:
        """Get a list of unique user IDs from SQLite."""
        query = f"SELECT DISTINCT json_extract(payload, '$.user_id') FROM {self.collection_name}"

        users = []
        with self._lock:
            cursor = self.connection.execute(query)
            for row in cursor.fetchall():
                if row[0]:
                    users.append(str(row[0]))

        return users

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
