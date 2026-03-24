"""
Sub-store migration status management module.

This module provides a database-backed migration status manager for sub-stores.
"""

import logging
import uuid
import json
from datetime import datetime
import sqlite3
from stmem.utils.utils import get_current_datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MigrationStatus:
    """Migration status enum"""
    PENDING = "pending"        # Registered, not started
    MIGRATING = "migrating"    # Migration in progress
    COMPLETED = "completed"    # Completed, ready for routing
    FAILED = "failed"          # Migration failed


class SubStoreMigrationManager:
    """
    Sub-store migration status manager (database-backed).
    
    This class manages migration status for sub-stores using a database table.
    All status information is persisted in the database, supporting multi-process
    sharing and application restarts.
    """
    
    def __init__(self, vector_store, main_collection_name: str):
        """
        Initialize migration manager.
        
        Args:
            vector_store: Vector store instance (must support SQL operations)
            main_collection_name: Main collection name
        """
        self.vector_store = vector_store
        self.main_collection_name = main_collection_name
        
        # Table name for migration status
        self.status_table = "sub_store_migration_status"
        
        # Ensure table exists
        self._init_status_table()
    
    def _init_status_table(self):
        """Initialize migration status table"""
        try:
            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.status_table} (
                        id TEXT PRIMARY KEY,
                        main_collection_name TEXT NOT NULL,
                        sub_store_name TEXT NOT NULL,
                        routing_filter TEXT,
                        status TEXT DEFAULT 'pending',
                        migrated_count INTEGER DEFAULT 0,
                        total_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        started_at TEXT,
                        completed_at TEXT,
                        UNIQUE (main_collection_name, sub_store_name)
                    )
                    """
                )
                conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_sub_store_main_collection
                    ON {self.status_table} (main_collection_name)
                    """
                )
                conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_sub_store_status
                    ON {self.status_table} (status)
                    """
                )
                conn.commit()
                logger.info(f"Migration status table '{self.status_table}' initialized")
                return

            if not hasattr(self.vector_store, "execute_sql"):
                logger.warning(
                    "Vector store does not support SQL operations, migration status will not be persisted"
                )
                return

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.status_table} (
                id VARCHAR(64) PRIMARY KEY,
                main_collection_name VARCHAR(128) NOT NULL,
                sub_store_name VARCHAR(128) NOT NULL,
                routing_filter TEXT,
                status VARCHAR(32) DEFAULT 'pending',
                migrated_count INT DEFAULT 0,
                total_count INT DEFAULT 0,
                error_message TEXT,
                created_at VARCHAR(128),
                updated_at VARCHAR(128),
                started_at VARCHAR(128),
                completed_at VARCHAR(128)
            )
            """
            self.vector_store.execute_sql(create_table_sql)
            
        except Exception as e:
            logger.warning(f"Failed to initialize migration status table: {e}")
            logger.warning("Migration status will not be persisted")
    
    def register_sub_store(
        self,
        sub_store_name: str,
        routing_filter: Dict
    ):
        """
        Register a sub store with pending status.
        
        Args:
            sub_store_name: Sub store name
            routing_filter: Routing filter dict
        """
        try:
            status_id = str(uuid.uuid4())
            now = get_current_datetime().isoformat()
            
            # Convert routing filter to JSON string
            routing_filter_json = json.dumps(routing_filter)

            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                cursor = conn.execute(
                    f"""
                    SELECT id FROM {self.status_table}
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    LIMIT 1
                    """,
                    (self.main_collection_name, sub_store_name),
                )
                row = cursor.fetchone()
                if row:
                    conn.execute(
                        f"""
                        UPDATE {self.status_table}
                        SET routing_filter = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (routing_filter_json, now, row[0]),
                    )
                else:
                    conn.execute(
                        f"""
                        INSERT INTO {self.status_table}
                        (id, main_collection_name, sub_store_name, routing_filter, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            status_id,
                            self.main_collection_name,
                            sub_store_name,
                            routing_filter_json,
                            MigrationStatus.PENDING,
                            now,
                            now,
                        ),
                    )
                conn.commit()
                logger.info(
                    f"Registered sub store '{sub_store_name}' with status: pending"
                )
                return

            if not hasattr(self.vector_store, "execute_sql"):
                logger.debug(
                    "Skipping registration: vector store does not support SQL"
                )
                return

            insert_sql = f"""
            INSERT INTO {self.status_table} 
                (id, main_collection_name, sub_store_name, routing_filter, status, created_at, updated_at)
            VALUES 
                ('{status_id}', '{self.main_collection_name}', '{sub_store_name}', 
                 '{routing_filter_json}', '{MigrationStatus.PENDING}', '{now}', '{now}')
            """
            self.vector_store.execute_sql(insert_sql)
            logger.info(f"Registered sub store '{sub_store_name}' with status: pending")
            
        except Exception as e:
            logger.error(f"Failed to register sub store: {e}")
    
    def is_ready(self, sub_store_name: str) -> bool:
        """
        Check if sub store is ready (migration completed).
        
        Args:
            sub_store_name: Sub store name
            
        Returns:
            True if migration completed, False otherwise
        """
        try:
            status = self.get_status(sub_store_name)
            if status:
                return status.get("status") == MigrationStatus.COMPLETED
            return False
            
        except Exception as e:
            logger.error(f"Failed to check sub store readiness: {e}")
            return False
    
    def mark_migrating(self, sub_store_name: str, total_count: int):
        """
        Mark sub store as migrating.
        
        Args:
            sub_store_name: Sub store name
            total_count: Total number of records to migrate
        """
        try:
            now = get_current_datetime().isoformat()

            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                conn.execute(
                    f"""
                    UPDATE {self.status_table}
                    SET status = ?,
                        total_count = ?,
                        started_at = ?,
                        updated_at = ?
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    """,
                    (
                        MigrationStatus.MIGRATING,
                        int(total_count),
                        now,
                        now,
                        self.main_collection_name,
                        sub_store_name,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Marked sub store '{sub_store_name}' as migrating (total: {total_count})"
                )
                return

            if not hasattr(self.vector_store, "execute_sql"):
                return

            update_sql = f"""
            UPDATE {self.status_table}
            SET status = '{MigrationStatus.MIGRATING}',
                total_count = {total_count},
                started_at = '{now}',
                updated_at = '{now}'
            WHERE main_collection_name = '{self.main_collection_name}'
              AND sub_store_name = '{sub_store_name}'
            """
            self.vector_store.execute_sql(update_sql)
            logger.info(f"Marked sub store '{sub_store_name}' as migrating (total: {total_count})")
            
        except Exception as e:
            logger.error(f"Failed to mark sub store as migrating: {e}")
    
    def update_progress(self, sub_store_name: str, migrated_count: int):
        """
        Update migration progress.
        
        Args:
            sub_store_name: Sub store name
            migrated_count: Number of migrated records
        """
        try:
            now = get_current_datetime().isoformat()

            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                conn.execute(
                    f"""
                    UPDATE {self.status_table}
                    SET migrated_count = ?, updated_at = ?
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    """,
                    (int(migrated_count), now, self.main_collection_name, sub_store_name),
                )
                conn.commit()
                logger.debug(f"Updated progress for '{sub_store_name}': {migrated_count}")
                return

            if not hasattr(self.vector_store, "execute_sql"):
                return

            update_sql = f"""
            UPDATE {self.status_table}
            SET migrated_count = {migrated_count},
                updated_at = '{now}'
            WHERE main_collection_name = '{self.main_collection_name}'
              AND sub_store_name = '{sub_store_name}'
            """
            self.vector_store.execute_sql(update_sql)
            logger.debug(f"Updated progress for '{sub_store_name}': {migrated_count}")
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
    
    def mark_completed(self, sub_store_name: str, migrated_count: int):
        """
        Mark sub store migration as completed.
        
        Args:
            sub_store_name: Sub store name
            migrated_count: Final migrated count
        """
        try:
            now = get_current_datetime().isoformat()

            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                conn.execute(
                    f"""
                    UPDATE {self.status_table}
                    SET status = ?,
                        migrated_count = ?,
                        completed_at = ?,
                        updated_at = ?,
                        error_message = NULL
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    """,
                    (
                        MigrationStatus.COMPLETED,
                        int(migrated_count),
                        now,
                        now,
                        self.main_collection_name,
                        sub_store_name,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Marked sub store '{sub_store_name}' as completed ({migrated_count} records)"
                )
                return

            if not hasattr(self.vector_store, "execute_sql"):
                return

            update_sql = f"""
            UPDATE {self.status_table}
            SET status = '{MigrationStatus.COMPLETED}',
                migrated_count = {migrated_count},
                completed_at = '{now}',
                updated_at = '{now}',
                error_message = NULL
            WHERE main_collection_name = '{self.main_collection_name}'
              AND sub_store_name = '{sub_store_name}'
            """
            self.vector_store.execute_sql(update_sql)
            logger.info(f"Marked sub store '{sub_store_name}' as completed ({migrated_count} records)")
            
        except Exception as e:
            logger.error(f"Failed to mark sub store as completed: {e}")
    
    def mark_failed(self, sub_store_name: str, error_message: str):
        """
        Mark sub store migration as failed.
        
        Args:
            sub_store_name: Sub store name
            error_message: Error message
        """
        try:
            now = get_current_datetime().isoformat()

            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                conn.execute(
                    f"""
                    UPDATE {self.status_table}
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    """,
                    (
                        MigrationStatus.FAILED,
                        str(error_message),
                        now,
                        self.main_collection_name,
                        sub_store_name,
                    ),
                )
                conn.commit()
                logger.error(
                    f"Marked sub store '{sub_store_name}' as failed: {error_message}"
                )
                return

            if not hasattr(self.vector_store, "execute_sql"):
                return

            error_message_escaped = error_message.replace("'", "''")
            update_sql = f"""
            UPDATE {self.status_table}
            SET status = '{MigrationStatus.FAILED}',
                error_message = '{error_message_escaped}',
                updated_at = '{now}'
            WHERE main_collection_name = '{self.main_collection_name}'
              AND sub_store_name = '{sub_store_name}'
            """
            self.vector_store.execute_sql(update_sql)
            logger.error(f"Marked sub store '{sub_store_name}' as failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to mark sub store as failed: {e}")
    
    def get_status(self, sub_store_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status for a sub store.
        
        Args:
            sub_store_name: Sub store name
            
        Returns:
            Status dict or None if not found
        """
        try:
            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                cursor = conn.execute(
                    f"""
                    SELECT id, main_collection_name, sub_store_name, routing_filter, status,
                           migrated_count, total_count, error_message, created_at, updated_at,
                           started_at, completed_at
                    FROM {self.status_table}
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    LIMIT 1
                    """,
                    (self.main_collection_name, sub_store_name),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "main_collection_name": row[1],
                    "sub_store_name": row[2],
                    "routing_filter": json.loads(row[3] or "{}"),
                    "status": row[4],
                    "migrated_count": row[5] or 0,
                    "total_count": row[6] or 0,
                    "error_message": row[7],
                    "created_at": row[8],
                    "updated_at": row[9],
                    "started_at": row[10],
                    "completed_at": row[11],
                }

            if not hasattr(self.vector_store, "execute_sql"):
                return None

            query_sql = f"""
            SELECT * FROM {self.status_table}
            WHERE main_collection_name = '{self.main_collection_name}'
              AND sub_store_name = '{sub_store_name}'
            LIMIT 1
            """
            result = self.vector_store.execute_sql(query_sql)
            if result and len(result) > 0:
                row = result[0]
                return {
                    "id": row.get("id"),
                    "main_collection_name": row.get("main_collection_name"),
                    "sub_store_name": row.get("sub_store_name"),
                    "routing_filter": json.loads(row.get("routing_filter", "{}")),
                    "status": row.get("status"),
                    "migrated_count": row.get("migrated_count", 0),
                    "total_count": row.get("total_count", 0),
                    "error_message": row.get("error_message"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "started_at": row.get("started_at"),
                    "completed_at": row.get("completed_at"),
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return None
    
    def get_migration_progress(self, sub_store_name: str) -> Optional[Dict[str, Any]]:
        """
        Get migration progress including percentage.
        
        Args:
            sub_store_name: Sub store name
            
        Returns:
            Progress dict or None if not found
        """
        try:
            status = self.get_status(sub_store_name)
            if not status:
                return None
            
            total_count = status.get("total_count", 0)
            migrated_count = status.get("migrated_count", 0)
            
            # Calculate percentage
            if total_count > 0:
                progress_percentage = (migrated_count / total_count) * 100
            else:
                progress_percentage = 0.0
            
            # Calculate elapsed time
            started_at = status.get("started_at")
            updated_at = status.get("updated_at")
            elapsed_seconds = None
            
            if started_at and updated_at:
                try:
                    start_time = datetime.fromisoformat(started_at)
                    current_time = datetime.fromisoformat(updated_at)
                    elapsed_seconds = (current_time - start_time).total_seconds()
                except Exception:
                    pass
            
            return {
                "sub_store_name": sub_store_name,
                "status": status.get("status"),
                "progress_percentage": progress_percentage,
                "migrated_count": migrated_count,
                "total_count": total_count,
                "elapsed_seconds": elapsed_seconds,
                "error_message": status.get("error_message"),
                "is_ready": status.get("status") == MigrationStatus.COMPLETED
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration progress: {e}")
            return None
    
    def list_all_status(self) -> List[Dict[str, Any]]:
        """
        List status for all sub stores under this main collection.
        
        Returns:
            List of status dicts
        """
        try:
            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                cursor = conn.execute(
                    f"""
                    SELECT id, sub_store_name, routing_filter, status, migrated_count, total_count,
                           error_message, created_at, updated_at, started_at, completed_at
                    FROM {self.status_table}
                    WHERE main_collection_name = ?
                    ORDER BY created_at
                    """,
                    (self.main_collection_name,),
                )
                rows = cursor.fetchall()
                status_list: list[Dict[str, Any]] = []
                for row in rows:
                    status_list.append(
                        {
                            "id": row[0],
                            "sub_store_name": row[1],
                            "routing_filter": json.loads(row[2] or "{}"),
                            "status": row[3],
                            "migrated_count": row[4] or 0,
                            "total_count": row[5] or 0,
                            "error_message": row[6],
                            "created_at": row[7],
                            "updated_at": row[8],
                            "started_at": row[9],
                            "completed_at": row[10],
                        }
                    )
                return status_list

            if not hasattr(self.vector_store, "execute_sql"):
                return []

            query_sql = f"""
            SELECT * FROM {self.status_table}
            WHERE main_collection_name = '{self.main_collection_name}'
            ORDER BY created_at
            """
            results = self.vector_store.execute_sql(query_sql)
            status_list = []
            for row in results:
                status_list.append(
                    {
                        "id": row.get("id"),
                        "sub_store_name": row.get("sub_store_name"),
                        "routing_filter": json.loads(row.get("routing_filter", "{}")),
                        "status": row.get("status"),
                        "migrated_count": row.get("migrated_count", 0),
                        "total_count": row.get("total_count", 0),
                        "error_message": row.get("error_message"),
                        "created_at": row.get("created_at"),
                        "updated_at": row.get("updated_at"),
                        "started_at": row.get("started_at"),
                        "completed_at": row.get("completed_at"),
                    }
                )
            return status_list
            
        except Exception as e:
            logger.error(f"Failed to list all status: {e}")
            return []
    
    def reset_status(self, sub_store_name: str):
        """
        Reset sub store migration status to pending (for retry).
        
        Args:
            sub_store_name: Sub store name
        """
        try:
            now = get_current_datetime().isoformat()

            conn = getattr(self.vector_store, "connection", None)
            if isinstance(conn, sqlite3.Connection):
                conn.execute(
                    f"""
                    UPDATE {self.status_table}
                    SET status = ?,
                        migrated_count = 0,
                        total_count = 0,
                        error_message = NULL,
                        started_at = NULL,
                        completed_at = NULL,
                        updated_at = ?
                    WHERE main_collection_name = ? AND sub_store_name = ?
                    """,
                    (
                        MigrationStatus.PENDING,
                        now,
                        self.main_collection_name,
                        sub_store_name,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Reset migration status for sub store '{sub_store_name}'"
                )
                return

            if not hasattr(self.vector_store, "execute_sql"):
                return

            update_sql = f"""
            UPDATE {self.status_table}
            SET status = '{MigrationStatus.PENDING}',
                migrated_count = 0,
                total_count = 0,
                error_message = NULL,
                started_at = NULL,
                completed_at = NULL,
                updated_at = '{now}'
            WHERE main_collection_name = '{self.main_collection_name}'
              AND sub_store_name = '{sub_store_name}'
            """
            self.vector_store.execute_sql(update_sql)
            logger.info(f"Reset migration status for sub store '{sub_store_name}'")
            
        except Exception as e:
            logger.error(f"Failed to reset status: {e}")
