from typing import Optional

try:
    from pydantic import AliasChoices, Field
except Exception:
    from stmem.utils.pydantic_compat import AliasChoices, Field
from stmem.settings import settings_config

from stmem.storage.config.base import BaseVectorStoreConfig, BaseGraphStoreConfig


class SQLiteConfig(BaseVectorStoreConfig):
    """Configuration for SQLite vector store."""
    
    _provider_name = "sqlite"
    _class_path = "stmem.storage.sqlite.sqlite_vector_store.SQLiteVectorStore"
    
    model_config = settings_config("VECTOR_STORE_", extra="forbid", env_file=None)
    
    database_path: str = Field(
        default="./database/stmem_dev.db",
        validation_alias=AliasChoices(
            "database_path",
            "SQLITE_PATH",
        ),
        description="Path to SQLite database file"
    )
    
    collection_name: str = Field(
        default="memories",
        validation_alias=AliasChoices(
            "collection_name",
            "SQLITE_COLLECTION",
        ),
        description="Name of the collection/table"
    )
    
    enable_wal: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_wal",
            "SQLITE_ENABLE_WAL",
        ),
        description="Enable Write-Ahead Logging for better concurrency"
    )
    
    timeout: int = Field(
        default=30,
        validation_alias=AliasChoices(
            "timeout",
            "SQLITE_TIMEOUT",
        ),
        description="Connection timeout in seconds"
    )


class SQLiteGraphConfig(BaseGraphStoreConfig):
    _provider_name = "sqlite"
    _class_path = "stmem.storage.sqlite.sqlite_graph_store.SQLiteGraphStore"

    model_config = settings_config("GRAPH_STORE_", extra="forbid", env_file=None)

    database_path: str = Field(
        default="./data/stmem_graph.db",
        validation_alias=AliasChoices(
            "database_path",
            "GRAPH_STORE_DATABASE_PATH",
            "GRAPH_STORE_SQLITE_PATH",
            "SQLITE_PATH",
        ),
        description="Path to SQLite database file for graph store"
    )

    entities_table: str = Field(
        default="graph_entities",
        validation_alias=AliasChoices(
            "entities_table",
            "GRAPH_STORE_ENTITIES_TABLE",
        ),
        description="SQLite table name for graph entities"
    )

    relationships_table: str = Field(
        default="graph_relationships",
        validation_alias=AliasChoices(
            "relationships_table",
            "GRAPH_STORE_RELATIONSHIPS_TABLE",
        ),
        description="SQLite table name for graph relationships"
    )
