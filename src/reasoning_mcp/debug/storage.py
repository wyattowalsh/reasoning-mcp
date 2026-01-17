"""Storage layer for trace persistence and retrieval.

This module provides multiple storage backends for persisting reasoning traces,
including in-memory storage for testing, file-based storage for development,
and SQLite storage for production use.
"""

import sqlite3
from pathlib import Path
from typing import Protocol

from reasoning_mcp.models.debug import Trace


class TraceStorage(Protocol):
    """Protocol for trace storage backends.

    Defines the interface that all trace storage implementations must follow.
    This allows for flexible storage backend selection based on deployment needs.
    """

    def save(self, trace: Trace) -> None:
        """Save a trace to storage.

        Args:
            trace: The trace to save
        """
        ...

    def load(self, trace_id: str) -> Trace | None:
        """Load a trace by ID from storage.

        Args:
            trace_id: The unique identifier of the trace to load

        Returns:
            The loaded trace, or None if not found
        """
        ...

    def list_traces(self, session_id: str) -> list[str]:
        """List trace IDs for a given session.

        Args:
            session_id: The session identifier to filter traces by

        Returns:
            List of trace IDs belonging to the session
        """
        ...

    def delete(self, trace_id: str) -> bool:
        """Delete a trace from storage.

        Args:
            trace_id: The unique identifier of the trace to delete

        Returns:
            True if the trace was deleted, False if it didn't exist
        """
        ...


class MemoryTraceStorage:
    """In-memory trace storage for testing and development.

    This storage backend keeps all traces in memory, making it ideal for
    testing and development scenarios where persistence is not required.
    All data is lost when the process exits.
    """

    def __init__(self) -> None:
        """Initialize in-memory storage with empty trace and index dictionaries."""
        self._traces: dict[str, Trace] = {}
        self._session_index: dict[str, list[str]] = {}

    def save(self, trace: Trace) -> None:
        """Save a trace to memory.

        Args:
            trace: The trace to save
        """
        self._traces[trace.trace_id] = trace
        if trace.session_id not in self._session_index:
            self._session_index[trace.session_id] = []
        if trace.trace_id not in self._session_index[trace.session_id]:
            self._session_index[trace.session_id].append(trace.trace_id)

    def load(self, trace_id: str) -> Trace | None:
        """Load a trace from memory.

        Args:
            trace_id: The unique identifier of the trace to load

        Returns:
            The loaded trace, or None if not found
        """
        return self._traces.get(trace_id)

    def list_traces(self, session_id: str) -> list[str]:
        """List trace IDs for a session from memory.

        Args:
            session_id: The session identifier to filter traces by

        Returns:
            List of trace IDs belonging to the session
        """
        return self._session_index.get(session_id, [])

    def delete(self, trace_id: str) -> bool:
        """Delete a trace from memory.

        Args:
            trace_id: The unique identifier of the trace to delete

        Returns:
            True if the trace was deleted, False if it didn't exist
        """
        if trace_id not in self._traces:
            return False
        trace = self._traces.pop(trace_id)
        if trace.session_id in self._session_index:
            self._session_index[trace.session_id] = [
                tid for tid in self._session_index[trace.session_id] if tid != trace_id
            ]
        return True


class FileTraceStorage:
    """JSON file-based trace storage for development and small deployments.

    This storage backend persists traces as individual JSON files in a directory.
    Each trace is stored in a separate file named {trace_id}.json. This approach
    is simple and works well for development and small-scale deployments.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize file-based storage with the specified directory.

        Args:
            directory: The directory where trace files will be stored
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _trace_path(self, trace_id: str) -> Path:
        """Get the file path for a trace ID.

        Args:
            trace_id: The trace identifier

        Returns:
            The full path to the trace file
        """
        return self.directory / f"{trace_id}.json"

    def save(self, trace: Trace) -> None:
        """Save a trace to a JSON file.

        Args:
            trace: The trace to save
        """
        path = self._trace_path(trace.trace_id)
        path.write_text(trace.model_dump_json(indent=2))

    def load(self, trace_id: str) -> Trace | None:
        """Load a trace from a JSON file.

        Args:
            trace_id: The unique identifier of the trace to load

        Returns:
            The loaded trace, or None if not found
        """
        path = self._trace_path(trace_id)
        if not path.exists():
            return None
        return Trace.model_validate_json(path.read_text())

    def list_traces(self, session_id: str) -> list[str]:
        """List trace IDs for a session by scanning JSON files.

        This method scans all JSON files in the directory and loads each one
        to check if it belongs to the specified session. This can be slow for
        large numbers of traces.

        Args:
            session_id: The session identifier to filter traces by

        Returns:
            List of trace IDs belonging to the session
        """
        result = []
        for path in self.directory.glob("*.json"):
            try:
                trace = Trace.model_validate_json(path.read_text())
                if trace.session_id == session_id:
                    result.append(trace.trace_id)
            except Exception:
                continue
        return result

    def delete(self, trace_id: str) -> bool:
        """Delete a trace file.

        Args:
            trace_id: The unique identifier of the trace to delete

        Returns:
            True if the trace was deleted, False if it didn't exist
        """
        path = self._trace_path(trace_id)
        if not path.exists():
            return False
        path.unlink()
        return True


class SQLiteTraceStorage:
    """SQLite-based trace storage for production deployments.

    This storage backend uses SQLite to persist traces in a relational database.
    It provides better performance than file-based storage for large numbers of
    traces and supports efficient querying by session ID.

    For in-memory databases, a persistent connection is maintained to preserve
    the data across method calls. For file-based databases, new connections are
    created for each operation.
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        """Initialize SQLite storage with the specified database path.

        Args:
            db_path: Path to the SQLite database file, or ":memory:" for in-memory DB
        """
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._conn: sqlite3.Connection | None = None

        # For in-memory databases, keep a persistent connection
        if self._is_memory:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        Returns:
            A connection to the SQLite database
        """
        if self._is_memory and self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize the database schema.

        Creates the traces table and session_id index if they don't exist.
        """
        if self._is_memory and self._conn is not None:
            # For in-memory, use the persistent connection
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON traces(session_id)")
            self._conn.commit()
        else:
            # For file-based, use a temporary connection
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS traces (
                        trace_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        data TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON traces(session_id)")

    def save(self, trace: Trace) -> None:
        """Save a trace to the SQLite database.

        Args:
            trace: The trace to save
        """
        if self._is_memory and self._conn is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, session_id, data) VALUES (?, ?, ?)",
                (trace.trace_id, trace.session_id, trace.model_dump_json()),
            )
            self._conn.commit()
        else:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO traces (trace_id, session_id, data) VALUES (?, ?, ?)",
                    (trace.trace_id, trace.session_id, trace.model_dump_json()),
                )

    def load(self, trace_id: str) -> Trace | None:
        """Load a trace from the SQLite database.

        Args:
            trace_id: The unique identifier of the trace to load

        Returns:
            The loaded trace, or None if not found
        """
        if self._is_memory and self._conn is not None:
            row = self._conn.execute(
                "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            return Trace.model_validate_json(row[0])
        else:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
                ).fetchone()
                if row is None:
                    return None
                return Trace.model_validate_json(row[0])

    def list_traces(self, session_id: str) -> list[str]:
        """List trace IDs for a session from the SQLite database.

        Args:
            session_id: The session identifier to filter traces by

        Returns:
            List of trace IDs belonging to the session
        """
        if self._is_memory and self._conn is not None:
            rows = self._conn.execute(
                "SELECT trace_id FROM traces WHERE session_id = ?", (session_id,)
            ).fetchall()
            return [row[0] for row in rows]
        else:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT trace_id FROM traces WHERE session_id = ?", (session_id,)
                ).fetchall()
                return [row[0] for row in rows]

    def delete(self, trace_id: str) -> bool:
        """Delete a trace from the SQLite database.

        Args:
            trace_id: The unique identifier of the trace to delete

        Returns:
            True if the trace was deleted, False if it didn't exist
        """
        if self._is_memory and self._conn is not None:
            cursor = self._conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            self._conn.commit()
            return cursor.rowcount > 0
        else:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
                return cursor.rowcount > 0

    def close(self) -> None:
        """Close the persistent connection if it exists."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Close the persistent connection on deletion."""
        self.close()

    def __enter__(self) -> "SQLiteTraceStorage":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit - close connection."""
        self.close()


def get_storage(
    storage_type: str = "memory", storage_path: Path | str | None = None
) -> TraceStorage:
    """Factory function to create storage backends.

    Creates and returns a storage backend based on the specified type.
    Supports memory, file, and sqlite storage backends.

    Args:
        storage_type: Type of storage backend ("memory", "file", or "sqlite")
        storage_path: Path for file or sqlite storage (required for "file", optional for "sqlite")

    Returns:
        A storage backend implementing the TraceStorage protocol

    Raises:
        ValueError: If storage_type is unknown or required storage_path is missing
    """
    if storage_type == "memory":
        return MemoryTraceStorage()
    elif storage_type == "file":
        if storage_path is None:
            raise ValueError("storage_path required for file storage")
        return FileTraceStorage(Path(storage_path))
    elif storage_type == "sqlite":
        return SQLiteTraceStorage(storage_path or ":memory:")
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
