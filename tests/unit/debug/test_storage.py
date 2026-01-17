"""Tests for trace storage backends.

This module tests all trace storage implementations including in-memory,
file-based, and SQLite storage backends.
"""

from datetime import datetime
from pathlib import Path

import pytest

from reasoning_mcp.debug.storage import (
    FileTraceStorage,
    MemoryTraceStorage,
    SQLiteTraceStorage,
    get_storage,
)
from reasoning_mcp.models.debug import SpanStatus, Trace, TraceSpan


@pytest.fixture
def sample_trace() -> Trace:
    """Create a sample trace for testing.

    Returns:
        A sample Trace instance with a root span
    """
    root_span = TraceSpan(
        span_id="span-1",
        parent_id=None,
        name="root",
        start_time=datetime.now(),
        end_time=datetime.now(),
        status=SpanStatus.COMPLETED,
        attributes={},
    )
    return Trace(
        trace_id="trace-1",
        session_id="session-1",
        root_span=root_span,
        spans=[root_span],
        steps=[],
        decisions=[],
        errors=[],
    )


class TestMemoryStorage:
    """Test suite for MemoryTraceStorage."""

    def test_save_and_load(self, sample_trace: Trace) -> None:
        """Test saving and loading a trace from memory storage."""
        storage = MemoryTraceStorage()
        storage.save(sample_trace)
        loaded = storage.load(sample_trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == sample_trace.trace_id

    def test_load_nonexistent(self) -> None:
        """Test loading a nonexistent trace returns None."""
        storage = MemoryTraceStorage()
        assert storage.load("nonexistent") is None

    def test_list_traces(self, sample_trace: Trace) -> None:
        """Test listing traces by session ID."""
        storage = MemoryTraceStorage()
        storage.save(sample_trace)
        traces = storage.list_traces(sample_trace.session_id)
        assert sample_trace.trace_id in traces

    def test_list_traces_empty_session(self) -> None:
        """Test listing traces for a session with no traces."""
        storage = MemoryTraceStorage()
        assert storage.list_traces("nonexistent") == []

    def test_delete(self, sample_trace: Trace) -> None:
        """Test deleting a trace from memory storage."""
        storage = MemoryTraceStorage()
        storage.save(sample_trace)
        assert storage.delete(sample_trace.trace_id)
        assert storage.load(sample_trace.trace_id) is None

    def test_delete_nonexistent(self) -> None:
        """Test deleting a nonexistent trace returns False."""
        storage = MemoryTraceStorage()
        assert not storage.delete("nonexistent")

    def test_save_multiple_traces_same_session(self, sample_trace: Trace) -> None:
        """Test saving multiple traces for the same session."""
        storage = MemoryTraceStorage()

        # Save first trace
        storage.save(sample_trace)

        # Create and save second trace with same session
        root_span_2 = TraceSpan(
            span_id="span-2",
            parent_id=None,
            name="root2",
            start_time=datetime.now(),
            end_time=datetime.now(),
            status=SpanStatus.COMPLETED,
            attributes={},
        )
        trace_2 = Trace(
            trace_id="trace-2",
            session_id=sample_trace.session_id,
            root_span=root_span_2,
            spans=[root_span_2],
            steps=[],
            decisions=[],
            errors=[],
        )
        storage.save(trace_2)

        # Verify both traces are listed for the session
        traces = storage.list_traces(sample_trace.session_id)
        assert len(traces) == 2
        assert sample_trace.trace_id in traces
        assert trace_2.trace_id in traces

    def test_save_duplicate_trace_id(self, sample_trace: Trace) -> None:
        """Test that saving a trace with duplicate ID overwrites the existing one."""
        storage = MemoryTraceStorage()
        storage.save(sample_trace)

        # Create a new trace with same ID but different data
        root_span_modified = TraceSpan(
            span_id="span-modified",
            parent_id=None,
            name="root_modified",
            start_time=datetime.now(),
            end_time=datetime.now(),
            status=SpanStatus.COMPLETED,
            attributes={"modified": True},
        )
        modified_trace = Trace(
            trace_id=sample_trace.trace_id,  # Same ID
            session_id="session-modified",
            root_span=root_span_modified,
            spans=[root_span_modified],
            steps=[],
            decisions=[],
            errors=[],
        )
        storage.save(modified_trace)

        # Verify the modified trace replaced the original
        loaded = storage.load(sample_trace.trace_id)
        assert loaded is not None
        assert loaded.session_id == "session-modified"
        assert loaded.root_span.name == "root_modified"


class TestFileStorage:
    """Test suite for FileTraceStorage."""

    def test_save_and_load(self, sample_trace: Trace, tmp_path: Path) -> None:
        """Test saving and loading a trace from file storage."""
        storage = FileTraceStorage(tmp_path)
        storage.save(sample_trace)
        loaded = storage.load(sample_trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == sample_trace.trace_id

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading a nonexistent trace returns None."""
        storage = FileTraceStorage(tmp_path)
        assert storage.load("nonexistent") is None

    def test_delete(self, sample_trace: Trace, tmp_path: Path) -> None:
        """Test deleting a trace from file storage."""
        storage = FileTraceStorage(tmp_path)
        storage.save(sample_trace)
        assert storage.delete(sample_trace.trace_id)
        assert storage.load(sample_trace.trace_id) is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting a nonexistent trace returns False."""
        storage = FileTraceStorage(tmp_path)
        assert not storage.delete("nonexistent")

    def test_list_traces(self, sample_trace: Trace, tmp_path: Path) -> None:
        """Test listing traces by session ID from file storage."""
        storage = FileTraceStorage(tmp_path)
        storage.save(sample_trace)
        traces = storage.list_traces(sample_trace.session_id)
        assert sample_trace.trace_id in traces

    def test_list_traces_filters_by_session(self, sample_trace: Trace, tmp_path: Path) -> None:
        """Test that list_traces only returns traces for the specified session."""
        storage = FileTraceStorage(tmp_path)

        # Save trace for session-1
        storage.save(sample_trace)

        # Create and save trace for different session
        root_span_2 = TraceSpan(
            span_id="span-2",
            parent_id=None,
            name="root2",
            start_time=datetime.now(),
            end_time=datetime.now(),
            status=SpanStatus.COMPLETED,
            attributes={},
        )
        trace_2 = Trace(
            trace_id="trace-2",
            session_id="session-2",
            root_span=root_span_2,
            spans=[root_span_2],
            steps=[],
            decisions=[],
            errors=[],
        )
        storage.save(trace_2)

        # Verify only session-1 trace is returned
        traces = storage.list_traces("session-1")
        assert len(traces) == 1
        assert sample_trace.trace_id in traces
        assert trace_2.trace_id not in traces

    def test_directory_creation(self, tmp_path: Path) -> None:
        """Test that FileTraceStorage creates the directory if it doesn't exist."""
        storage_dir = tmp_path / "nested" / "storage" / "dir"
        assert not storage_dir.exists()

        _ = FileTraceStorage(storage_dir)
        assert storage_dir.exists()
        assert storage_dir.is_dir()

    def test_list_traces_ignores_invalid_files(self, sample_trace: Trace, tmp_path: Path) -> None:
        """Test that list_traces ignores invalid JSON files."""
        storage = FileTraceStorage(tmp_path)
        storage.save(sample_trace)

        # Create an invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json")

        # Should still return the valid trace
        traces = storage.list_traces(sample_trace.session_id)
        assert sample_trace.trace_id in traces


class TestSQLiteStorage:
    """Test suite for SQLiteTraceStorage."""

    def test_save_and_load(self, sample_trace: Trace) -> None:
        """Test saving and loading a trace from SQLite storage."""
        with SQLiteTraceStorage() as storage:  # in-memory
            storage.save(sample_trace)
            loaded = storage.load(sample_trace.trace_id)
            assert loaded is not None
            assert loaded.trace_id == sample_trace.trace_id

    def test_load_nonexistent(self) -> None:
        """Test loading a nonexistent trace returns None."""
        with SQLiteTraceStorage() as storage:
            assert storage.load("nonexistent") is None

    def test_list_traces(self, sample_trace: Trace) -> None:
        """Test listing traces by session ID from SQLite storage."""
        with SQLiteTraceStorage() as storage:
            storage.save(sample_trace)
            traces = storage.list_traces(sample_trace.session_id)
            assert sample_trace.trace_id in traces

    def test_delete(self, sample_trace: Trace) -> None:
        """Test deleting a trace from SQLite storage."""
        with SQLiteTraceStorage() as storage:
            storage.save(sample_trace)
            assert storage.delete(sample_trace.trace_id)
            assert storage.load(sample_trace.trace_id) is None

    def test_delete_nonexistent(self) -> None:
        """Test deleting a nonexistent trace returns False."""
        with SQLiteTraceStorage() as storage:
            assert not storage.delete("nonexistent")

    def test_file_based_sqlite(self, sample_trace: Trace, tmp_path: Path) -> None:
        """Test SQLite storage with a file-based database."""
        db_path = tmp_path / "traces.db"
        storage = SQLiteTraceStorage(db_path)
        storage.save(sample_trace)

        # Verify the database file was created
        assert db_path.exists()

        # Create a new storage instance with the same file
        storage2 = SQLiteTraceStorage(db_path)
        loaded = storage2.load(sample_trace.trace_id)
        assert loaded is not None
        assert loaded.trace_id == sample_trace.trace_id

    def test_list_traces_filters_by_session(self, sample_trace: Trace) -> None:
        """Test that list_traces only returns traces for the specified session."""
        with SQLiteTraceStorage() as storage:
            # Save trace for session-1
            storage.save(sample_trace)

            # Create and save trace for different session
            root_span_2 = TraceSpan(
                span_id="span-2",
                parent_id=None,
                name="root2",
                start_time=datetime.now(),
                end_time=datetime.now(),
                status=SpanStatus.COMPLETED,
                attributes={},
            )
            trace_2 = Trace(
                trace_id="trace-2",
                session_id="session-2",
                root_span=root_span_2,
                spans=[root_span_2],
                steps=[],
                decisions=[],
                errors=[],
            )
            storage.save(trace_2)

            # Verify only session-1 trace is returned
            traces = storage.list_traces("session-1")
            assert len(traces) == 1
            assert sample_trace.trace_id in traces
            assert trace_2.trace_id not in traces

    def test_save_overwrites_existing(self, sample_trace: Trace) -> None:
        """Test that saving a trace with the same ID overwrites the existing one."""
        with SQLiteTraceStorage() as storage:
            storage.save(sample_trace)

            # Create modified trace with same ID
            root_span_modified = TraceSpan(
                span_id="span-modified",
                parent_id=None,
                name="root_modified",
                start_time=datetime.now(),
                end_time=datetime.now(),
                status=SpanStatus.COMPLETED,
                attributes={"modified": True},
            )
            modified_trace = Trace(
                trace_id=sample_trace.trace_id,  # Same ID
                session_id="session-modified",
                root_span=root_span_modified,
                spans=[root_span_modified],
                steps=[],
                decisions=[],
                errors=[],
            )
            storage.save(modified_trace)

            # Verify the modified trace replaced the original
            loaded = storage.load(sample_trace.trace_id)
            assert loaded is not None
            assert loaded.session_id == "session-modified"
            assert loaded.root_span.name == "root_modified"

    def test_context_manager_closes_connection(self, sample_trace: Trace) -> None:
        """Test that using SQLiteTraceStorage as a context manager properly closes the connection."""
        storage = SQLiteTraceStorage()
        storage.save(sample_trace)

        # Verify the connection is open
        assert storage._conn is not None

        # Use close method
        storage.close()

        # Verify the connection is closed
        assert storage._conn is None


class TestGetStorage:
    """Test suite for the storage factory function."""

    def test_memory_storage(self) -> None:
        """Test creating a memory storage backend."""
        storage = get_storage("memory")
        assert isinstance(storage, MemoryTraceStorage)

    def test_file_storage(self, tmp_path: Path) -> None:
        """Test creating a file storage backend."""
        storage = get_storage("file", tmp_path)
        assert isinstance(storage, FileTraceStorage)

    def test_sqlite_storage(self) -> None:
        """Test creating an in-memory SQLite storage backend."""
        storage = get_storage("sqlite")
        assert isinstance(storage, SQLiteTraceStorage)

    def test_sqlite_storage_with_path(self, tmp_path: Path) -> None:
        """Test creating a file-based SQLite storage backend."""
        db_path = tmp_path / "test.db"
        storage = get_storage("sqlite", db_path)
        assert isinstance(storage, SQLiteTraceStorage)

    def test_file_storage_requires_path(self) -> None:
        """Test that file storage raises ValueError if path is not provided."""
        with pytest.raises(ValueError, match="storage_path required for file storage"):
            get_storage("file")

    def test_unknown_type(self) -> None:
        """Test that unknown storage type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown storage type: unknown"):
            get_storage("unknown")
