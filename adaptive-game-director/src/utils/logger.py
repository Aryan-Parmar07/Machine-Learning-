"""Logging and telemetry recording utilities.

Provides:
- Standardized project logger configuration.
- TelemetryRecorder for stream-writing gameplay data to CSV.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_logger(name: str = "GameDirector", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger.

    Args:
        name: Name of the logger instance.
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger instance configured with consistent formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class TelemetryRecorder:
    """Handles real-time streaming of telemetry observations to CSV.

    Uses Python's standard library `csv` module to guarantee zero heavy
    dependencies during human play recording sessions.

    Attributes:
        filepath: Path to the target CSV file.
        fieldnames: List of column header names.
        buffer_flush_frequency: How often to flush writes to disk (in rows).
    """

    def __init__(
        self,
        filepath: Path,
        fieldnames: Optional[List[str]] = None,
        buffer_flush_frequency: int = 10,
    ) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self.buffer_flush_frequency = buffer_flush_frequency
        self._file = None
        self._writer = None
        self._row_count = 0
        self._is_header_written = False

    def open(self, mode: str = "w") -> None:
        """Opens the CSV file for writing."""
        self._file = open(self.filepath, mode=mode, newline="", encoding="utf-8")
        if self.fieldnames:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            if mode == "w" or self.filepath.stat().st_size == 0:
                self._writer.writeheader()
                self._is_header_written = True
                self._file.flush()

    def record_row(self, row: Dict[str, Any]) -> None:
        """Records a single telemetry observation.

        Args:
            row: Dictionary mapping column names to telemetry values.
        """
        if self._file is None:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            self.open(mode="w")

        # Fallback if headers were not explicitly provided upfront
        if not self._is_header_written and self._writer is None:
            self.fieldnames = list(row.keys())
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            self._writer.writeheader()
            self._is_header_written = True

        self._writer.writerow(row)
        self._row_count += 1

        if self._row_count % self.buffer_flush_frequency == 0:
            self.flush()

    def flush(self) -> None:
        """Flushes written rows to persistent storage."""
        if self._file and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        """Closes the underlying CSV file safely."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    @property
    def row_count(self) -> int:
        """Returns total recorded rows in this session."""
        return self._row_count

    def __enter__(self) -> "TelemetryRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
