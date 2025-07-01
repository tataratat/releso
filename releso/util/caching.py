"""Caching utilities for Releso. To be used in SPOR steps."""

import json
import sqlite3


class RelesoSporCache:
    """A simple SQLite cache for storing SPOR results.

    Params:
        db_path (str): Path to the SQLite database file.
        This file will be created if it does not exist.
        The cache will store key-value pairs where keys are strings and values are JSON-serializable
        dictionaries.
    """

    def __init__(self, db_path: str, example_data: dict):
        self.db_path = db_path

        self.keys = example_data.keys()
        self.value_accessor = ", ".join(f"{key}" for key in self.keys)
        self.value_question_mark = ", ".join("?" for _ in self.keys)

        self._initialize_db()

    def _initialize_db(self):
        """Initialize the SQLite database and create the cache table if it doesn't exist."""
        table_definitions = ", ".join(f"{key} TEXT" for key in self.keys)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS spor_cache (key TEXT PRIMARY KEY, {table_definitions})"
            )
            conn.commit()

    def set(self, key: str, value: dict):
        """Store a value in the cache."""
        if not isinstance(value, dict):
            if value.keys() != self.keys:
                raise ValueError(
                    f"Value must be a dictionary with keys: {self.keys}"
                )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT OR REPLACE INTO spor_cache (key, {self.value_accessor}) VALUES (?, {self.value_question_mark})",
                (key, *[json.dumps(value[key]) for key in self.keys]),
            )
            conn.commit()

    def get(self, key: str) -> dict:
        """Retrieve a value from the cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {self.value_accessor} FROM spor_cache WHERE key = '{key}'"
                )
                rows = cursor.fetchone()
                return [json.loads(row) for row in rows] if rows else None
        except sqlite3.OperationalError:
            return None
