"""Caching utilities for Releso. To be used in SPOR steps."""

import json
import sqlite3


class RelesoSporCache:
    """A simple SQLite cache for storing SPOR results.

    This cache is designed for use in Releso's SPOR steps, allowing for
    efficient storage and retrieval of key-value pairs where keys are strings
    and values are JSON-serializable dictionaries. The cache is backed by an
    SQLite database, which is created if it does not already exist. It is designed
    to be used both in single as well as multi environment setups. In addition,
    it can allow caching between different training runs. As long as the key and
    value pair is consistent, the cache can be reused across different runs.

    It is advisable to only cache the input and output if the most computationally
    expensive part of the SPOR step and not cache the objective and reward since
    these are usually not expensive to compute and can change between runs.


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
        """Store a value in the cache.

        Args:
            key (str): The key to store the value under.
            value (dict): The value to store, must be a dictionary with keys matching
                          the example data keys.
        """
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

    def get(self, key: str) -> list[dict] | None:
        """Retrieve a value from the cache.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            list[dict] | None: The cached value as a list of dictionaries, or
            None if not found.
        """
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
