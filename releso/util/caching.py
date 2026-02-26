"""Caching utilities for Releso. To be used in SPOR steps."""

import json
import sqlite3

import numpy as np


class CachedValue:
    """A simple data structure to hold cached values."""

    def __init__(self, value: list[str], error: int | bool = False):
        self.value: list[str] = value
        self.error: int | bool = error

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index: int):
        """Default dunder functionality.

        Args:
            index (int): Index to retrieve from the cached value.

        Returns:
            Any: The requested item if found, otherwise raises an IndexError.
        """
        return self.value[index]

    def get(self, index: int | list[int]):
        """Retrieve items from the cached value.

        Args:
            index (int | list[int]): Index or list of indices to retrieve from the
            cached value.

        Returns:
            Any: The requested item if found.
        """
        if isinstance(index, list):
            return [self.value[i] for i in index]
        return self.value[index]

    def __bool__(self):
        raise RuntimeError(
            "CachedValue cannot be used in a boolean context. Use the 'error' attribute to check for errors."
        )

    def __iter__(self):
        return iter(self.value)


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

    Ensure that the key that you use a string that uniquely identifies the
    input parameters of the SPOR step. You can use the :py:meth:`make_cache_key`
    method to create a consistent key based on the input parameters. The function
    allows list and numpy array as inputs. The input will be flattened and rounded
    to a specified number of decimal places to ensure that the key is unique and
    consistent across runs and floating point arithmetic does not produce additional
    superfluous keys.

    Example usage:

    .. code-block:: python

        from releso.util.caching import RelesoSporCache

        def main(args, logger, func_data):
            # first call of function
            if func_data is None:
                # the data_keys and values are not the input parameters of the
                # function, but rather the data that you want to cache.
                func_data["cache"] = RelesoSporCache(
                    db_path="spor_cache.db",
                    example_data={"data_key1": "value1", "data_key2": "value2"},
                    # error_handling=True, # optional, if you want to store error information as well
                )
            if args.reset:
                # don't touch the cache, only reset the other func_data entries

            input_params = args.json_object["info"]["geometry_information"]
            key = RelesoSporCache.make_cache_key(input_params)
            cached_value = func_data["cache"].get(key)
            if cached_value is not None:
                if cached_value: # was an error thrown?
                    # an error is indicated for this result
                    pass
                # Use cached value
                pass
            else:
                # Perform expensive computation
                result = expensive_computation(input_params)
                # Store result in cache
                # The data you send here needs to be serializable by JSON.
                func_data["cache"].set(
                    key,
                    {"data_key1": result[0], "data_key2": result[1]},
                )
                # if error handling is enabled you can add the error information as well
                # func_data["cache"].set(
                #     key,
                #     {"data_key1": result[0], "data_key2": result[1]},
                #     error=error_occurred,
                # )


    You can use error handling to also store if an error occurred during the computation.
    This can be useful to avoid repeated attempts of expensive computations that are likely
    to fail. If error handling is enabled, you can set the "error" key in the value
    dictionary to indicate if an error occurred. When retrieving from the cache, you can
    check the "error" key to see if the cached value is valid or if it indicates a previous
    error.

    Params:
        db_path (str): Path to the SQLite database file.
            This file will be created if it does not exist.
            The cache will store key-value pairs where keys are strings and values are
            JSON-serializable dictionaries.
        example_data (dict): An example dictionary that defines the structure of the
            values to be stored in the cache. The keys of this dictionary will be used
            as the columns in the SQLite table.
        error_handling (bool): If True, caching will also store and retrieve info if an
            error occurred. Default is False.
    """

    @staticmethod
    def make_cache_key(
        key_array: np.ndarray | list[list[float]] | list[float],
        rounding: int = 6,
    ) -> str:
        """Create a cache key based on the provided arguments.

        This function was initially generated by AI.

        Args:
            key_array (np.ndarray | list[list[float]]): Array or list of control
                point y-coordinates to be used as a key for caching.
            rounding (int): Number of decimal places to round the values in the key.
                Default is 6.
        Returns:
            str: A JSON-serialized string that serves as a unique key for caching.
        """
        # Convert args and kwargs to a JSON-serializable string
        key_array = np.asarray(key_array)
        normalized_args = key_array.flatten().round(rounding).tolist()
        key = json.dumps(normalized_args, sort_keys=True)
        return key

    def __init__(
        self, db_path: str, example_data: dict, error_handling: bool = False
    ):
        self.db_path = db_path
        self.error_handling = error_handling

        self.keys = list(example_data.keys())
        self.value_accessor = ", ".join(f"{key}" for key in self.keys)
        self.value_type = ["TEXT" for _ in self.keys]
        self.value_question_mark = ", ".join("?" for _ in self.keys)
        if error_handling:
            self.keys.append("error")
            self.value_accessor += ", error"
            self.value_type.append("INT")
            self.value_question_mark += ", ?"

        self._initialize_db()

    def _initialize_db(self):
        """Initialize the SQLite database and create the cache table if it doesn't exist."""
        table_definitions = ", ".join(
            f"{key} {type_}" for key, type_ in zip(self.keys, self.value_type)
        )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS spor_cache (key TEXT PRIMARY KEY, {table_definitions})"
            )
            conn.commit()

    def _convert_value_to_database_values(self, value: dict) -> list[any]:
        """Convert a value dictionary to a list of database values.

        Args:
            value (dict): The value dictionary to convert.

        Returns:
            list[any]: A list of values corresponding to the database columns.
        """
        ret_values = [
            json.dumps(value[key]) for key in self.keys if key != "error"
        ]
        if self.error_handling:
            ret_values.append(value.get("error", value.get("error", 0)))
        return ret_values

    def set(self, key: str, value: dict, error: bool | int = False):
        """Store a value in the cache.

        If error handling is enabled, this method will also ensure that the "error" key is present
        in the value dictionary. If the value is None and an error is indicated, it will create a
        new dictionary with all keys set to -1 except for the "error" key, which will be set to
        -1. This allows for storing error information even when the actual value is not available.

        Args:
            key (str): The key to store the value under.
            value (dict): The value to store, must be a dictionary with keys matching
                          the example data keys.
            error (bool | int): Indicates if an error occurred during computation. Default is False.
        """
        if self.error_handling:
            # if error handling is enabled, we need to ensure that the "error" key is present in the value dictionary
            # if the value is None, we will create a new dictionary with the "error" key set to the error value and all
            # other keys set to -1. This allows us to store error information even when the actual value is not available.
            if value is None:
                if bool(error):
                    value = {}
                    for k in self.keys:
                        if k != "error":
                            value[k] = -1
                else:
                    raise ValueError(
                        f"Value must be a dictionary with keys: {self.keys}"
                    )
            # add error value to the value dictionary if it is not already present
            if value.get("error") is None:
                value["error"] = int(error)
        # check if all values are present and if the keys match the example data keys
        if not isinstance(value, dict) or list(value.keys()) != self.keys:
            raise ValueError(
                f"Value must be a dictionary with keys: {self.keys}, but has keys: {list(value.keys())}"
            )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT OR REPLACE INTO spor_cache (key, {self.value_accessor}) VALUES (?, {self.value_question_mark})",
                (key, *self._convert_value_to_database_values(value)),
            )
            conn.commit()

    def _cached_value_from_row(self, row) -> CachedValue:
        """Convert a database row to a CachedValue instance.

        Args:
            row (tuple): A tuple representing a row from the database.

        Returns:
            CachedValue: An instance of CachedValue containing the cached data.
        """
        print(row)
        if self.error_handling:
            value = [json.loads(r) for r in row[:-1]]
        else:
            value = [json.loads(r) for r in row]
        error_value = row[-1] if self.error_handling else False
        return CachedValue(value=value, error=error_value)

    def get(self, key: str) -> CachedValue | None:
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
                    f"SELECT {self.value_accessor} FROM spor_cache WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                return self._cached_value_from_row(row) if row else None
        except sqlite3.OperationalError:
            return None
