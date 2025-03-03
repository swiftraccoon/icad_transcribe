import logging
import os
import sqlite3
import traceback
from sqlite3 import Connection
from contextlib import contextmanager

import bcrypt
module_logger = logging.getLogger('icad_transcribe.sqlite_module')


class SQLiteDatabase:
    """
    SQLite wrapper that returns structured responses.

    This class provides convenience methods to create or connect to a SQLite database,
    automatically apply an initial schema, and perform common queries with a consistent
    return structure.
    """

    def __init__(self):
        """
        Initialize the SQLite database. If the file doesn't exist, create it and
        build an initial schema, plus an example admin user.

        :raises ValueError:
            If the `db_path` is empty or not a valid string.
        :raises IsADirectoryError:
            If the provided `db_path` points to a directory rather than a file.
        :raises Exception:
            For unexpected errors during the creation of the database or the admin user.
        """
        db_path = os.getenv('SQLITE_DATABASE_PATH', 1)

        # Validate input
        if db_path is None or not isinstance(db_path, str) or not db_path.strip():
            raise ValueError("Invalid database path provided (empty or not a string).")

        if os.path.isdir(db_path):
            raise IsADirectoryError(f"Provided path '{db_path}' is a directory, not a file.")

        self.db_path = db_path

        if not os.path.exists(self.db_path):
            module_logger.warning(f"Database file not found at '{self.db_path}'. Will create a new database.")
            try:
                self._create_database()
                self._create_admin_user()
                self._create_default_config()
                module_logger.info("Database created and admin user added successfully.")
            except Exception as e:
                traceback.print_exc()
                raise

    @contextmanager
    def _get_connection(self) -> Connection:
        """
        Provide a context-managed connection to the SQLite database.

        :return:
            A context manager that yields a `sqlite3.Connection` object.
            The connection is automatically closed upon exiting the context block.

        :rtype: sqlite3.Connection

        :raises sqlite3.Error:
            If there is a failure to open the database file or other SQLite issues.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        finally:
            if conn:
                conn.close()

    def _create_database(self) -> None:
        """
        Create the initial schema for the database by executing `sql_schema`.

        :raises sqlite3.Error:
            If there is a problem executing the schema (e.g., syntax error in `sql_schema`).
        """
        with open('init_db/transcribe_db.sql', 'r') as f:
            schema = f.read()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(schema)
            cursor.close()

    def _create_admin_user(self) -> None:
        """
        Create a default admin user.

        :raises sqlite3.Error:
            If insertion of the admin user fails.
        """
        password_hash = bcrypt.hashpw("admin".encode(), bcrypt.gensalt())
        query = "INSERT INTO users (user_username, user_password) VALUES (?, ?)"
        params = ("admin", password_hash)
        resp = self.execute_commit(query, params)
        if not resp["success"]:
            raise sqlite3.Error(resp["message"])

    def _create_default_config(self) -> None:
        """
        Create a default app transcribe config.

        :raises sqlite3.Error:
            If insertion of the admin user fails.
        """
        insert_query = "INSERT INTO transcribe_config (transcribe_config_name) VALUES (%s)"
        insert_params = ("Default Config",)
        resp = self.execute_commit(insert_query, insert_params, return_row_id=True)
        if not resp["success"]:
            raise sqlite3.Error(f"Failed adding default transcribe config {resp['message']}")

        vad_insert_query = "INSERT INTO vad_config (transcribe_config_id) VALUES (%s)"
        vad_insert_params = (resp.get("result"),)
        vad_insert_resp = self.execute_commit(vad_insert_query, vad_insert_params)
        if not vad_insert_resp["success"]:
            raise sqlite3.Error(f"Failed adding default vad config {resp['message']}")

        amplify_insert_query = "INSERT INTO amplify_config (transcribe_config_id) VALUES (%s)"
        amplify_insert_params = (resp.get("result"),)
        amplify_insert_resp = self.execute_commit(amplify_insert_query, amplify_insert_params)
        if not amplify_insert_resp["success"]:
            raise sqlite3.Error(f"Failed adding default amplify config {resp['message']}")

        tone_removal_insert_query = "INSERT INTO tone_removal_config (transcribe_config_id) VALUES (%s)"
        tone_removal_params = (resp.get("result"),)
        tone_removal_resp = self.execute_commit(tone_removal_insert_query, tone_removal_params)
        if not tone_removal_resp["success"]:
            raise sqlite3.Error(f"Failed adding default tone removal config {resp['message']}")


    def execute_query(self, query: str, params=None, fetch_mode: str = "all") -> dict:
        """
        Execute a SELECT-like query, returning data in a structured dictionary.

        :param query:
            The SQL query string to execute. Uses '?' placeholders for parameters.
        :type query: str

        :param params:
            Optional parameters to bind to the query. Could be a tuple or list.
            If None, the query is executed as-is.
        :type params: tuple or list, optional

        :param fetch_mode:
            Determines how results are fetched:
            - "all": fetch all rows as a list of dictionaries.
            - "many": fetch a limited chunk of rows (defaults to the driver’s default chunk size).
            - "one": fetch a single row.
        :type fetch_mode: str

        :return:
            A dictionary with the following structure:
            ::
                {
                  "success": bool,
                  "message": str,
                  "result": <list of dicts> or <dict>
                }

            If `fetch_mode == "one"`, `result` is a single dictionary (or empty dict if no row).
            Otherwise, `result` is a list of dictionaries.
        :rtype: dict

        :raises ValueError:
            If `fetch_mode` is invalid.
        """
        if "%s" in query:
            query = query.replace("%s", "?")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                column_names = [desc[0] for desc in cursor.description] if cursor.description else []

                if fetch_mode == "all":
                    rows = cursor.fetchall()
                    result = [self._row_to_dict(row, column_names) for row in rows]

                elif fetch_mode == "many":
                    rows = cursor.fetchmany()
                    result = [self._row_to_dict(row, column_names) for row in rows]

                elif fetch_mode == "one":
                    row = cursor.fetchone()
                    result = self._row_to_dict(row, column_names) if row else {}

                else:
                    return {
                        "success": False,
                        "message": f"Invalid fetch_mode: {fetch_mode}",
                        "result": []
                    }

                return {
                    "success": True,
                    "message": "SQLite Query executed successfully",
                    "result": result
                }

            except sqlite3.Error as e:
                module_logger.error(f"SQLite Query Failure: {e} | Query: {query} | Params: {params}")
                return {
                    "success": False,
                    "message": str(e),
                    "result": []
                }
            finally:
                cursor.close()

    def execute_commit(self,
                       query: str,
                       params=None,
                       return_row_id: bool = True,
                       return_count: bool = False) -> dict:
        """
        Execute an INSERT/UPDATE/DELETE query (write) and commit changes,
        returning a structured response.

        :param query:
            The SQL query string to execute. Uses '?' placeholders for parameters.
        :type query: str

        :param params:
            Optional parameters to bind to the query. Could be a tuple or list.
            If None, the query is executed as-is.
        :type params: tuple or list, optional

        :param return_row_id:
            If True, the result dictionary includes the `lastrowid` of the operation.
            If False, `result` is an empty list unless `return_count` is True.
        :type return_row_id: bool

        :param return_count:
            If True, the result dictionary includes the number of affected rows.
            Ignored if `return_row_id` is True.
        :type return_count: bool

        :return:
            A dictionary in the format:
            ::
                {
                    "success": bool,
                    "message": str,
                    "result": <int or list>
                }

            Where `result` may be:
            - last inserted row ID (if `return_row_id` is True),
            - number of affected rows (if `return_count` is True),
            - or an empty list.
        :rtype: dict

        :raises sqlite3.Error:
            If there is an issue executing or committing the query.
        """
        if "%s" in query:
            query = query.replace("%s", "?")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                row_id = cursor.lastrowid
                affected = cursor.rowcount
                conn.commit()

                if return_row_id:
                    res = row_id
                elif return_count:
                    res = affected
                else:
                    res = []

                module_logger.debug(f"SQLite Commit: | Query: {query} | Params: {params}")

                return {
                    "success": True,
                    "message": "SQLite Commit Query executed successfully",
                    "result": res
                }

            except sqlite3.Error as e:
                module_logger.error(f"SQLite Commit Failure: {e} | Query: {query} | Params: {params}")
                conn.rollback()
                return {
                    "success": False,
                    "message": str(e),
                    "result": []
                }
            finally:
                cursor.close()

    def execute_many_commit(self,
                            query: str,
                            data: list,
                            batch_size: int = 1000) -> dict:
        """
        Execute multiple INSERT/UPDATE/DELETE statements in batches (similar to MySQL),
        returning a structured response.

        :param query:
            The SQL query string to execute, containing '?' placeholders.
        :type query: str

        :param data:
            A list of parameter tuples to be executed in batches.
        :type data: list

        :param batch_size:
            Number of rows to process in each batch commit. Defaults to 1000.
        :type batch_size: int

        :return:
            A dictionary of the form:
            ::
                {
                    "success": bool,
                    "message": str,
                    "result": []
                }
        :rtype: dict

        :raises sqlite3.Error:
            If there's a problem executing any batch or committing the transaction.
        """
        if "%s" in query:
            query = query.replace("%s", "?")

        if not data:
            module_logger.warning("No data provided for batch execution.")
            return {
                "success": False,
                "message": "No data provided for batch execution.",
                "result": []
            }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                total_rows = len(data)
                total_batches = (total_rows + batch_size - 1) // batch_size

                for batch_num in range(total_batches):
                    start_index = batch_num * batch_size
                    batch_data = data[start_index:start_index + batch_size]
                    cursor.executemany(query, batch_data)
                    conn.commit()

                return {
                    "success": True,
                    "message": "SQLite Multi-Commit Query executed successfully",
                    "result": []
                }
            except sqlite3.Error as e:
                module_logger.debug(f"SQLite Multi-Commit Failure: {e} | Query: {query} | Data: {data}")
                conn.rollback()
                return {
                    "success": False,
                    "message": str(e),
                    "result": []
                }
            finally:
                cursor.close()

    @staticmethod
    def _row_to_dict(row, column_names) -> dict:
        """
        Convert a single database row (tuple) into a dictionary using the provided column names.

        :param row:
            A row tuple returned by a database cursor. If None, an empty dict is returned.
        :type row: tuple or None

        :param column_names:
            The list of column names corresponding to the row's columns.
        :type column_names: list

        :return:
            A dictionary mapping column names to values. If `row` is None, returns an empty dict.
        :rtype: dict
        """
        if not row:
            return {}
        row_dict = {}
        for i, val in enumerate(row):
            row_dict[column_names[i]] = val
        return row_dict
