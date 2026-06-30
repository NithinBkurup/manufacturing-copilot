"""
SQL Service – Manufacturing Copilot
Executes ONLY approved MPAS stored procedures.
Supports executing across multiple enabled SQL connections and merging results.
"""

import logging
from typing import Any, Dict, List, Optional
import pyodbc

from config.settings import settings

logger = logging.getLogger("copilot.sql")

PROCEDURE_REGISTRY: Dict[str, Dict[str, Any]] = {}
MASTER_TABLES_REGISTRY: Dict[str, Dict[str, Any]] = {}


def load_procedures_config():
    global APPROVED_PROCEDURES, PROCEDURE_REGISTRY
    import json
    import os
    config_path = "config/procedures.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    PROCEDURE_REGISTRY.clear()
                    PROCEDURE_REGISTRY.update(saved)
        except Exception:
            pass
    APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())


def load_master_tables_config():
    global MASTER_TABLES_REGISTRY
    import json
    import os
    config_path = "config/master_tables.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    MASTER_TABLES_REGISTRY.clear()
                    MASTER_TABLES_REGISTRY.update(saved)
        except Exception:
            pass


load_procedures_config()
APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())
load_master_tables_config()


class SQLService:
    """Executes approved MPAS stored procedures only. Read-only. No exceptions."""

    def __init__(self):
        self._conn_str = settings.sql_connection_string

    def _get_connection(self) -> pyodbc.Connection:
        return pyodbc.connect(self._conn_str, autocommit=False)

    def _validate_proc(self, proc_name: str) -> None:
        if proc_name not in APPROVED_PROCEDURES:
            raise PermissionError(
                f"Procedure '{proc_name}' is not in the approved registry. "
                "Register it in sql_service.PROCEDURE_REGISTRY before use."
            )

    def _rows_to_dicts(self, cursor: pyodbc.Cursor) -> List[Dict[str, Any]]:
        # Consume intermediate result sets (e.g. from temp tables or SET NOCOUNT OFF/ON side effects)
        while cursor.description is None:
            if not cursor.nextset():
                return []
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute_procedure(
        self,
        proc_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self._validate_proc(proc_name)

        # Ensure procedure name is wrapped in brackets for spaces/semi-colons
        safe_proc_name = proc_name
        if not safe_proc_name.startswith("["):
            if ";1" in safe_proc_name:
                base_name = safe_proc_name.split(";1")[0]
                safe_proc_name = f"[{base_name}];1"
            else:
                safe_proc_name = f"[{safe_proc_name}]"

        param_items = list((params or {}).items())
        if param_items:
            sql = f"EXEC {safe_proc_name} {', '.join(k + '=?' for k, _ in param_items)}"
            param_values = [v for _, v in param_items]
        else:
            sql = f"EXEC {safe_proc_name}"
            param_values = []

        logger.info("SQL EXEC: %s | params=%s", proc_name, {k: v for k, v in param_items})

        from services.connections_service import get_enabled_connections
        enabled_conns = get_enabled_connections("sql")

        # If no custom enabled connections, use settings default
        if not enabled_conns:
            try:
                with pyodbc.connect(self._conn_str, autocommit=False) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql, param_values)
                    rows = self._rows_to_dicts(cursor)
                    logger.info("SQL default returned %d rows", len(rows))
                    return rows
            except pyodbc.Error as exc:
                logger.error("SQL default error executing %s: %s", proc_name, exc)
                raise

        # Execute across all enabled connections and merge
        all_rows = []
        success_count = 0
        last_exception = None

        for db_conn in enabled_conns:
            config = db_conn.get("config", {})
            conn_str = (
                f"DRIVER={{{config.get('driver', settings.SQL_DRIVER)}}};"
                f"SERVER={config.get('server', settings.SQL_SERVER)};"
                f"DATABASE={config.get('database', settings.SQL_DATABASE)};"
                f"UID={config.get('username', settings.SQL_USERNAME)};"
                f"PWD={config.get('password', '')};"
                f"Connection Timeout={config.get('timeout', 30)};"
            )
            logger.info("SQL EXEC on custom database '%s' (%s)", db_conn.get("name"), config.get("database"))
            try:
                with pyodbc.connect(conn_str, autocommit=False) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql, param_values)
                    rows = self._rows_to_dicts(cursor)
                    logger.info("SQL custom db '%s' returned %d rows", db_conn.get("name"), len(rows))
                    all_rows.extend(rows)
                    success_count += 1
            except pyodbc.Error as exc:
                logger.warning("SQL execution failed on custom database '%s': %s", db_conn.get("name"), exc)
                last_exception = exc

        if success_count == 0 and last_exception:
            raise last_exception

        return all_rows

    def procedure_info(self, proc_name: str) -> Optional[Dict[str, Any]]:
        return PROCEDURE_REGISTRY.get(proc_name)

    def registry_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "procedure": name,
                "category": meta.get("category", "order"),
                "description": meta.get("description", ""),
                "intent": meta.get("intent", ""),
                "columns_to_consider": meta.get("columns_to_consider", ""),
                "params": meta.get("params", []),
                "optional_params": meta.get("optional_params", []),
                "example_questions": meta.get("example_questions", []),
                "keywords": meta.get("keywords", ""),
                "column_details": meta.get("column_details", []),
            }
            for name, meta in PROCEDURE_REGISTRY.items()
        ]

    def execute_master_table_query(self, table_name: str) -> List[Dict[str, Any]]:
        """Safe select all rows from an approved master table to load reference data."""
        import re
        if not re.match(r"^[a-zA-Z0-9_.]+$", table_name):
            raise ValueError("Invalid master table name format.")
        
        sql = f"SELECT * FROM {table_name}"
        logger.info("SQL SELECT MASTER: %s", table_name)
        
        try:
            with pyodbc.connect(self._conn_str, autocommit=False) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                return self._rows_to_dicts(cursor)
        except Exception as exc:
            logger.error("Error querying master table %s: %s", table_name, exc)
            return []