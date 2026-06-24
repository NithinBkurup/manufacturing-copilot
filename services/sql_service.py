"""
SQL Service – Manufacturing Copilot
Executes ONLY approved MPAS stored procedures via the saapi read-only account.
No direct table access. No INSERT/UPDATE/DELETE. Ever.

Approved procedures are the EXISTING MPAS procedures registered in the
Knowledge Administration Module. No new procedures are created.
"""

import logging
from typing import Any, Dict, List, Optional
import pyodbc

from config.settings import settings

logger = logging.getLogger("copilot.sql")


# ---------------------------------------------------------------------------
# Approved MPAS Stored Procedures Registry
# ---------------------------------------------------------------------------
# Key   : procedure name as it exists in MPAS_DB
# Value : metadata for the intent engine to reference
#
# HOW TO ADD A NEW PROCEDURE:
#   1. Confirm it exists in MPAS_DB and saapi has EXECUTE permission
#   2. Add it here with description, params, and example_questions
#   3. Register its parameter handler in manufacturing_service.py
#   NO code change needed anywhere else.
# ---------------------------------------------------------------------------

PROCEDURE_REGISTRY: Dict[str, Dict[str, Any]] = {}


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


load_procedures_config()
APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())


class SQLService:
    """Executes approved MPAS stored procedures only. Read-only. No exceptions."""

    def __init__(self):
        self._conn_str = settings.sql_connection_string

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> pyodbc.Connection:
        return pyodbc.connect(self._conn_str, autocommit=False)

    def _validate_proc(self, proc_name: str) -> None:
        if proc_name not in APPROVED_PROCEDURES:
            raise PermissionError(
                f"Procedure '{proc_name}' is not in the approved registry. "
                "Register it in sql_service.PROCEDURE_REGISTRY before use."
            )

    def _rows_to_dicts(self, cursor: pyodbc.Cursor) -> List[Dict[str, Any]]:
        if cursor.description is None:
            return []
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Public API — called only by manufacturing_service.py
    # ------------------------------------------------------------------

    def execute_procedure(
        self,
        proc_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute an approved stored procedure and return rows as list of dicts.
        Params dict keys must match @ParamName format (e.g. {"@OrderNo": "147190737"}).
        """
        self._validate_proc(proc_name)

        param_items = list((params or {}).items())
        param_placeholders = ", ".join("?" * len(param_items))

        if param_items:
            sql = f"EXEC {proc_name} {', '.join(k + '=?' for k, _ in param_items)}"
            param_values = [v for _, v in param_items]
        else:
            sql = f"EXEC {proc_name}"
            param_values = []

        logger.info("SQL EXEC: %s | params=%s", proc_name, {k: v for k, v in param_items})

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, param_values)
                rows = self._rows_to_dicts(cursor)
                logger.info("SQL returned %d rows", len(rows))
                return rows
        except pyodbc.Error as exc:
            logger.error("SQL error executing %s: %s", proc_name, exc)
            raise

    def procedure_info(self, proc_name: str) -> Optional[Dict[str, Any]]:
        """Return registry metadata for a procedure (used by intent engine)."""
        return PROCEDURE_REGISTRY.get(proc_name)

    def registry_summary(self) -> List[Dict[str, Any]]:
        """Return all registered procedures — for the Knowledge Admin Module."""
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
            }
            for name, meta in PROCEDURE_REGISTRY.items()
        ]