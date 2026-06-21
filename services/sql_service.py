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

PROCEDURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ---- ORDER / PRODUCTION ---------------------------------------------------
    "SP_MPAS_CREATED_ORDERS": {
        "description": "Returns active order information including status, material, stage progress, and bypass status",
        "category": "order",
        "params": ["@OrderNo", "@LineCode"],
        "optional_params": ["@LineCode"],
        "example_questions": [
            "What is the status of order 147190737?",
            "Show me order details for RHPW133610",
            "Is order 147190737 complete?",
            "What orders are running on line 3?",
        ],
    },

    # ---- PROCESS DATA --------------------------------------------------------
    "PNR_Get_updateProcessData_API_Raw_Data": {
        "description": "Returns process execution data — tool results, tightening data, stage pass/fail",
        "category": "process",
        "params": ["@OrderNo"],
        "optional_params": [],
        "example_questions": [
            "Show process results for order 147190737",
            "What stages passed for serial RHPW133610?",
            "Did order 147190737 pass all operations?",
            "Show me tightening results for this order",
        ],
    },

    # ---- QR / SCAN DATA ------------------------------------------------------
    "PNR_Get_updateQRV2_API_Raw_Data": {
        "description": "Returns QR scan data — part scans, material traceability, scan timestamps",
        "category": "quality",
        "params": ["@OrderNo"],
        "optional_params": [],
        "example_questions": [
            "What parts were scanned on order 147190737?",
            "Show material scan history for RHPW133610",
            "Were all parts scanned for order 147190737?",
            "Show QR scan log for this order",
        ],
    },

    # ---- HOURLY METRICS -------------------------------------------------------
    # NOTE: MPAS_HOURLY_PRODUCTIONMETRICS is updated in-place — use RowUpdateTime
    "SP_MPAS_HOURLY_PRODUCTION_METRICS": {
        "description": "Returns hourly production metrics per line — actual vs target, efficiency",
        "category": "production",
        "params": ["@LineCode", "@ShiftDate", "@ShiftID"],
        "optional_params": ["@ShiftDate", "@ShiftID"],
        "example_questions": [
            "How many units did line 3 produce this shift?",
            "What is the current efficiency on line 1?",
            "Show hourly output for shift A today",
            "How are we tracking against target?",
        ],
    },

    # ---- BOM / SAP -----------------------------------------------------------
    "SP_MPAS_SAP_ORDER_BOM": {
        "description": "Returns planned BOM from SAP_ORDER_BOM_V2 — planned parts vs scanned parts comparison",
        "category": "material",
        "params": ["@OrderNo"],
        "optional_params": [],
        "example_questions": [
            "What parts are on the BOM for order 147190737?",
            "Show SAP BOM vs actual scan for RHPW133610",
            "Are there any missing parts on order 147190737?",
            "What materials are planned for this order?",
        ],
    },

    # ---- BYPASS WORKFLOW -----------------------------------------------------
    "SP_MPAS_BYPASS_AUDIT": {
        "description": "Returns bypass request and approval audit trail — multi-step workflow history",
        "category": "bypass",
        "params": ["@OrderNo"],
        "optional_params": [],
        "example_questions": [
            "Was there a bypass on order 147190737?",
            "Show bypass history for RHPW133610",
            "Who approved the bypass for this order?",
            "Is there a pending bypass request?",
        ],
    },

    # ---- TORQUE / QUALITY ----------------------------------------------------
    "SP_MPAS_PROCESS_STATUS_TORQUE": {
        "description": "Returns torque data TorqueBolt1-38 from MPAS_PROCESS_STATUS for quality traceability",
        "category": "quality",
        "params": ["@OrderNo", "@SerialNo"],
        "optional_params": ["@SerialNo"],
        "example_questions": [
            "Show torque results for order 147190737",
            "What was the torque on bolt 5 for RHPW133610?",
            "Did all torque values pass for this order?",
            "Show quality traceability for order 147190737",
        ],
    },

    # ---- DEVIATION -----------------------------------------------------------
    "SP_MPAS_ORDER_DEVIATIONS": {
        "description": "Returns deviation records — DeviationStatus, DeviationComment, DeviationDoneBy",
        "category": "quality",
        "params": ["@OrderNo"],
        "optional_params": [],
        "example_questions": [
            "Are there any deviations on order 147190737?",
            "Show deviation log for RHPW133610",
            "Who raised the deviation on this order?",
            "What is the deviation status?",
        ],
    },
}

# Build a set of all approved procedure names for fast validation
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
                "category": meta["category"],
                "description": meta["description"],
                "params": meta["params"],
                "optional_params": meta.get("optional_params", []),
                "example_questions": meta["example_questions"],
            }
            for name, meta in PROCEDURE_REGISTRY.items()
        ]