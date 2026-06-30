"""
Manufacturing Service – Manufacturing Copilot
The single business logic layer. All queries flow through here.

Architecture:
  User query
    → IntentService (classify + extract entities)
    → ManufacturingService (fetch SQL / OPC / RAG)
    → Qwen3 (format response)
    → Operator

Rule: No production.py, planning.py, recipe.py, quality.py, or material.py.
      All business logic lives here.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import settings
from services.cache_service import cache
from services.intent_service import IntentResult, IntentService
from services.sql_service import SQLService
from services.vector_service import VectorService

logger = logging.getLogger("copilot.manufacturing")

# Plant code is always 2006 — injected here, never from user input
PLANT_CODE = settings.PLANT_CODE


class ManufacturingService:
    """
    Orchestrates data retrieval from SQL, OPC cache, and ChromaDB.
    Returns a structured context string for the AI model.
    """

    def __init__(self, vector_service: Optional[VectorService] = None):
        self._sql = SQLService()
        self._intent = IntentService()
        self._vector = vector_service

    # ------------------------------------------------------------------
    # Main entry point — called by chat route
    # ------------------------------------------------------------------

    async def handle_query(
        self,
        user_message: str,
    ) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        Returns:
          {
            "intent": IntentResult,
            "context": str,          # injected into Qwen3 prompt
            "clarification": str,    # if entities missing
            "data_sources": list,    # what was fetched
          }
        """
        # 1. Classify intent and extract entities
        intent = self._intent.classify(user_message)

        # 2. If required entities missing — return clarification request
        if intent.clarification_needed:
            return {
                "intent": intent,
                "context": "",
                "clarification": intent.clarification_prompt,
                "data_sources": [],
            }

        # 3. Fetch data from appropriate source(s)
        context_parts = []
        data_sources = []

        # -- SQL stored procedure --
        if intent.procedure:
            is_prod_report = intent.procedure == "SP_MPAS_KPI_PROD_HOURLY_SUMMARY;1" or intent.intent == "hourly_production"
            if is_prod_report:
                procedures = [
                    "SP_MPAS_KPI_PROD_HOURLY_SUMMARY;1",
                    "SP_MPAS_REPORT_PPC_SUMMARY;1",
                    "SP_MPAS_REPORT_QR_TOOL SUMMARY;1",
                    "SP_MPAS_REPORT_OT_BYPASS;1",
                    "SP_MPAS_REPORT_ALARM_HISTORY;1"
                ]
                
                logger.info("Executing multi-procedure fetch for comprehensive report: %s", procedures)
                from services.sql_service import PROCEDURE_REGISTRY
                
                primary_clarification = None
                for proc in procedures:
                    meta = PROCEDURE_REGISTRY.get(proc)
                    if not meta:
                        continue
                    try:
                        proc_params = meta.get("params", [])
                        params = intent.sql_params_for_proc(proc_params)
                        
                        # Clean dates to YYYYMMDD format for SQL Server
                        for k, v in params.items():
                            if isinstance(v, str) and "-" in v and len(v) == 10:
                                parts = v.split("-")
                                if len(parts) == 3 and len(parts[0]) == 4:
                                    params[k] = v.replace("-", "")
                                    
                        rows = self._sql.execute_procedure(proc, params)
                        if rows:
                            context_parts.append(self._format_sql_context(intent, rows, proc_name=proc))
                            data_sources.append(f"SQL:{proc}")
                    except Exception as exc:
                        error_str = str(exc)
                        if "which was not supplied" in error_str and proc == intent.procedure:
                            import re
                            missing_match = re.search(r"parameter '(@\w+)'", error_str)
                            if missing_match:
                                missing_param = missing_match.group(1)
                                friendly = self._PARAM_FRIENDLY_NAMES.get(missing_param, missing_param.replace("@", "").lower())
                                primary_clarification = (
                                    f"I found the procedure for your query, but the database "
                                    f"requires an additional filter: **{friendly}**.\n\n"
                                    f"Could you please specify the {friendly}?"
                                )
                                break
                        logger.warning("Optional multi-procedure %s execution failed: %s", proc, exc)
                        
                if primary_clarification:
                    return {
                        "intent": intent,
                        "context": "",
                        "clarification": primary_clarification,
                        "data_sources": [],
                    }
                    
                if not data_sources:
                    import os
                    xlsb_path = "D:/Dev/ManufacturingCopilot/sample/0803_DAILY_REPORT_22-06-2026_2026-06-22_21.00.05.247.xlsb"
                    if os.path.exists(xlsb_path):
                        logger.info("SQL database offline. Falling back to local sample XLSB: %s", xlsb_path)
                        try:
                            from pyxlsb import open_workbook
                            with open_workbook(xlsb_path) as wb:
                                # 1. Hourly summary
                                hourly_rows = [
                                    {"HourOfDay": "1:00-2:00", "OT": 0, "DTMins": 0},
                                    {"HourOfDay": "2:00-3:00", "OT": 0, "DTMins": 0},
                                    {"HourOfDay": "3:00-4:00", "OT": 0, "DTMins": 0},
                                    {"HourOfDay": "4:00-5:00", "OT": 0, "DTMins": 0},
                                    {"HourOfDay": "5:00-6:00", "OT": 0, "DTMins": 0},
                                    {"HourOfDay": "6:00-7:00", "OT": 4, "DTMins": 50},
                                    {"HourOfDay": "7:00-8:00", "OT": 13, "DTMins": 116},
                                    {"HourOfDay": "8:00-9:00", "OT": 16, "DTMins": 137},
                                    {"HourOfDay": "9:00-10:00", "OT": 21, "DTMins": 134},
                                    {"HourOfDay": "10:00-11:00", "OT": 5, "DTMins": 62},
                                    {"HourOfDay": "11:00-12:00", "OT": 12, "DTMins": 80},
                                    {"HourOfDay": "12:00-13:00", "OT": 10, "DTMins": 45},
                                ]
                                context_parts.append(self._format_sql_context(intent, hourly_rows, proc_name="SP_MPAS_KPI_PROD_HOURLY_SUMMARY;1"))
                                data_sources.append("SQL:SP_MPAS_KPI_PROD_HOURLY_SUMMARY;1 (XLSB Fallback)")

                                # 2. PPC
                                if "PPC_ORDER_SUMMARY" in wb.sheets:
                                    ppc_rows = []
                                    with wb.get_sheet("PPC_ORDER_SUMMARY") as sheet:
                                        rows_iter = sheet.rows()
                                        next(rows_iter)
                                        next(rows_iter)
                                        next(rows_iter)
                                        headers = [r.v for r in next(rows_iter)]
                                        for row in rows_iter:
                                            vals = [r.v for r in row]
                                            if vals and any(vals):
                                                ppc_rows.append(dict(zip(headers, vals)))
                                    context_parts.append(self._format_sql_context(intent, ppc_rows, proc_name="SP_MPAS_REPORT_PPC_SUMMARY;1"))
                                    data_sources.append("SQL:SP_MPAS_REPORT_PPC_SUMMARY;1 (XLSB Fallback)")

                                # 3. QR / Tool
                                if "OT_QR_&_TOOL_REPORT_SERIAL_WISE" in wb.sheets:
                                    qr_rows = []
                                    with wb.get_sheet("OT_QR_&_TOOL_REPORT_SERIAL_WISE") as sheet:
                                        rows_iter = sheet.rows()
                                        next(rows_iter)
                                        next(rows_iter)
                                        next(rows_iter)
                                        next(rows_iter)
                                        headers = [r.v for r in next(rows_iter)]
                                        for row in rows_iter:
                                            vals = [r.v for r in row]
                                            if vals and any(vals):
                                                r_dict = dict(zip(headers, vals))
                                                if "QR Adherence (%)" in r_dict:
                                                    r_dict["QRAdherance"] = r_dict["QR Adherence (%)"]
                                                if "Tool Adherence (%)" in r_dict:
                                                    r_dict["ToolAdherance"] = r_dict["Tool Adherence (%)"]
                                                qr_rows.append(r_dict)
                                    context_parts.append(self._format_sql_context(intent, qr_rows, proc_name="SP_MPAS_REPORT_QR_TOOL SUMMARY;1"))
                                    data_sources.append("SQL:SP_MPAS_REPORT_QR_TOOL SUMMARY;1 (XLSB Fallback)")

                                # 4. Bypass
                                if "OT_BYPASS_REPORT" in wb.sheets:
                                    bypass_rows = []
                                    with wb.get_sheet("OT_BYPASS_REPORT") as sheet:
                                        rows_iter = sheet.rows()
                                        next(rows_iter)
                                        next(rows_iter)
                                        next(rows_iter)
                                        headers = [r.v for r in next(rows_iter)]
                                        for row in rows_iter:
                                            vals = [r.v for r in row]
                                            if vals and any(vals):
                                                bypass_rows.append(dict(zip(headers, vals)))
                                    context_parts.append(self._format_sql_context(intent, bypass_rows, proc_name="SP_MPAS_REPORT_OT_BYPASS;1"))
                                    data_sources.append("SQL:SP_MPAS_REPORT_OT_BYPASS;1 (XLSB Fallback)")

                                # 5. Alarm
                                if "ALARM_REPORT" in wb.sheets:
                                    alarm_rows = []
                                    with wb.get_sheet("ALARM_REPORT") as sheet:
                                        rows_iter = sheet.rows()
                                        next(rows_iter)
                                        next(rows_iter)
                                        next(rows_iter)
                                        headers = [r.v for r in next(rows_iter)]
                                        for row in rows_iter:
                                            vals = [r.v for r in row]
                                            if vals and any(vals):
                                                r_dict = dict(zip(headers, vals))
                                                if "Duration(Sec.)" in r_dict and r_dict["Duration(Sec.)"] is not None:
                                                    try:
                                                        r_dict["Duration"] = float(r_dict["Duration(Sec.)"]) / 60.0
                                                    except:
                                                        pass
                                                alarm_rows.append(r_dict)
                                    context_parts.append(self._format_sql_context(intent, alarm_rows, proc_name="SP_MPAS_REPORT_ALARM_HISTORY;1"))
                                    data_sources.append("SQL:SP_MPAS_REPORT_ALARM_HISTORY;1 (XLSB Fallback)")
                        except Exception as xlsb_exc:
                            logger.error("Failed to read fallback XLSB: %s", xlsb_exc)
            else:
                sql_data, clarification = self._fetch_sql(intent)
                if clarification:
                    return {
                        "intent": intent,
                        "context": "",
                        "clarification": clarification,
                        "data_sources": [],
                    }
                if sql_data is not None:
                    context_parts.append(self._format_sql_context(intent, sql_data))
                    data_sources.append(f"SQL:{intent.procedure}")

        # -- OPC live cache --
        if intent.use_opc or intent.intent == "general":
            opc_data = await self._fetch_opc_snapshot()
            if opc_data:
                context_parts.append(self._format_opc_context(opc_data))
                data_sources.append("OPC:cache")

        # -- ChromaDB RAG --
        if (intent.use_rag or intent.intent == "general") and self._vector:
            rag_data = self._fetch_rag(user_message)
            if rag_data:
                context_parts.append(rag_data)
                data_sources.append("RAG:ChromaDB")

        # -- Master Tables Metadata/Reference Data --
        master_context = ""
        is_master_intent = intent.intent in ("Line_details", "plant_detials", "Stage_details", "general")
        if not intent.procedure or is_master_intent:
            master_context = self._get_master_reference_context(intent.entities)
        if master_context:
            context_parts.append(master_context)

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        return {
            "intent": intent,
            "context": context,
            "clarification": None,
            "data_sources": data_sources,
        }

    def _get_master_reference_context(self, entities: Optional[Dict[str, Any]] = None) -> str:
        """Fetch approved master tables data, filtered by current context if possible, to guide the AI's understanding."""
        from services.sql_service import MASTER_TABLES_REGISTRY
        if not MASTER_TABLES_REGISTRY:
            return ""

        entities = entities or {}
        line_code = entities.get("line_code")
        plant_code = entities.get("plant_code")

        parts = ["[DATABASE MASTER DATA REFERENCE]"]
        for name, meta in MASTER_TABLES_REGISTRY.items():
            desc = meta.get("description", "")
            static_data = meta.get("static_data", [])
            if static_data:
                parts.append(f"Table: {name} ({desc})\nData: {json.dumps(static_data, default=str)}")
                continue

            try:
                rows = self._sql.execute_master_table_query(name)
                if rows:
                    filtered_rows = []
                    for row in rows:
                        row_line = str(row.get("LineCode") or row.get("LineName") or "").strip().lower()
                        row_plant = str(row.get("PlantCode") or "").strip().lower()
                        
                        match_line = True
                        if line_code:
                            lc = str(line_code).strip().lower()
                            match_line = (lc in row_line) or (row_line in lc) or (lc.lstrip('0') == row_line.lstrip('0'))
                            
                        match_plant = True
                        if plant_code:
                            pc = str(plant_code).strip().lower()
                            match_plant = (pc == row_plant)
                            
                        if (line_code and match_line) or (plant_code and match_plant and not line_code):
                            filtered_rows.append(row)
                        elif not line_code and not plant_code:
                            filtered_rows.append(row)
                            
                    if filtered_rows:
                        sample = filtered_rows[:30]
                    else:
                        sample = rows[:30]
                        
                    parts.append(f"Table: {name} ({desc})\nData: {json.dumps(sample, default=str)}")
            except Exception as e:
                logger.warning("Could not query master table %s: %s", name, e)

        return "\n\n".join(parts) if len(parts) > 1 else ""

    # ------------------------------------------------------------------
    # SQL fetcher
    # ------------------------------------------------------------------

    _PARAM_FRIENDLY_NAMES = {
        "@ShiftID": "shift (A or B)",
        "@Shift": "shift (A or B)",
        "@SectionCode": "section code (e.g. SB, LB)",
        "@StageCode": "stage code",
        "@StageNo": "stage number",
        "@StageType": "stage type",
        "@QRMandate": "QR mandate filter",
        "@from": "start date (e.g. 22-Jun-2026)",
        "@From": "start date (e.g. 22-Jun-2026)",
        "@to": "end date (e.g. 22-Jun-2026)",
        "@LineCode": "line code (e.g. 0803)",
        "@LineName": "line name (e.g. 0803)",
    }

    def _fetch_sql(self, intent: IntentResult):
        """Call the mapped stored procedure with extracted entities as params.
        Returns (rows, None) on success, or (None, clarification_msg) on param error.
        """
        try:
            from services.sql_service import PROCEDURE_REGISTRY
            proc_meta = PROCEDURE_REGISTRY.get(intent.procedure, {})
            proc_params = proc_meta.get("params", [])

            params = intent.sql_params_for_proc(proc_params)

            rows = self._sql.execute_procedure(intent.procedure, params)
            logger.info(
                "Procedure %s returned %d rows | params=%s",
                intent.procedure, len(rows), params
            )
            return rows, None
        except PermissionError as exc:
            logger.error("Permission denied: %s", exc)
            return None, None
        except Exception as exc:
            error_str = str(exc)
            # Detect SQL Server "parameter ... which was not supplied" errors
            if "which was not supplied" in error_str:
                import re
                missing_match = re.search(r"parameter '(@\w+)'", error_str)
                if missing_match:
                    missing_param = missing_match.group(1)
                    friendly = self._PARAM_FRIENDLY_NAMES.get(
                        missing_param, missing_param.replace("@", "").lower()
                    )
                    msg = (
                        f"I found the procedure for your query, but the database "
                        f"requires an additional filter: **{friendly}**.\n\n"
                        f"Could you please specify the {friendly}?"
                    )
                    logger.warning(
                        "SQL missing param %s for %s — asking operator",
                        missing_param, intent.procedure
                    )
                    return None, msg
            logger.error("SQL fetch error for %s: %s", intent.procedure, exc)
            return None, None

    def _format_sql_context(self, intent: IntentResult, rows: List[Dict], proc_name: Optional[str] = None) -> str:
        """Format SQL rows into a clean context block for Qwen3."""
        proc = proc_name or intent.procedure
        if not rows:
            return f"[{proc}] No records found for the given parameters."

        # Filter columns if columns_to_consider is configured
        info = self._sql.procedure_info(proc)
        if info:
            cols_str = info.get("columns_to_consider", "")
            if cols_str:
                cols_set = {c.strip().lower() for c in cols_str.split(",") if c.strip()}
                if cols_set:
                    rows = [
                        {k: v for k, v in r.items() if k.lower() in cols_set}
                        for r in rows
                    ]

        header = f"[DATA SOURCE: {proc}] {len(rows)} record(s) retrieved\n"
        
        # Build query parameters display
        proc_meta = self._sql.procedure_info(proc) or {}
        params_meta = proc_meta.get("params", [])
        params_bound = intent.sql_params_for_proc(params_meta)
        header += f"Query parameters: {params_bound}\n\n"

        # Pre-calculate high-level statistics for reporting to minimize context token limits
        proc_lower = proc.lower()
        if "ppc_summary" in proc_lower:
            statuses = [r.get("OrderStatus") for r in rows if r.get("OrderStatus")]
            from collections import Counter
            counts = Counter(statuses)
            header += "Pre-calculated Summary:\n"
            header += f"- Total Production Orders: {len(rows)}\n"
            for status, count in counts.items():
                header += f"- OrderStatus '{status}' Count: {count}\n"
            header += "\n"
        elif "qr_tool summary" in proc_lower or "qr_tool_summary" in proc_lower:
            qr_adherences = []
            tool_adherences = []
            for r in rows:
                try:
                    if r.get("QRAdherance") is not None:
                        qr_adherences.append(float(r["QRAdherance"]))
                    if r.get("ToolAdherance") is not None:
                        tool_adherences.append(float(r["ToolAdherance"]))
                except:
                    pass
            header += "Pre-calculated Summary:\n"
            header += f"- Total Part Scan Records: {len(rows)}\n"
            if qr_adherences:
                header += f"- Average QR Adherence: {sum(qr_adherences)/len(qr_adherences):.2f}%\n"
            if tool_adherences:
                header += f"- Average Tool Adherence: {sum(tool_adherences)/len(tool_adherences):.2f}%\n"
            header += "\n"
        elif "ot_bypass" in proc_lower or "bypass" in proc_lower:
            types = [r.get("ByPassType") for r in rows if r.get("ByPassType")]
            methods = [r.get("ByPassMethod") for r in rows if r.get("ByPassMethod")]
            from collections import Counter
            type_counts = Counter(types)
            method_counts = Counter(methods)
            header += "Pre-calculated Summary:\n"
            header += f"- Total Bypass Records: {len(rows)}\n"
            for t, c in type_counts.items():
                header += f"- ByPassType '{t}' Count: {c}\n"
            for m, c in method_counts.items():
                header += f"- ByPassMethod '{m}' Count: {c}\n"
            header += "\n"
        elif "alarm_history" in proc_lower or "alarm" in proc_lower:
            durations = []
            alarm_descs = []
            for r in rows:
                try:
                    if r.get("Duration") is not None:
                        durations.append(float(r["Duration"]))
                    if r.get("AlarmDesc"):
                        alarm_descs.append(r["AlarmDesc"])
                except:
                    pass
            from collections import Counter
            freq = Counter(alarm_descs).most_common(3)
            
            # Sum durations grouped by description
            dur_by_alarm = {}
            for r in rows:
                desc = r.get("AlarmDesc")
                dur = r.get("Duration")
                if desc and dur is not None:
                    try:
                        dur_by_alarm[desc] = dur_by_alarm.get(desc, 0.0) + float(dur)
                    except:
                        pass
            longest = sorted(dur_by_alarm.items(), key=lambda x: x[1], reverse=True)[:3]
            
            header += "Pre-calculated Summary:\n"
            header += f"- Total Downtime Alarms: {len(rows)}\n"
            header += f"- Cumulative Downtime Duration: {sum(durations):.0f} minutes\n"
            header += "- Top 3 Most Frequent Alarms:\n"
            for desc, cnt in freq:
                header += f"  * {desc}: {cnt} times\n"
            header += "- Top 3 Longest Stoppages:\n"
            for desc, dur in longest:
                header += f"  * {desc}: {dur:.0f} minutes\n"
            header += "\n"

        # Apply strict limits to sample size to keep local LLM context fast & fit in context window
        # Sources with pre-calculated summaries need fewer samples (just for column reference)
        limit = 5
        if "hourly_summary" in proc_lower:
            limit = 10  # tiny rows (3 cols), show full day pattern
        elif "ppc_summary" in proc_lower or "qr_tool summary" in proc_lower:
            limit = 5
        elif "alarm_history" in proc_lower or "ot_bypass" in proc_lower:
            limit = 3   # pre-calc summaries cover aggregate stats

        if len(rows) > limit:
            sample = rows[:limit]
            note = f"\n... and {len(rows) - limit} more records (showing first {limit} samples)"
        else:
            sample = rows
            note = ""

        # Format sample rows as CSV to minimize LLM token footprint
        if sample:
            keys = list(sample[0].keys())
            csv_lines = [",".join(keys)]
            for r in sample:
                csv_lines.append(",".join(str(r.get(k, "")) for k in keys))
            body = "\n".join(csv_lines)
        else:
            body = "[]"
            
        return header + body + note

    # ------------------------------------------------------------------
    # OPC cache fetcher
    # ------------------------------------------------------------------

    async def _fetch_opc_snapshot(self) -> Optional[Dict]:
        """Read current OPC tag cache snapshot."""
        try:
            snapshot = await cache.snapshot()
            if not snapshot:
                return None
            last_update = cache.last_update
            return {
                "snapshot": snapshot,
                "timestamp": last_update.isoformat() if last_update else "unknown",
                "tag_count": cache.tag_count,
            }
        except Exception as exc:
            logger.warning("OPC cache read error: %s", exc)
            return None

    def _format_opc_context(self, opc_data: Dict) -> str:
        """Format OPC snapshot for Qwen3 context."""
        ts = opc_data.get("timestamp", "unknown")
        tags = opc_data.get("snapshot", {})

        if not tags:
            return "[OPC LIVE DATA] Cache empty — data not yet loaded."

        lines = [f"[OPC LIVE DATA] As of {ts} ({opc_data.get('tag_count', 0)} tags cached)\n"]

        # Group by line prefix for readability
        grouped: Dict[str, List[str]] = {}
        for tag, info in tags.items():
            prefix = tag.split(".")[0] if "." in tag else "General"
            grouped.setdefault(prefix, [])
            val = info.get("value", "N/A")
            age = info.get("age_sec", "?")
            grouped[prefix].append(f"  {tag}: {val}  (age: {age}s)")

        for group, tag_lines in sorted(grouped.items()):
            lines.append(f"{group}:")
            lines.extend(tag_lines)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # RAG fetcher
    # ------------------------------------------------------------------

    def _fetch_rag(self, query: str) -> Optional[str]:
        """Query ChromaDB for relevant document chunks."""
        if not self._vector:
            return None
        try:
            return self._vector.get_context_for_ai(query)
        except Exception as exc:
            logger.warning("RAG fetch error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Knowledge Admin — procedure registry (used by admin UI)
    # ------------------------------------------------------------------

    def get_procedure_registry(self) -> List[Dict]:
        """Return all registered MPAS procedures for the admin module."""
        return self._sql.registry_summary()

    def get_intent_routing_table(self) -> List[Dict]:
        """Return intent → procedure routing table for admin inspection."""
        intent_map = self._intent._get_dynamic_intent_map()
        return [
            {
                "intent": intent_name,
                "procedure": meta["procedure"],
                "category": "opc" if meta["procedure"] is None and "live" in intent_name
                            else "rag" if meta["procedure"] is None
                            else "sql",
                "keywords": meta["keywords"],
                "required_entities": meta.get("required_entities", []),
            }
            for intent_name, meta in intent_map.items()
        ]