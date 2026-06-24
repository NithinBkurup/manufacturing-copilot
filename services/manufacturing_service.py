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
            sql_data = self._fetch_sql(intent)
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

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        return {
            "intent": intent,
            "context": context,
            "clarification": None,
            "data_sources": data_sources,
        }

    # ------------------------------------------------------------------
    # SQL fetcher
    # ------------------------------------------------------------------

    def _fetch_sql(self, intent: IntentResult) -> Optional[List[Dict]]:
        """Call the mapped stored procedure with extracted entities as params."""
        try:
            params = intent.sql_params()

            # Always inject PlantCode where needed (never from user input)
            # Some MPAS procedures filter by PlantCode internally — pass if needed
            # params["@PlantCode"] = PLANT_CODE   # Uncomment if required by proc

            rows = self._sql.execute_procedure(intent.procedure, params)
            logger.info(
                "Procedure %s returned %d rows | params=%s",
                intent.procedure, len(rows), params
            )
            return rows
        except PermissionError as exc:
            logger.error("Permission denied: %s", exc)
            return None
        except Exception as exc:
            logger.error("SQL fetch error for %s: %s", intent.procedure, exc)
            return None

    def _format_sql_context(self, intent: IntentResult, rows: List[Dict]) -> str:
        """Format SQL rows into a clean context block for Qwen3."""
        if not rows:
            return f"[{intent.procedure}] No records found for the given parameters."

        # Filter columns if columns_to_consider is configured
        info = self._sql.procedure_info(intent.procedure)
        if info:
            cols_str = info.get("columns_to_consider", "")
            if cols_str:
                cols_set = {c.strip().lower() for c in cols_str.split(",") if c.strip()}
                if cols_set:
                    rows = [
                        {k: v for k, v in r.items() if k.lower() in cols_set}
                        for r in rows
                    ]

        # Highlight key fields based on intent category
        header = f"[DATA SOURCE: {intent.procedure}] {len(rows)} record(s) retrieved\n"
        header += f"Query parameters: {intent.sql_params()}\n\n"

        # For large result sets, summarise rather than dump everything
        if len(rows) > 20:
            sample = rows[:20]
            note = f"\n... and {len(rows) - 20} more records (showing first 20)"
        else:
            sample = rows
            note = ""

        body = json.dumps(sample, default=str, indent=2)
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