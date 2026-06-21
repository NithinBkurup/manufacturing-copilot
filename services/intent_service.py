"""
Intent Service – Manufacturing Copilot
Classifies user queries and extracts entities (OrderNo, SerialNo, LineCode, etc.)
so manufacturing_service.py knows exactly which stored procedure to call
and what parameters to pass.

Rule: The LLM never touches the database.
      User → Intent Engine → Stored Procedure → Result → Qwen3 → Response
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("copilot.intent")


# ---------------------------------------------------------------------------
# Entity patterns
# ---------------------------------------------------------------------------

# OrderNo — numeric, typically 9 digits
ORDER_NO_PATTERN = re.compile(r"\b(1[0-9]{8})\b")

# SerialNo — multiple formats:
#   RHPW133610  (axles)     — alpha-prefix + digits
#   TRPS706859  (trucks)    — alpha-prefix + digits
#   MB1G9VHD1TPRK3407       — VIN format (17 chars alphanumeric)
SERIAL_AXLE_PATTERN  = re.compile(r"\b(RHPW[0-9]{6,})\b", re.IGNORECASE)
SERIAL_TRUCK_PATTERN = re.compile(r"\b(TRPS[0-9]{6,})\b", re.IGNORECASE)
SERIAL_VIN_PATTERN   = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")  # VIN — no I/O/Q

# LineCode — "line 1", "L3", "line3", "assembly line 2"
LINE_PATTERN = re.compile(r"\b(?:line\s*|L)([1-9][0-9]?)\b", re.IGNORECASE)

# Shift
SHIFT_PATTERN = re.compile(r"\bshift\s*([AB])\b", re.IGNORECASE)

# Date — YYYY-MM-DD or DD/MM/YYYY
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b")

# Bolt number — "bolt 5", "torque bolt 12"
BOLT_PATTERN = re.compile(r"\bbolt\s*([1-9][0-9]?)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Intent definitions — map to MPAS stored procedures
# ---------------------------------------------------------------------------

INTENT_MAP: Dict[str, Dict] = {
    "order_status": {
        "procedure": "SP_MPAS_CREATED_ORDERS",
        "required_entities": [],          # Can run with just OrderNo OR LineCode
        "keywords": [
            "status", "order", "where is", "progress", "stage",
            "complete", "running", "active", "created", "details"
        ],
        "priority": 2,
    },
    "process_results": {
        "procedure": "PNR_Get_updateProcessData_API_Raw_Data",
        "required_entities": ["order_no"],
        "keywords": [
            "process", "result", "pass", "fail", "operation", "stage result",
            "tool", "ok", "ng", "tightening", "station"
        ],
        "priority": 3,
    },
    "qr_scan": {
        "procedure": "PNR_Get_updateQRV2_API_Raw_Data",
        "required_entities": ["order_no"],
        "keywords": [
            "scan", "scanned", "qr", "part scan", "barcode", "missing scan",
            "part traced", "traceability scan"
        ],
        "priority": 3,
    },
    "production_metrics": {
        "procedure": "SP_MPAS_HOURLY_PRODUCTION_METRICS",
        "required_entities": [],
        "keywords": [
            "how many", "output", "count", "units", "produced", "efficiency",
            "target", "actual", "throughput", "hourly", "shift output",
            "production count", "takt"
        ],
        "priority": 2,
    },
    "bom_material": {
        "procedure": "SP_MPAS_SAP_ORDER_BOM",
        "required_entities": ["order_no"],
        "keywords": [
            "bom", "bill of material", "parts", "material", "planned",
            "sap", "missing part", "component", "part list"
        ],
        "priority": 3,
    },
    "bypass": {
        "procedure": "SP_MPAS_BYPASS_AUDIT",
        "required_entities": ["order_no"],
        "keywords": [
            "bypass", "bypassed", "approved", "bypass request",
            "who approved", "bypass history", "override"
        ],
        "priority": 4,
    },
    "torque_quality": {
        "procedure": "SP_MPAS_PROCESS_STATUS_TORQUE",
        "required_entities": [],
        "keywords": [
            "torque", "bolt", "tightening torque", "nm", "newton",
            "quality trace", "torque result", "bolt torque"
        ],
        "priority": 4,
    },
    "deviation": {
        "procedure": "SP_MPAS_ORDER_DEVIATIONS",
        "required_entities": ["order_no"],
        "keywords": [
            "deviation", "deviations", "deviated", "who raised",
            "deviation status", "deviation comment", "ncr"
        ],
        "priority": 4,
    },
    "live_opc": {
        "procedure": None,               # Served from cache — no SQL call
        "required_entities": [],
        "keywords": [
            "live", "right now", "current status", "machine status",
            "is running", "opc", "plc", "recipe running", "current recipe",
            "what is on line", "line status"
        ],
        "priority": 1,
    },
    "document_search": {
        "procedure": None,               # Served from ChromaDB RAG
        "required_entities": [],
        "keywords": [
            "sop", "procedure", "instruction", "manual", "how to", "work instruction",
            "troubleshoot", "what is the process for", "guide", "document"
        ],
        "priority": 1,
    },
}


# ---------------------------------------------------------------------------
# Dataclass — structured intent result
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent: str                          # e.g. "order_status"
    procedure: Optional[str]             # e.g. "SP_MPAS_CREATED_ORDERS"
    confidence: float = 0.0              # 0.0 – 1.0
    entities: Dict[str, Optional[str]] = field(default_factory=dict)
    use_opc: bool = False
    use_rag: bool = False
    missing_required: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    clarification_prompt: Optional[str] = None

    def sql_params(self) -> Dict[str, str]:
        """Build @Param dict for sql_service.execute_procedure()."""
        mapping = {
            "order_no":   "@OrderNo",
            "serial_no":  "@SerialNo",
            "line_code":  "@LineCode",
            "shift_id":   "@ShiftID",
            "shift_date": "@ShiftDate",
        }
        result = {}
        for entity_key, param_name in mapping.items():
            val = self.entities.get(entity_key)
            if val is not None:
                result[param_name] = val
        return result


# ---------------------------------------------------------------------------
# Intent Engine
# ---------------------------------------------------------------------------

class IntentService:
    """
    Classifies the user's message and extracts manufacturing entities.
    Routes to the correct stored procedure or data source.
    """

    def classify(self, message: str) -> IntentResult:
        """Main entry point — returns a fully populated IntentResult."""
        msg_lower = message.lower()

        # 1. Extract all entities
        entities = self._extract_entities(message)
        logger.debug("Extracted entities: %s", entities)

        # 2. Score all intents
        scores = self._score_intents(msg_lower)
        if not scores:
            return self._general_fallback(entities)

        # 3. Pick highest scoring intent
        intent_name, score = scores[0]
        intent_meta = INTENT_MAP[intent_name]

        # 4. Check required entities
        missing = self._check_required(intent_meta, entities)

        # 5. Determine data sources
        use_opc = intent_name == "live_opc"
        use_rag = intent_name == "document_search"

        result = IntentResult(
            intent=intent_name,
            procedure=intent_meta["procedure"],
            confidence=score,
            entities=entities,
            use_opc=use_opc,
            use_rag=use_rag,
            missing_required=missing,
            clarification_needed=bool(missing),
        )

        if missing:
            result.clarification_prompt = self._build_clarification(intent_name, missing)

        logger.info(
            "Intent: %s | confidence=%.2f | proc=%s | entities=%s | missing=%s",
            intent_name, score, intent_meta["procedure"], entities, missing
        )
        return result

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, message: str) -> Dict[str, Optional[str]]:
        entities: Dict[str, Optional[str]] = {
            "order_no":   None,
            "serial_no":  None,
            "line_code":  None,
            "shift_id":   None,
            "shift_date": None,
            "bolt_no":    None,
        }

        # OrderNo
        m = ORDER_NO_PATTERN.search(message)
        if m:
            entities["order_no"] = m.group(1)

        # SerialNo — try all formats, use the first match
        for pattern in [SERIAL_AXLE_PATTERN, SERIAL_TRUCK_PATTERN, SERIAL_VIN_PATTERN]:
            m = pattern.search(message)
            if m:
                entities["serial_no"] = m.group(1).upper()
                break

        # If no OrderNo found but SerialNo found, note it — manufacturing_service
        # may need to resolve the OrderNo from the SerialNo first

        # LineCode
        m = LINE_PATTERN.search(message)
        if m:
            entities["line_code"] = m.group(1)

        # ShiftID
        m = SHIFT_PATTERN.search(message)
        if m:
            entities["shift_id"] = m.group(1).upper()

        # Date
        m = DATE_PATTERN.search(message)
        if m:
            entities["shift_date"] = m.group(1)

        # Bolt number
        m = BOLT_PATTERN.search(message)
        if m:
            entities["bolt_no"] = m.group(1)

        return entities

    # ------------------------------------------------------------------
    # Intent scoring
    # ------------------------------------------------------------------

    def _score_intents(self, msg_lower: str) -> List[Tuple[str, float]]:
        scores = []
        words = set(msg_lower.split())

        for intent_name, meta in INTENT_MAP.items():
            keyword_hits = sum(1 for kw in meta["keywords"] if kw in msg_lower)
            if keyword_hits == 0:
                continue

            # Normalise: hits / total keywords, weighted by priority
            raw = keyword_hits / len(meta["keywords"])
            priority_boost = meta["priority"] * 0.05
            score = min(raw + priority_boost, 1.0)
            scores.append((intent_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Required entity check
    # ------------------------------------------------------------------

    def _check_required(self, meta: Dict, entities: Dict) -> List[str]:
        missing = []
        for req in meta.get("required_entities", []):
            if entities.get(req) is None:
                missing.append(req)
        return missing

    # ------------------------------------------------------------------
    # Clarification prompt builder
    # ------------------------------------------------------------------

    def _build_clarification(self, intent: str, missing: List[str]) -> str:
        prompts = {
            "order_no":  "Could you provide the order number (e.g. 147190737)?",
            "serial_no": "Could you provide the serial number (e.g. RHPW133610)?",
            "line_code": "Which production line are you asking about (e.g. Line 1)?",
            "shift_id":  "Which shift — A or B?",
        }
        parts = [prompts.get(m, f"Please provide the {m.replace('_', ' ')}.") for m in missing]
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _general_fallback(self, entities: Dict) -> IntentResult:
        """No intent matched — use RAG + OPC as best effort."""
        return IntentResult(
            intent="general",
            procedure=None,
            confidence=0.0,
            entities=entities,
            use_opc=True,
            use_rag=True,
        )