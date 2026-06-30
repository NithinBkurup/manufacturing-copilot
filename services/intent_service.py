"""
Intent Service – Manufacturing Copilot
Classifies user queries and extracts entities (OrderNo, SerialNo, LineCode, etc.)
Dynamic routing based on trigger keywords from SQL, OPC, and KB.
"""

import logging
import re
from datetime import datetime
import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from config.settings import settings

logger = logging.getLogger("copilot.intent")


# --- Cosine Similarity Helpers ---
def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1, v2):
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if not mag1 or not mag2:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)


def load_kb_intent_keywords() -> list:
    config_path = "config/kb_intent.json"
    default_keywords = ["sop", "procedure", "instruction", "manual", "how to", "work instruction", "troubleshoot", "guide", "document"]
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return default_keywords


def load_opc_tag_keywords() -> list:
    config_path = "config/opc_tags.json"
    keywords = []
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for tag_info in data.values():
                        kws = tag_info.get("keywords") or ""
                        if kws:
                            keywords.extend([kw.strip().lower() for kw in kws.split(",") if kw.strip()])
        except Exception:
            pass
    return keywords


# Entity patterns
ORDER_NO_PATTERN = re.compile(r"\b([01][0-9]{2}\s*[-_]?\s*[0-9]{3}\s*[-_]?\s*[0-9]{3,4})\b")
SERIAL_AXLE_PATTERN  = re.compile(r"\b(RHPW\s*[-_]?\s*[0-9]{6,})\b", re.IGNORECASE)
SERIAL_TRUCK_PATTERN = re.compile(r"\b(TRPS\s*[-_]?\s*[0-9]{6,})\b", re.IGNORECASE)
SERIAL_TLP_PATTERN   = re.compile(r"\b(TLP[A-Z]\s*[-_]?\s*[0-9]{6})\b", re.IGNORECASE)
SERIAL_VIN_PATTERN   = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
LINE_PATTERN = re.compile(r"(?:line\s*(?:code)?\s*[-=:]?\s*|\bL)([0-9]{1,4})\b", re.IGNORECASE)
SHIFT_PATTERN = re.compile(r"\bshift\s*[-=:]?\s*([AB]|ALL)\b", re.IGNORECASE)
DATE_PATTERN = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{2}/\d{2}/\d{4}"
    r"|\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}"
    r")\b", re.IGNORECASE
)
BOLT_PATTERN = re.compile(r"\bbolt\s*([1-9][0-9]?)\b", re.IGNORECASE)
PLANT_PATTERN = re.compile(r"\b(?:plant\s*|P)([0-9]{4})\b", re.IGNORECASE)


@dataclass
class IntentResult:
    intent: str
    procedure: Optional[str]
    confidence: float = 0.0
    entities: Dict[str, Optional[str]] = field(default_factory=dict)
    use_opc: bool = False
    use_rag: bool = False
    missing_required: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    clarification_prompt: Optional[str] = None

    def sql_params(self) -> Dict[str, str]:
        """Map extracted entities to SQL parameter names.
        Handles both old-style (@OrderNo, @SerialNo) and new-style (@PlantCode, @from, @to) params.
        """
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

    def sql_params_for_proc(self, proc_params: List[str]) -> Dict[str, str]:
        """Build SQL params dict matching the specific procedure's parameter list.
        Maps extracted entities to whichever param names the SP actually uses.
        Guarantees that all expected parameters are bound (at least as None/NULL) to avoid SQL errors.
        """
        from config.settings import settings
        PLANT_CODE = settings.PLANT_CODE

        result = {}
        for p in proc_params:
            p_lower = p.lower().replace("@", "")
            if p_lower == "plantcode":
                val = self.entities.get("plant_code")
                result[p] = val if val is not None else PLANT_CODE
            elif p_lower == "linecode" or p_lower == "linename":
                val = self.entities.get("line_code")
                result[p] = val if val is not None else None
            elif p_lower == "from":
                val = self.entities.get("from_date")
                if val and "-" in val:
                    val = val.replace("-", "")
                result[p] = val if val is not None else None
            elif p_lower == "to":
                val = self.entities.get("to_date")
                if val and "-" in val:
                    val = val.replace("-", "")
                result[p] = val if val is not None else None
            elif p_lower in ("shiftid", "shift"):
                val = self.entities.get("shift_id")
                # Default to 'ALL' to match procedure query expectations
                result[p] = val if val is not None else "ALL"
            elif p_lower == "orderno":
                val = self.entities.get("order_no")
                result[p] = val if val is not None else None
            elif p_lower == "serialno":
                val = self.entities.get("serial_no")
                result[p] = val if val is not None else None
            elif p_lower == "stagecode" or p_lower == "stageno":
                result[p] = "ALL"
            elif p_lower == "sectioncode":
                result[p] = "ALL"
            elif p_lower == "qrmandate":
                result[p] = "ALL"
            elif p_lower == "stagetype":
                result[p] = "ALL"
            else:
                result[p] = "ALL"
        return result


class IntentService:

    def _get_dynamic_intent_map(self) -> Dict[str, Dict]:
        # Gather KB document-specific keywords
        kb_keywords = load_kb_intent_keywords()
        try:
            from app import app
            if hasattr(app, "state") and hasattr(app.state, "vector_service") and app.state.vector_service:
                for file_info in app.state.vector_service.indexed_files:
                    kws = file_info.get("keywords") or ""
                    if kws:
                        kb_keywords.extend([kw.strip().lower() for kw in kws.split(",") if kw.strip()])
        except Exception:
            pass

        intent_map = {
            "live_opc": {
                "procedure": None,
                "required_entities": [],
                "keywords": ["live", "right now", "current status", "machine status", "is running", "opc", "plc", "recipe", "line status"] + load_opc_tag_keywords(),
                "priority": 4
            },
            "document_search": {
                "procedure": None,
                "required_entities": [],
                "keywords": list(set(kb_keywords)),
                "priority": 4
            }
        }
        
        # Dynamically add procedures from PROCEDURE_REGISTRY
        from services.sql_service import PROCEDURE_REGISTRY
        for proc_name, meta in PROCEDURE_REGISTRY.items():
            keywords = []
            
            # User defined intent keywords
            keywords_val = meta.get("keywords") or meta.get("intent") or ""
            if keywords_val:
                keywords.extend([k.strip().lower() for k in keywords_val.split(",") if k.strip()])
            
            # Category and description keywords
            desc_val = meta.get("description", "")
            if desc_val:
                keywords.extend([k.strip().lower() for k in desc_val.split() if len(k) > 3])
                
            cat_val = meta.get("category", "")
            if cat_val:
                keywords.append(cat_val.lower())
            
            required = []
            params = meta.get("params", [])
            for p in params:
                p_lower = p.lower().replace("@", "")
                # Only line_code is truly required for production queries
                if p_lower in ("linecode", "linename") and "line_code" not in required:
                    required.append("line_code")
                # All other params (@PlantCode, @from, @to, @ShiftID,
                # @SectionCode, @StageCode, @QRMandate, @StageType, @StageNo)
                # are optional — auto-injected or passed if available
                    
            intent_key = meta.get("intent", "").split(",")[0].strip() if meta.get("intent") else proc_name
            if not intent_key:
                intent_key = proc_name

            intent_map[intent_key] = {
                "procedure": proc_name,
                "required_entities": required,
                "keywords": list(set(keywords)),
                "priority": 3
            }
        return intent_map

    def classify(self, message: str) -> IntentResult:
        msg_lower = message.lower()
        entities = self._extract_entities(message)
        intent_map = self._get_dynamic_intent_map()

        scores = self._score_intents(msg_lower, intent_map)
        
        # Semantic fallback for low confidence or no matches
        if not scores or scores[0][1] < 0.35:
            sem_scores = self._semantic_fallback(message, intent_map)
            if sem_scores:
                if not scores or sem_scores[0][1] > scores[0][1]:
                    scores = sem_scores

        if not scores:
            return self._general_fallback(entities)

        intent_name, score = scores[0]
        intent_meta = intent_map[intent_name]
        missing = self._check_required(intent_meta, entities)

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

    def _extract_entities(self, message: str) -> Dict[str, Optional[str]]:
        entities: Dict[str, Optional[str]] = {
            "order_no":   None,
            "serial_no":  None,
            "line_code":  None,
            "shift_id":   None,
            "shift_date": None,
            "bolt_no":    None,
            "plant_code": None,
            "from_date":  None,
            "to_date":    None,
        }

        m = ORDER_NO_PATTERN.search(message)
        if m:
            entities["order_no"] = re.sub(r"\D", "", m.group(1))

        for pattern in [SERIAL_AXLE_PATTERN, SERIAL_TRUCK_PATTERN, SERIAL_TLP_PATTERN, SERIAL_VIN_PATTERN]:
            m = pattern.search(message)
            if m:
                entities["serial_no"] = re.sub(r"[\s\-_]", "", m.group(1)).upper()
                break

        m = LINE_PATTERN.search(message)
        if m:
            raw_line = m.group(1).lstrip("0") or "0"
            # Preserve 4-digit codes like 0803 as-is
            entities["line_code"] = m.group(1) if len(m.group(1)) == 4 else raw_line

        m = SHIFT_PATTERN.search(message)
        if m:
            val = m.group(1).upper()
            entities["shift_id"] = None if val == "ALL" else val

        dates = DATE_PATTERN.findall(message)
        if len(dates) >= 2:
            entities["from_date"] = self._normalize_date(dates[0])
            entities["to_date"] = self._normalize_date(dates[1])
            entities["shift_date"] = entities["from_date"]
        elif len(dates) == 1:
            entities["from_date"] = self._normalize_date(dates[0])
            entities["to_date"] = self._normalize_date(dates[0])
            entities["shift_date"] = entities["from_date"]

        m = BOLT_PATTERN.search(message)
        if m:
            entities["bolt_no"] = m.group(1)

        m = PLANT_PATTERN.search(message)
        if m:
            entities["plant_code"] = m.group(1)

        return entities

    def _score_intents(self, msg_lower: str, intent_map: Dict[str, Dict]) -> List[Tuple[str, float]]:
        scores = []
        for intent_name, meta in intent_map.items():
            keyword_hits = sum(1 for kw in meta["keywords"] if kw in msg_lower)
            if keyword_hits == 0:
                continue

            raw = keyword_hits / len(meta["keywords"]) if meta["keywords"] else 0
            priority_boost = meta.get("priority", 2) * 0.05
            score = min(raw + priority_boost, 1.0)
            scores.append((intent_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _semantic_fallback(self, msg: str, intent_map: Dict[str, Dict]) -> List[Tuple[str, float]]:
        try:
            from app import app
            if not (hasattr(app, "state") and hasattr(app.state, "vector_service") and app.state.vector_service):
                return []
            vector_service = app.state.vector_service
            if not vector_service._embed_fn:
                return []
            
            candidates = []
            candidates.append(("live_opc", "check live machine status and PLC tags current values recipe program"))
            candidates.append(("document_search", "standard operating procedure work instructions manuals troubleshoot guide how to"))
            
            from services.sql_service import PROCEDURE_REGISTRY
            for proc_name, meta in PROCEDURE_REGISTRY.items():
                desc = meta.get("description", "")
                examples = meta.get("example_questions", [])
                text_to_compare = desc + " " + " ".join(examples)
                intent_key = meta.get("intent", "").split(",")[0].strip() if meta.get("intent") else proc_name
                if not intent_key:
                    intent_key = proc_name
                candidates.append((intent_key, text_to_compare))
                
            texts_to_embed = [msg] + [c[1] for c in candidates]
            embeddings = vector_service._embed_fn(texts_to_embed)
            
            query_emb = embeddings[0]
            candidate_embs = embeddings[1:]
            
            scores = []
            for (intent_name, _), cand_emb in zip(candidates, candidate_embs):
                sim = cosine_similarity(query_emb, cand_emb)
                priority_boost = intent_map.get(intent_name, {}).get("priority", 2) * 0.05
                score = min(sim + priority_boost, 1.0)
                if sim > 0.35:
                    scores.append((intent_name, score))
                    
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
        except Exception as e:
            logger.warning("Semantic fallback failed: %s", e)
            return []

    @staticmethod
    def _normalize_date(raw: str) -> str:
        """Convert various date formats to YYYY-MM-DD for SQL."""
        for fmt in ("%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%b %d %Y",
                     "%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return raw  # return as-is if no format matched

    def _check_required(self, meta: Dict, entities: Dict) -> List[str]:
        missing = []
        for req in meta.get("required_entities", []):
            if entities.get(req) is None:
                missing.append(req)
        return missing

    def _build_clarification(self, intent: str, missing: List[str]) -> str:
        prompts = {
            "order_no":  "Could you provide the order number (e.g. 147190737)?",
            "serial_no": "Could you provide the serial number (e.g. RHPW133610)?",
            "line_code": "Which production line are you asking about (e.g. Line 1)?",
            "shift_id":  "Which shift — A or B?",
        }
        parts = [prompts.get(m, f"Please provide the {m.replace('_', ' ')}.") for m in missing]
        return " ".join(parts)

    def _general_fallback(self, entities: Dict) -> IntentResult:
        return IntentResult(
            intent="general",
            procedure=None,
            confidence=0.0,
            entities=entities,
            use_opc=True,
            use_rag=True,
        )