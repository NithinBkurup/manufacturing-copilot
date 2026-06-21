# Manufacturing Copilot — Detailed Technical Architecture

This document provides a deep-dive specification of the service structure, data mapping, caching cycles, and security boundaries implemented in the **Manufacturing Copilot**.

---

## 🛡️ Zero-Trust SQL Database Access Model

Relational databases in manufacturing (MES, ERP, and Traceability databases) hold critical recipe parameters, process logs, and production orders. Allowing LLMs to write queries directly (Text-to-SQL) introduces catastrophic failure modes, including:
1.  **Hallucinated Queries**: Generating invalid syntax or querying non-existent columns.
2.  **Inadvertent Mutating Commands**: Accidental database modifications due to prompt injection or loose syntax.
3.  **Performance Overload**: Executing un-indexed queries on production databases, causing service disruption.

To solve this, the Copilot uses **Pre-Approved Stored Procedure Registry Isolation**:

```
User Query ──> IntentService ──> [Entity Extraction] ──> Execute Approved Stored Procedure ──> Context Block ──> LLM ──> Operator
```

*   The [SQLService](file:///d:/Dev/ManufacturingCopilot/services/sql_service.py#L151) operates using a read-only SQL Server account `saapi`.
*   Direct table queries are strictly blocked.
*   Only procedures registered inside the [PROCEDURE_REGISTRY](file:///d:/Dev/ManufacturingCopilot/services/sql_service.py#L32) in [services/sql_service.py](file:///d:/Dev/ManufacturingCopilot/services/sql_service.py) can be executed.
*   The `PlantCode` parameter (set in settings to `2006`) is injected programmatically and cannot be overridden by user messages.

---

## 🎛️ Service Breakdown

### 1. [IntentService](file:///d:/Dev/ManufacturingCopilot/services/intent_service.py#L184)
*   **Location**: [services/intent_service.py](file:///d:/Dev/ManufacturingCopilot/services/intent_service.py)
*   **Role**: Pre-LLM classification and entity parser.
*   **Entity RegEx Patterns**:
    *   `OrderNo`: Exactly 9 digits starting with `1` (e.g., `147190737`).
    *   `SerialNo`: Matches axle formats (`RHPW` followed by 6+ digits), truck formats (`TRPS` followed by 6+ digits), and 17-character VIN formats.
    *   `LineCode`: Matches lines 1 to 4 (`Line 1`, `L3`, `assembly line 2`).
    *   `ShiftID`: Matches Shift `A` or `B`.
    *   `ShiftDate`: Matches `YYYY-MM-DD` or `DD/MM/YYYY`.
*   **Intent Resolution**: Checks keywords mapped to specific intents. If required parameters are missing for an intent (e.g., querying order status without an order number), it halts and triggers a parameter clarification prompt.

### 2. [ManufacturingService](file:///d:/Dev/ManufacturingCopilot/services/manufacturing_service.py#L33)
*   **Location**: [services/manufacturing_service.py](file:///d:/Dev/ManufacturingCopilot/services/manufacturing_service.py)
*   **Role**: Data-layer orchestrator. It executes the mapped queries and compiles the output:
    1.  Calls `IntentService.classify` to parse the query.
    2.  If a stored procedure is mapped and parameters are complete, it invokes `SQLService.execute_procedure`.
    3.  If live tag status or general queries are requested, it fetches the thread-safe OPC cache snapshot.
    4.  If manual or documentation search is requested, it queries `VectorService.get_context_for_ai`.
    5.  Joins all three components using clear, structured delimiters (`---`) to construct the final system-prompt context.

### 3. [OPCService](file:///d:/Dev/ManufacturingCopilot/services/opc_service.py#L88) and [CacheService](file:///d:/Dev/ManufacturingCopilot/services/cache_service.py#L16)
*   **Location**: [services/opc_service.py](file:///d:/Dev/ManufacturingCopilot/services/opc_service.py), [services/cache_service.py](file:///d:/Dev/ManufacturingCopilot/services/cache_service.py)
*   **Role**: Industrial PLC tag caching.
*   **Mechanics**:
    *   Direct PLC connection polling per user chat query is strictly prohibited.
    *   An asynchronous background task (`OPCService._cache_loop`) runs every 5 seconds, retrieving registered tags from the Takebishi DXP server.
    *   Updates are saved in a thread-safe, in-memory [CacheService](file:///d:/Dev/ManufacturingCopilot/services/cache_service.py#L16) singleton.
    *   If the OPC UA library or server is unavailable, `OPCService` automatically runs in **simulation mode**, populating the cache with realistic production cycles for offline development and testing.

### 4. [VectorService](file:///d:/Dev/ManufacturingCopilot/services/vector_service.py#L20)
*   **Location**: [services/vector_service.py](file:///d:/Dev/ManufacturingCopilot/services/vector_service.py)
*   **Role**: RAG document engine.
*   **Configuration**:
    *   **Vector Database**: ChromaDB Persistent Client.
    *   **Embeddings**: Local `all-MiniLM-L6-v2` transformer model (saves cache inside `sentence_transformers_cache`).
    *   **Chunking**: Chunks of 800 characters with 100 characters overlap.
    *   **Extraction Support**: Extracted text from PDF (`pdfplumber`), DOCX (`python-docx`), PPTX (`python-pptx`), XLSX (`openpyxl`), and TXT files.

### 5. [AIService](file:///d:/Dev/ManufacturingCopilot/services/ai_service.py#L35)
*   **Location**: [services/ai_service.py](file:///d:/Dev/ManufacturingCopilot/services/ai_service.py)
*   **Role**: Ollama/Qwen3 controller.
*   **Mechanics**:
    *   Loads prompt configuration from `prompts/system_prompt.txt`.
    *   Maintains conversation history and inserts the compiled context string from `ManufacturingService` inside an assistant message wrapper: `[Manufacturing data loaded]`.
    *   Streams tokens asynchronously using FastAPI's `StreamingResponse` using Server-Sent Events (SSE).
