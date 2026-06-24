"""
Manufacturing Copilot — Main Application
All plant-specific values come from .env — never hardcoded here.

Architecture:
  Operator → Web Chat → FastAPI → Intent Engine
  → Manufacturing Service → SQL / OPC / RAG → AI Model → Response

Rule: LLM never touches the database.
      All business logic flows through manufacturing_service.py only.
      Services only start when explicitly enabled via admin panel.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import os
import json
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx

from config.settings import settings
from services.ai_service import AIService
from services.cache_service import cache
from services.manufacturing_service import ManufacturingService
from services.opc_service import OPCService
from services.vector_service import VectorService
from services.sql_service import SQLService, PROCEDURE_REGISTRY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE, encoding="utf-8"),
    ],
)
logging.getLogger("asyncua").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logger = logging.getLogger("copilot")


# ---------------------------------------------------------------------------
# Service Registry — all services start DISABLED
# ---------------------------------------------------------------------------
class ServiceRegistry:
    def __init__(self):
        self.enabled: Dict[str, bool] = {
            "llm": False, "sql": False, "opc": False, "kb": False
        }
        import os
        import json
        config_path = "config/services.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    for k, v in saved.items():
                        if k in self.enabled:
                            self.enabled[k] = bool(v)
            except Exception:
                pass

        self.opc_service: Optional[OPCService] = None
        self.vector_service: Optional[VectorService] = None
        self.ai_service: Optional[AIService] = None
        self.manufacturing_service: Optional[ManufacturingService] = None

svc = ServiceRegistry()


# ---------------------------------------------------------------------------
# Lifespan — minimal startup, no auto-connect
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Manufacturing Copilot starting up")
    logger.info("Plant: %s | DB: %s | Model: %s",
                settings.PLANT_NAME, settings.SQL_DATABASE, settings.OLLAMA_MODEL)
    logger.info("=" * 60)

    svc.opc_service = OPCService()
    svc.vector_service = VectorService()
    svc.ai_service = AIService()
    svc.manufacturing_service = ManufacturingService(vector_service=svc.vector_service)

    app.state.vector_service = svc.vector_service
    app.state.opc_service = svc.opc_service
    app.state.manufacturing_service = svc.manufacturing_service
    app.state.ai_service = svc.ai_service

    # Auto-initialize saved active services
    if svc.enabled.get("llm"):
        ok = await svc.ai_service.is_available()
        if not ok:
            logger.warning("Auto-start: AI service is not reachable, disabling.")
            svc.enabled["llm"] = False

    if svc.enabled.get("sql"):
        try:
            sql = SQLService()
            conn = sql._get_connection()
            conn.close()
            svc.manufacturing_service._sql = sql
        except Exception as e:
            logger.warning("Auto-start: SQL Database connection failed: %s. Disabling.", e)
            svc.enabled["sql"] = False

    if svc.enabled.get("opc"):
        try:
            await svc.opc_service.start_cache_loop()
        except Exception as e:
            logger.warning("Auto-start: OPC UA Cache Loop failed: %s. Disabling.", e)
            svc.enabled["opc"] = False

    if svc.enabled.get("kb"):
        try:
            await svc.vector_service.initialize()
            svc.manufacturing_service._vector = svc.vector_service
        except Exception as e:
            logger.warning("Auto-start: Knowledge Base initialization failed: %s. Disabling.", e)
            svc.enabled["kb"] = False

    logger.info("Ready on port %d — open http://localhost:%d", settings.APP_PORT, settings.APP_PORT)
    yield

    if svc.enabled.get("opc") and svc.opc_service:
        await svc.opc_service.stop_cache_loop()
    logger.info("Manufacturing Copilot shut down cleanly")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_TITLE,
    description="Manufacturing Intelligence Engine",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    user: Optional[str] = None

class IndexDocumentRequest(BaseModel):
    file_path: str

class ServiceToggleRequest(BaseModel):
    enabled: bool

class LLMConfigRequest(BaseModel):
    url: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_length: Optional[int] = None

class SQLConfigRequest(BaseModel):
    name: Optional[str] = None
    server: Optional[str] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    driver: Optional[str] = None
    timeout: Optional[int] = None

class OPCConfigRequest(BaseModel):
    url: Optional[str] = None
    namespace: Optional[str] = None
    security_policy: Optional[str] = None
    message_security: Optional[str] = None
    poll_interval: Optional[int] = None
    connection_timeout: Optional[int] = None

class KBConfigRequest(BaseModel):
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    server_path: Optional[str] = None
    upload_dir: Optional[str] = None

class ProceduresSaveRequest(BaseModel):
    procedures: List[Dict[str, Any]]

class PlantConfigRequest(BaseModel):
    plant_name: Optional[str] = None
    plant_code: Optional[str] = None
    server_name: Optional[str] = None


# ---------------------------------------------------------------------------
# UI Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/home")

@app.get("/home", response_class=HTMLResponse)
async def home_ui(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/setting", response_class=HTMLResponse)
async def setting_ui(request: Request):
    return templates.TemplateResponse("setting.html", {"request": request})

@app.get("/Setting", response_class=HTMLResponse)
async def setting_ui_cap(request: Request):
    return RedirectResponse(url="/setting")

@app.get("/admin", response_class=HTMLResponse)
async def admin_redirect(request: Request):
    return RedirectResponse(url="/setting")


# ---------------------------------------------------------------------------
# Config API — plant identity, read dynamically by frontend
# ---------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    """
    Returns all non-secret configuration values.
    Frontend uses this to display plant name, server, model — no hardcoding in HTML.
    """
    return {
        "plant_name": settings.PLANT_NAME,
        "plant_code": settings.PLANT_CODE,
        "server_name": settings.SERVER_NAME,
        "sql_server": settings.SQL_SERVER,
        "sql_database": settings.SQL_DATABASE,
        "sql_username": settings.SQL_USERNAME,
        "opc_server_url": settings.OPC_SERVER_URL,
        "opc_namespace": settings.OPC_NAMESPACE,
        "opc_poll_interval": settings.OPC_CACHE_INTERVAL_SEC,
        "llm_url": settings.OLLAMA_BASE_URL,
        "llm_model": settings.OLLAMA_MODEL,
        "llm_temperature": settings.OLLAMA_TEMPERATURE,
        "llm_max_tokens": settings.OLLAMA_MAX_TOKENS,
        "llm_timeout": settings.OLLAMA_TIMEOUT,
        "kb_top_k": settings.CHROMA_TOP_K,
        "app_version": settings.APP_VERSION,
        "services": svc.enabled,
    }

@app.post("/api/admin/plant/config")
async def save_plant_config(body: PlantConfigRequest):
    """Update plant identity at runtime (persisted in .env file)."""
    updates = {}
    if body.plant_name:
        settings.PLANT_NAME = body.plant_name
        updates["PLANT_NAME"] = body.plant_name
    if body.plant_code:
        settings.PLANT_CODE = body.plant_code
        updates["PLANT_CODE"] = body.plant_code
    if body.server_name:
        settings.SERVER_NAME = body.server_name
        updates["SERVER_NAME"] = body.server_name
    if updates:
        from config.settings import update_env_values
        update_env_values(updates)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    mfg: ManufacturingService = request.app.state.manufacturing_service
    ai: AIService = request.app.state.ai_service

    if not svc.enabled.get("llm"):
        async def svc_offline():
            msg = "AI Model is currently offline. Please enable it in Setting → AI Model."
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'data_sources': []})}\n\n"
        return StreamingResponse(
            svc_offline(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await mfg.handle_query(body.message)

    if result["clarification"]:
        async def clarify_stream():
            yield f"data: {json.dumps({'token': result['clarification']})}\n\n"
            yield f"data: {json.dumps({'done': True, 'data_sources': []})}\n\n"
        return StreamingResponse(
            clarify_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    history = [{"role": m.role, "content": m.content} for m in (body.history or [])]
    context = result["context"]
    data_sources = result["data_sources"]

    logger.info("QUERY | plant=%s | intent=%s | proc=%s | sources=%s | user=%s | query=%s",
        settings.PLANT_NAME, result["intent"].intent,
        result["intent"].procedure, data_sources, body.user or "anonymous", body.message[:80])

    async def event_stream():
        try:
            async for token in ai.chat_stream(body.message, context, history):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'data_sources': data_sources})}\n\n"
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    mfg: ManufacturingService = request.app.state.manufacturing_service
    ai: AIService = request.app.state.ai_service
    if not svc.enabled.get("llm"):
        return {"reply": "AI Model is offline. Enable it in Setting → AI Model.", "data_sources": []}
    result = await mfg.handle_query(body.message)
    if result["clarification"]:
        return {"reply": result["clarification"], "data_sources": []}
    history = [{"role": m.role, "content": m.content} for m in (body.history or [])]
    reply = await ai.chat(body.message, result["context"], history)
    return {"reply": reply, "data_sources": result["data_sources"]}


# ---------------------------------------------------------------------------
# Admin — Service Toggle
# ---------------------------------------------------------------------------
@app.post("/api/admin/service/{name}")
async def toggle_service(name: str, body: ServiceToggleRequest, request: Request):
    if name not in svc.enabled:
        return JSONResponse({"error": "Unknown service"}, status_code=404)

    if body.enabled:
        if name == "llm":
            ok = await svc.ai_service.is_available()
            if not ok:
                return JSONResponse({
                    "status": "error",
                    "error": f"Cannot connect to AI engine at {settings.OLLAMA_BASE_URL}"
                })
            svc.enabled["llm"] = True

        elif name == "sql":
            try:
                sql = SQLService()
                conn = sql._get_connection()
                conn.close()
                svc.enabled["sql"] = True
                request.app.state.manufacturing_service._sql = sql
            except Exception as e:
                return JSONResponse({"status": "error", "error": str(e)})

        elif name == "opc":
            await svc.opc_service.start_cache_loop()
            svc.enabled["opc"] = True

        elif name == "kb":
            await svc.vector_service.initialize()
            svc.enabled["kb"] = True
            request.app.state.manufacturing_service._vector = svc.vector_service
    else:
        if name == "opc" and svc.opc_service:
            await svc.opc_service.stop_cache_loop()
        svc.enabled[name] = False

    # Write enabled states to config/services.json
    try:
        import json
        with open("config/services.json", "w", encoding="utf-8") as f:
            json.dump(svc.enabled, f, indent=2)
    except Exception:
        pass

    return {"status": "ok", "enabled": svc.enabled[name]}


@app.get("/api/admin/services/status")
async def services_status():
    return {name: {"enabled": en} for name, en in svc.enabled.items()}


@app.get("/api/admin/service/{name}/status")
async def service_status(name: str):
    if name == "llm":
        return {
            "engine": "Local AI Engine",
            "model": settings.OLLAMA_MODEL,
            "context_length": 4096,
            "compute": "CPU",
            "enabled": svc.enabled.get("llm", False),
        }
    if name == "sql":
        return {
            "name": "Primary",
            "server": settings.SQL_SERVER,
            "database": settings.SQL_DATABASE,
            "username": settings.SQL_USERNAME,
            "connected": svc.enabled.get("sql", False),
        }
    if name == "opc":
        return svc.opc_service.status if svc.opc_service else {}
    if name == "kb":
        count = svc.vector_service.document_count if svc.vector_service else 0
        size_mb = 0.0
        if os.path.exists(settings.CHROMA_PERSIST_DIR):
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(settings.CHROMA_PERSIST_DIR):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
                size_mb = round(total_size / (1024 * 1024), 1)
            except Exception:
                pass
        return {
            "document_chunks": count,
            "document_count": count,
            "enabled": svc.enabled.get("kb", False),
            "size_mb": size_mb,
            "top_k": settings.CHROMA_TOP_K,
            "chunk_size": settings.KB_CHUNK_SIZE,
            "chunk_overlap": settings.KB_CHUNK_OVERLAP,
            "upload_dir": settings.KB_UPLOAD_DIR,
        }
    return {}


# ---------------------------------------------------------------------------
# Admin — LLM Config
# ---------------------------------------------------------------------------
@app.post("/api/admin/llm/test")
async def llm_test(body: dict):
    url = body.get("url", settings.OLLAMA_BASE_URL).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{url}/api/tags")
            if r.status_code == 200:
                data = r.json()
                models = [
                    {
                        "name": m["name"],
                        "size": f"{m.get('size', 0) // 1073741824:.1f}GB",
                        "modified": m.get("modified_at", "")[:10],
                    }
                    for m in data.get("models", [])
                ]
                return {"ok": True, "engine": "Local AI Engine",
                        "model_count": len(models), "models": models,
                        "active_model": settings.OLLAMA_MODEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Unexpected response"}

@app.get("/api/admin/llm/models")
async def llm_models():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            data = r.json()
            models = [
                {
                    "name": m["name"],
                    "size": f"{m.get('size', 0) // 1073741824:.1f}GB",
                    "modified": m.get("modified_at", "")[:10],
                }
                for m in data.get("models", [])
            ]
            return {"models": models, "active_model": settings.OLLAMA_MODEL}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.post("/api/admin/llm/config")
async def llm_config(body: LLMConfigRequest):
    updates = {}
    if body.model:
        settings.OLLAMA_MODEL = body.model
        svc.ai_service._model = body.model
        updates["OLLAMA_MODEL"] = body.model
    if body.url:
        settings.OLLAMA_BASE_URL = body.url
        svc.ai_service._base = body.url.rstrip("/")
        updates["OLLAMA_BASE_URL"] = body.url
    if body.temperature is not None:
        settings.OLLAMA_TEMPERATURE = body.temperature
        updates["OLLAMA_TEMPERATURE"] = str(body.temperature)
    if body.max_tokens:
        settings.OLLAMA_MAX_TOKENS = body.max_tokens
        updates["OLLAMA_MAX_TOKENS"] = str(body.max_tokens)
    if body.timeout:
        settings.OLLAMA_TIMEOUT = body.timeout
        svc.ai_service._timeout = body.timeout
        updates["OLLAMA_TIMEOUT"] = str(body.timeout)
    if updates:
        from config.settings import update_env_values
        update_env_values(updates)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin — SQL Config
# ---------------------------------------------------------------------------
@app.post("/api/admin/sql/test")
async def sql_test(body: SQLConfigRequest):
    import pyodbc
    conn_str = (
        f"DRIVER={{{body.driver or settings.SQL_DRIVER}}};"
        f"SERVER={body.server or settings.SQL_SERVER};"
        f"DATABASE={body.database or settings.SQL_DATABASE};"
        f"UID={body.username or settings.SQL_USERNAME};"
        f"PWD={body.password or settings.SQL_PASSWORD};"
        f"Connection Timeout=10;"
    )
    try:
        conn = pyodbc.connect(conn_str, autocommit=False)
        cursor = conn.cursor()
        cursor.execute("SELECT @@SERVERNAME, DB_NAME(), @@VERSION")
        row = cursor.fetchone()
        conn.close()
        server = str(row[0]) if row else "unknown"
        db = str(row[1]) if row else "unknown"
        ver = str(row[2])[:80] if row else "Connected"
        return {"ok": True, "server": server, "database": db, "message": ver}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/admin/sql/config")
async def sql_config_save(body: SQLConfigRequest):
    updates = {}
    if body.server is not None:
        settings.SQL_SERVER = body.server
        updates["SQL_SERVER"] = body.server
    if body.database is not None:
        settings.SQL_DATABASE = body.database
        updates["SQL_DATABASE"] = body.database
    if body.username is not None:
        settings.SQL_USERNAME = body.username
        updates["SQL_USERNAME"] = body.username
    if body.password is not None:
        settings.SQL_PASSWORD = body.password
        updates["SQL_PASSWORD"] = body.password
    if body.driver is not None:
        settings.SQL_DRIVER = body.driver
        updates["SQL_DRIVER"] = body.driver
    if body.timeout is not None:
        settings.SQL_TIMEOUT = body.timeout
        updates["SQL_TIMEOUT"] = str(body.timeout)
    if updates:
        from config.settings import update_env_values
        update_env_values(updates)
    return {"status": "ok"}


@app.post("/api/admin/sql/databases")
async def sql_databases_list(body: SQLConfigRequest):
    import pyodbc
    conn_str = (
        f"DRIVER={{{body.driver or settings.SQL_DRIVER}}};"
        f"SERVER={body.server or settings.SQL_SERVER};"
        f"DATABASE=master;"
        f"UID={body.username or settings.SQL_USERNAME};"
        f"PWD={body.password or settings.SQL_PASSWORD};"
        f"Connection Timeout=10;"
    )
    try:
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
        dbs = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {"status": "ok", "databases": dbs}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Admin — OPC Config
# ---------------------------------------------------------------------------
@app.post("/api/admin/opc/discover")
async def opc_discover(body: dict):
    url = body.get("url", settings.OPC_SERVER_URL)
    if url.startswith("cpo.tcp://"):
        url = "opc.tcp://" + url[10:]
    try:
        from asyncua import Client as OPCClient
        endpoints = []
        async with OPCClient(url=url) as client:
            eps = await client.connect_and_get_server_endpoints()
            for ep in eps:
                endpoints.append({
                    "url": str(ep.EndpointUrl),
                    "security_policy": str(ep.SecurityPolicyUri).split("#")[-1],
                    "message_security": str(ep.SecurityMode),
                })
        return {"endpoints": endpoints}
    except Exception as e:
        return {"endpoints": [], "error": str(e)}

@app.post("/api/admin/opc/test")
async def opc_test(body: dict):
    url = body.get("url", settings.OPC_SERVER_URL)
    if url.startswith("cpo.tcp://"):
        url = "opc.tcp://" + url[10:]
    try:
        from asyncua import Client as OPCClient
        async with OPCClient(url=url) as client:
            return {"ok": True, "namespace": body.get("namespace", 2), "node_count": "connected"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/admin/opc/browse")
async def opc_browse(body: dict):
    url = body.get("url", settings.OPC_SERVER_URL)
    if url.startswith("cpo.tcp://"):
        url = "opc.tcp://" + url[10:]
    try:
        from asyncua import Client as OPCClient
        from asyncua import ua
        nodes = []
        async with OPCClient(url=url) as client:
            root = client.get_objects_node()
            children = await root.get_children()
            for child in children[:50]:
                try:
                    name = (await child.read_browse_name()).Name
                    nc = await child.read_node_class()
                    node_type = "Tag" if nc == ua.NodeClass.Variable else "Object"
                    nodes.append({"name": name, "node_id": str(child.nodeid), "type": node_type})
                except Exception:
                    pass
        return {"nodes": nodes}
    except Exception as e:
        return {"nodes": [], "error": str(e)}

@app.post("/api/admin/opc/config")
async def opc_config_save(body: OPCConfigRequest):
    updates = {}
    if body.url:
        url = body.url
        if url.startswith("cpo.tcp://"):
            url = "opc.tcp://" + url[10:]
        settings.OPC_SERVER_URL = url
        updates["OPC_SERVER_URL"] = url
        if svc.opc_service:
            svc.opc_service._url = url
    if body.namespace:
        settings.OPC_NAMESPACE = body.namespace
        updates["OPC_NAMESPACE"] = body.namespace
    if body.poll_interval:
        settings.OPC_CACHE_INTERVAL_SEC = body.poll_interval
        updates["OPC_CACHE_INTERVAL_SEC"] = str(body.poll_interval)
    if body.connection_timeout:
        settings.OPC_CONNECTION_TIMEOUT = body.connection_timeout
        updates["OPC_CONNECTION_TIMEOUT"] = str(body.connection_timeout)
    if updates:
        from config.settings import update_env_values
        update_env_values(updates)
    return {"status": "ok"}

@app.get("/api/opc/status")
async def opc_status(request: Request):
    return request.app.state.opc_service.status

@app.get("/api/opc/cache")
async def opc_cache_view():
    snapshot = await cache.snapshot()
    return {"cache": snapshot, "tag_count": cache.tag_count}


# ---------------------------------------------------------------------------
# Admin — Knowledge Base
# ---------------------------------------------------------------------------
@app.post("/api/admin/kb/upload")
async def kb_upload(file: UploadFile = File(...)):
    if not svc.enabled.get("kb"):
        return JSONResponse({"error": "Knowledge Base is offline — enable it in Setting first"}, status_code=400)
    
    filename = file.filename
    # Check duplicate in ChromaDB
    if svc.vector_service and svc.vector_service._collection is not None:
        existing = svc.vector_service._collection.get(where={"source": filename})
        if existing and existing.get("ids"):
            return JSONResponse({"error": f"File '{filename}' is already indexed in the Knowledge Base."}, status_code=409)

    upload_dir = settings.KB_UPLOAD_DIR.strip() if settings.KB_UPLOAD_DIR else ""
    if not upload_dir:
        upload_dir = "knowledge_base_files"
    
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, filename)

    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        return JSONResponse({"error": f"Failed to save uploaded file: {str(e)}"}, status_code=500)

    try:
        chunks = await svc.vector_service.index_document(dest_path, original_name=filename)
        return {"status": "ok", "chunks_indexed": chunks, "file": filename}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/index-document")
async def index_document(body: IndexDocumentRequest, request: Request):
    if not svc.enabled.get("kb"):
        return JSONResponse({"detail": "Knowledge Base is offline — enable it in Setting → Knowledge Base."}, status_code=400)
    vector: VectorService = request.app.state.vector_service
    try:
        chunks = await vector.index_document(body.file_path)
        return {"status": "ok", "chunks_indexed": chunks, "file": body.file_path}
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=409)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)

@app.get("/api/admin/knowledge-stats")
async def knowledge_stats(request: Request):
    vector: VectorService = request.app.state.vector_service
    files = vector.indexed_files if vector else []
    return {
        "document_chunks": vector.document_count if vector else 0,
        "document_count": len(files),
        "collection": settings.CHROMA_COLLECTION_DOCS,
        "persist_dir": settings.CHROMA_PERSIST_DIR,
        "files": files,
    }

@app.post("/api/admin/kb/clear")
async def kb_clear():
    try:
        if svc.vector_service and svc.vector_service._collection:
            svc.vector_service._collection.delete(where={"source": {"$ne": ""}})
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/kb/config")
async def kb_config(body: KBConfigRequest):
    updates = {}
    if body.top_k is not None:
        settings.CHROMA_TOP_K = body.top_k
        updates["CHROMA_TOP_K"] = str(body.top_k)
    if body.chunk_size is not None:
        settings.KB_CHUNK_SIZE = body.chunk_size
        updates["KB_CHUNK_SIZE"] = str(body.chunk_size)
    if body.chunk_overlap is not None:
        settings.KB_CHUNK_OVERLAP = body.chunk_overlap
        updates["KB_CHUNK_OVERLAP"] = str(body.chunk_overlap)
    if body.upload_dir is not None:
        settings.KB_UPLOAD_DIR = body.upload_dir
        updates["KB_UPLOAD_DIR"] = body.upload_dir
    if updates:
        from config.settings import update_env_values
        update_env_values(updates)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Admin — Stored Procedures
# ---------------------------------------------------------------------------
@app.get("/api/admin/procedures")
async def list_procedures(request: Request):
    mfg: ManufacturingService = request.app.state.manufacturing_service
    return {"procedures": mfg.get_procedure_registry()}

@app.post("/api/admin/sql/procedures")
async def sql_procedures_list(body: SQLConfigRequest):
    import pyodbc
    conn_str = (
        f"DRIVER={{{body.driver or settings.SQL_DRIVER}}};"
        f"SERVER={body.server or settings.SQL_SERVER};"
        f"DATABASE={body.database or settings.SQL_DATABASE};"
        f"UID={body.username or settings.SQL_USERNAME};"
        f"PWD={body.password or settings.SQL_PASSWORD};"
        f"Connection Timeout=10;"
    )
    try:
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        procs = []
        for row in cursor.procedures():
            proc_name = row[2]
            if proc_name and not proc_name.startswith("sp_") and not proc_name.startswith("dt_"):
                procs.append(proc_name)
        conn.close()
        return {"status": "ok", "procedures": sorted(list(set(procs)))}
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)

@app.post("/api/admin/procedures/save")
async def save_procedures(body: ProceduresSaveRequest):
    PROCEDURE_REGISTRY.clear()
    for proc in body.procedures:
        name = proc.get("procedure", "")
        if name:
            params = proc.get("params", [])
            if isinstance(params, str):
                params = [p.strip() for p in params.split(",") if p.strip()]
            PROCEDURE_REGISTRY[name] = {
                "category": proc.get("category", "order"),
                "description": proc.get("description", ""),
                "intent": proc.get("intent", ""),
                "columns_to_consider": proc.get("columns_to_consider", ""),
                "params": params,
                "example_questions": proc.get("example_questions", [])
            }
    
    import services.sql_service
    services.sql_service.APPROVED_PROCEDURES = set(PROCEDURE_REGISTRY.keys())
    
    try:
        import json
        config_path = "config/procedures.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(PROCEDURE_REGISTRY, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save procedures config: %s", e)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
        
    return {"status": "ok", "count": len(body.procedures)}

@app.get("/api/admin/intent-routing")
async def intent_routing(request: Request):
    mfg: ManufacturingService = request.app.state.manufacturing_service
    return {"routing": mfg.get_intent_routing_table()}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health(request: Request):
    ai: AIService = request.app.state.ai_service
    vector: VectorService = request.app.state.vector_service
    opc: OPCService = request.app.state.opc_service
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "plant": settings.PLANT_NAME,          # from .env, not hardcoded
        "database": settings.SQL_DATABASE,     # from .env, not hardcoded
        "model": settings.OLLAMA_MODEL,
        "services": svc.enabled,
        "llm_available": await ai.is_available() if svc.enabled.get("llm") else False,
        "opc_connected": opc.status["connected"] if svc.enabled.get("opc") else False,
        "kb_documents": vector.document_count,
        "opc_cache_tags": cache.tag_count,
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=settings.DEBUG)