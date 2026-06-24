"""
Manufacturing Copilot — Central Configuration
Plant: configurable via .env
All values must be overridden in .env before production deployment.
No plant-specific defaults exist here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 3020
    APP_TITLE: str = "Manufacturing Copilot"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # Plant Identity — MUST be set in .env
    # ------------------------------------------------------------------
    PLANT_NAME: str = "PLANT"          # e.g. PNRMPAS — set in .env
    PLANT_CODE: str = "0000"           # e.g. 2006   — set in .env, never from user
    SERVER_NAME: str = "ManufacturingCopilot"

    # ------------------------------------------------------------------
    # SQL Server — MUST be set in .env
    # ------------------------------------------------------------------
    SQL_SERVER: str = "localhost"
    SQL_DATABASE: str = "Production_DB"
    SQL_USERNAME: str = "dbuser"
    SQL_PASSWORD: str = ""             # MUST be set in .env
    SQL_DRIVER: str = "ODBC Driver 17 for SQL Server"
    SQL_TIMEOUT: int = 30

    @property
    def sql_connection_string(self) -> str:
        return (
            f"DRIVER={{{self.SQL_DRIVER}}};"
            f"SERVER={self.SQL_SERVER};"
            f"DATABASE={self.SQL_DATABASE};"
            f"UID={self.SQL_USERNAME};"
            f"PWD={self.SQL_PASSWORD};"
            f"Connection Timeout={self.SQL_TIMEOUT};"
        )

    # ------------------------------------------------------------------
    # OPC UA — MUST be set in .env
    # ------------------------------------------------------------------
    OPC_SERVER_URL: str = "opc.tcp://localhost:4840"
    OPC_NAMESPACE: str = "2"
    OPC_CACHE_INTERVAL_SEC: int = 5
    OPC_CONNECTION_TIMEOUT: int = 10

    # ------------------------------------------------------------------
    # AI Model Runtime
    # ------------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:14b"
    OLLAMA_TIMEOUT: int = 180
    OLLAMA_MAX_TOKENS: int = 2048
    OLLAMA_TEMPERATURE: float = 0.1
    LLM_PROVIDER: str = "ollama"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.5"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-opus"
    # ------------------------------------------------------------------
    # Knowledge Base (Vector Search)
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = "vector_db"
    CHROMA_COLLECTION_DOCS: str = "manufacturing_docs"
    CHROMA_TOP_K: int = 5
    SENTENCE_TRANSFORMERS_CACHE: str = "sentence_transformers_cache"
    KB_UPLOAD_DIR: str = ""
    KB_CHUNK_SIZE: int = 800
    KB_CHUNK_OVERLAP: int = 100

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/copilot.log"


settings = Settings()


def update_env_values(updates: dict):
    import os
    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            pass

    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            parts = stripped.split("=", 1)
            k = parts[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}\n")
                updated_keys.add(k)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in updated_keys:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append(f"{k}={v}\n")

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception:
        pass