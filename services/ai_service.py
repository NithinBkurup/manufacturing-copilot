"""
AI Service – Manufacturing Copilot
Ollama / Qwen3 14B interface.
Loads system prompt from prompts/system_prompt.txt.
"""

import json
import logging
from pathlib import Path
from typing import AsyncIterator, List, Optional

import httpx

from config.settings import settings

logger = logging.getLogger("copilot.ai")


def _load_prompt(filename: str) -> str:
    path = Path("prompts") / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    logger.warning("Prompt file not found: %s", path)
    return ""

SYSTEM_PROMPT_DEFAULT = f"""
You are Manufacturing Copilot for {settings.PLANT_NAME}.

Never identify yourself as a generic AI.

Use configured data sources only.
"""


class AIService:

    def __init__(self):
        self._base = settings.OLLAMA_BASE_URL.rstrip("/")
        self._model = settings.OLLAMA_MODEL
        self._timeout = settings.OLLAMA_TIMEOUT
        self._system_prompt = _load_prompt("system_prompt.txt") or SYSTEM_PROMPT_DEFAULT

    async def chat_stream(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(user_message, context, conversation_history)
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "num_predict": settings.OLLAMA_MAX_TOKENS,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    async def chat(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        messages = self._build_messages(user_message, context, conversation_history)
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "num_predict": settings.OLLAMA_MAX_TOKENS,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self._base}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def _build_messages(
        self,
        user_message: str,
        context: Optional[str],
        conversation_history: Optional[List[dict]],
    ) -> List[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]
        if context:
            messages.append({
                "role": "assistant",
                "content": f"[Manufacturing data loaded]\n\n{context}",
            })
        for turn in (conversation_history or []):
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})
        return messages