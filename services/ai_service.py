"""
AI Service – Manufacturing Copilot
Ollama / Gemini / OpenAI / Anthropic routing interface.
"""

import json
import logging
import time
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
        self._system_prompt = _load_prompt("system_prompt.txt") or SYSTEM_PROMPT_DEFAULT
        self.active_conn_id: Optional[str] = None
        self.active_config: Optional[dict] = None
        self.last_benchmark_time: Optional[float] = None
        self._is_available_cached: Optional[bool] = None
        self._last_available_check: float = 0.0
        self._failure_counts = {}
        self._disabled_connections = set()

    async def benchmark_connections(self) -> None:
        from services.connections_service import get_enabled_connections
        enabled_conns = get_enabled_connections("llm")
        
        active_candidates = [c for c in enabled_conns if c["id"] not in self._disabled_connections]
        if not active_candidates:
            logger.info("All enabled LLM connections were disabled by the circuit breaker. Resetting disabled list.")
            self._disabled_connections.clear()
            active_candidates = enabled_conns
            
        if not active_candidates:
            logger.info("No custom LLM connections enabled. Using settings defaults.")
            self.active_conn_id = None
            self.active_config = {
                "provider": settings.LLM_PROVIDER,
                "url": settings.OLLAMA_BASE_URL,
                "model": settings.OLLAMA_MODEL,
                "timeout": settings.OLLAMA_TIMEOUT,
                "online_url": settings.LLM_ONLINE_URL,
                "online_api_key": settings.GEMINI_API_KEY,
                "online_model": settings.GEMINI_MODEL,
                "online_provider": "Gemini"
            }
            self.last_benchmark_time = time.time()
            return

        logger.info("Benchmarking %d enabled LLM connections...", len(active_candidates))
        latencies = []
        for conn in active_candidates:
            latency = await self.measure_connection_latency(conn)
            logger.info("LLM Connection '%s' latency: %.3fs", conn.get("name"), latency)
            latencies.append((latency, conn))
        
        latencies.sort(key=lambda x: x[0])
        fastest_latency, fastest_conn = latencies[0]
        
        if fastest_latency == float("inf"):
            # Fallback to the manually selected active connection if available
            active_conn = next((c for c in active_candidates if c.get("active")), None)
            if active_conn:
                logger.warning("All enabled LLM connections failed latency check! Using active connection: '%s'", active_conn.get("name"))
                fastest_conn = active_conn
            else:
                logger.warning("All enabled LLM connections failed latency check! Using the first one.")
                fastest_conn = active_candidates[0]
            
        self.active_conn_id = fastest_conn["id"]
        self.active_config = fastest_conn["config"]
        self.last_benchmark_time = time.time()
        logger.info("Selected fastest LLM Connection: '%s' (id: %s)", fastest_conn.get("name"), self.active_conn_id)

    async def measure_connection_latency(self, conn: dict) -> float:
        start = time.time()
        try:
            provider = conn["config"].get("provider", "ollama")
            if provider == "ollama":
                url = conn["config"].get("url", "http://localhost:11434").rstrip("/")
                model = conn["config"].get("model", "qwen3:8b")
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "respond only with the word OK"}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": 0.0, "num_predict": 5}
                }
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(f"{url}/api/chat", json=payload)
                    if r.status_code == 200:
                        return time.time() - start
            elif provider == "online":
                online_provider = conn["config"].get("online_provider", "Gemini").lower()
                url = conn["config"].get("online_url", "").strip()
                api_key = conn["config"].get("online_api_key", "").strip()
                model = conn["config"].get("online_model", "").strip()
                if not api_key:
                    return float("inf")
                
                async with httpx.AsyncClient(timeout=10) as client:
                    if online_provider == "gemini":
                        if not url:
                            url = "https://generativelanguage.googleapis.com"
                        endpoint = f"{url.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
                        payload = {
                            "contents": [{"role": "user", "parts": [{"text": "respond only with the word OK"}]}],
                            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 5}
                        }
                        r = await client.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
                        if r.status_code == 200:
                            return time.time() - start
                    elif online_provider == "openai":
                        if not url:
                            url = "https://api.openai.com/v1"
                        endpoint = f"{url.rstrip('/')}/chat/completions"
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": "respond only with the word OK"}],
                            "temperature": 0.0,
                            "max_tokens": 5
                        }
                        r = await client.post(endpoint, json=payload, headers=headers)
                        if r.status_code == 200:
                            return time.time() - start
                    elif online_provider == "anthropic":
                        if not url:
                            url = "https://api.anthropic.com/v1"
                        endpoint = f"{url.rstrip('/')}/messages"
                        headers = {
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        }
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": "respond only with the word OK"}],
                            "max_tokens": 5,
                            "temperature": 0.0
                        }
                        r = await client.post(endpoint, json=payload, headers=headers)
                        if r.status_code in (200, 201):
                            return time.time() - start
        except Exception as e:
            logger.warning("Latency check failed for connection %s: %s", conn.get("name"), e)
        return float("inf")

    async def _ensure_active_config(self) -> None:
        now = time.time()
        if (self.last_benchmark_time is None or 
            self.active_config is None or 
            (now - self.last_benchmark_time) > 600):
            await self.benchmark_connections()

    async def chat_stream(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        await self._ensure_active_config()
        
        provider = self.active_config.get("provider", "ollama")
        t0 = time.time()
        conn_id = self.active_conn_id or "default"
        try:
            if provider == "online":
                async for token in self._chat_stream_online(user_message, context, conversation_history):
                    yield token
            else:
                async for token in self._chat_stream_offline(user_message, context, conversation_history):
                    yield token
            self._failure_counts[conn_id] = 0
        except Exception as e:
            self._failure_counts[conn_id] = self._failure_counts.get(conn_id, 0) + 1
            logger.error("LLM stream call failed for connection '%s' (fail count: %d): %s", conn_id, self._failure_counts[conn_id], e)
            if self._failure_counts[conn_id] >= 2:
                logger.warning("Circuit breaker triggered for connection '%s'. Disabling connection.", conn_id)
                if self.active_conn_id:
                    self._disabled_connections.add(self.active_conn_id)
                self._failure_counts[conn_id] = 0
            self.last_benchmark_time = None  # force re-benchmark next time
            raise e
        finally:
            elapsed = time.time() - t0
            if elapsed > 600:
                logger.warning("Query took %.1fs (>600s), resetting benchmark", elapsed)
                self.last_benchmark_time = None

    async def chat(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        await self._ensure_active_config()
        
        provider = self.active_config.get("provider", "ollama")
        t0 = time.time()
        conn_id = self.active_conn_id or "default"
        try:
            if provider == "online":
                res = await self._chat_online(user_message, context, conversation_history)
            else:
                res = await self._chat_offline(user_message, context, conversation_history)
            self._failure_counts[conn_id] = 0
            return res
        except Exception as e:
            self._failure_counts[conn_id] = self._failure_counts.get(conn_id, 0) + 1
            logger.error("LLM call failed for connection '%s' (fail count: %d): %s", conn_id, self._failure_counts[conn_id], e)
            if self._failure_counts[conn_id] >= 2:
                logger.warning("Circuit breaker triggered for connection '%s'. Disabling connection.", conn_id)
                if self.active_conn_id:
                    self._disabled_connections.add(self.active_conn_id)
                self._failure_counts[conn_id] = 0
            self.last_benchmark_time = None  # force re-benchmark next time
            raise e
        finally:
            elapsed = time.time() - t0
            if elapsed > 600:
                logger.warning("Query took %.1fs (>600s), resetting benchmark", elapsed)
                self.last_benchmark_time = None

    async def is_available(self) -> bool:
        await self._ensure_active_config()
        provider = self.active_config.get("provider", "ollama")
        if provider == "online":
            return bool(self.active_config.get("online_api_key", "").strip())
        
        now = time.time()
        if self._is_available_cached is not None and (now - self._last_available_check < 30.0):
            return self._is_available_cached

        url = self.active_config.get("url", "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{url}/api/tags")
                status = r.status_code == 200
                self._is_available_cached = status
                self._last_available_check = now
                return status
        except Exception:
            self._is_available_cached = False
            self._last_available_check = now
            return False

    async def _chat_offline(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        base = self.active_config.get("url", "http://localhost:11434").rstrip("/")
        model = self.active_config.get("model", "qwen3:8b")
        timeout = 900.0
        messages = self._build_messages(user_message, context, conversation_history)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "think": False,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "num_predict": settings.OLLAMA_MAX_TOKENS,
                "num_ctx": 16384,
            },
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base}/api/chat", json=payload)
            resp.raise_for_status()
            resp_json = resp.json()
            message = resp_json.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")
            if thinking:
                return f"<think>{thinking}</think>\n{content}"
            return content

    async def _chat_stream_offline(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        base = self.active_config.get("url", "http://localhost:11434").rstrip("/")
        model = self.active_config.get("model", "qwen3:8b")
        timeout = 900.0
        messages = self._build_messages(user_message, context, conversation_history)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
            "think": False,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "num_predict": settings.OLLAMA_MAX_TOKENS,
                "num_ctx": 16384,
            },
        }
        in_thinking = False
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{base}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content = msg.get("content", "")
                        thinking = msg.get("thinking", "")
                        
                        if thinking:
                            if not in_thinking:
                                yield "<think>"
                                in_thinking = True
                            yield thinking
                        
                        if content:
                            if in_thinking:
                                yield "</think>\n"
                                in_thinking = False
                            yield content
                            
                        if chunk.get("done"):
                            if in_thinking:
                                yield "</think>"
                            break
                    except json.JSONDecodeError:
                        continue

    async def _chat_online(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> str:
        online_provider = self.active_config.get("online_provider", "Gemini").lower()
        api_key = self.active_config.get("online_api_key", "")
        model = self.active_config.get("online_model", "")
        timeout = self.active_config.get("timeout", 30)
        
        if online_provider == "gemini":
            url = self.active_config.get("online_url") or "https://generativelanguage.googleapis.com"
            contents = []
            for turn in (conversation_history or []):
                role = "user" if turn["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn["content"]}]})
            if context:
                final_prompt = f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}"
            else:
                final_prompt = user_message
            contents.append({"role": "user", "parts": [{"text": final_prompt}]})

            payload = {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": self._system_prompt}]},
                "generationConfig": {
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "maxOutputTokens": settings.OLLAMA_MAX_TOKENS,
                }
            }
            headers = {"Content-Type": "application/json"}
            endpoint = f"{url.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                resp_json = resp.json()
                candidates = resp_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""
                
        elif online_provider == "openai":
            url = self.active_config.get("online_url") or "https://api.openai.com/v1"
            messages = [{"role": "system", "content": self._system_prompt}]
            for turn in (conversation_history or []):
                messages.append(turn)
            if context:
                messages.append({"role": "user", "content": f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}"})
            else:
                messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": settings.OLLAMA_TEMPERATURE,
                "max_tokens": settings.OLLAMA_MAX_TOKENS
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            endpoint = f"{url.rstrip('/')}/chat/completions"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                resp_json = resp.json()
                choices = resp_json.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
                
        elif online_provider == "anthropic":
            url = self.active_config.get("online_url") or "https://api.anthropic.com/v1"
            messages = []
            for turn in (conversation_history or []):
                messages.append(turn)
            if context:
                messages.append({"role": "user", "content": f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}"})
            else:
                messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": model,
                "system": self._system_prompt,
                "messages": messages,
                "max_tokens": settings.OLLAMA_MAX_TOKENS,
                "temperature": settings.OLLAMA_TEMPERATURE
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            endpoint = f"{url.rstrip('/')}/messages"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                resp_json = resp.json()
                content = resp_json.get("content", [])
                if content:
                    return content[0].get("text", "")
                return ""

        return ""

    async def _chat_stream_online(
        self,
        user_message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        online_provider = self.active_config.get("online_provider", "Gemini").lower()
        api_key = self.active_config.get("online_api_key", "")
        model = self.active_config.get("online_model", "")
        timeout = self.active_config.get("timeout", 30)

        if online_provider == "gemini":
            url = self.active_config.get("online_url") or "https://generativelanguage.googleapis.com"
            contents = []
            for turn in (conversation_history or []):
                role = "user" if turn["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn["content"]}]})
            if context:
                final_prompt = f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}"
            else:
                final_prompt = user_message
            contents.append({"role": "user", "parts": [{"text": final_prompt}]})

            payload = {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": self._system_prompt}]},
                "generationConfig": {
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "maxOutputTokens": settings.OLLAMA_MAX_TOKENS,
                }
            }
            headers = {"Content-Type": "application/json"}
            endpoint = f"{url.rstrip('/')}/v1beta/models/{model}:streamGenerateContent?key={api_key}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "{" in buffer:
                            start_idx = buffer.find("{")
                            depth = 0
                            end_idx = -1
                            for i in range(start_idx, len(buffer)):
                                if buffer[i] == "{":
                                    depth += 1
                                elif buffer[i] == "}":
                                    depth -= 1
                                    if depth == 0:
                                        end_idx = i
                                        break
                            if end_idx != -1:
                                obj_str = buffer[start_idx:end_idx+1]
                                buffer = buffer[end_idx+1:]
                                try:
                                    obj = json.loads(obj_str)
                                    candidates = obj.get("candidates", [])
                                    if candidates:
                                        parts = candidates[0].get("content", {}).get("parts", [])
                                        if parts:
                                            text = parts[0].get("text", "")
                                            if text:
                                                yield text
                                except Exception:
                                    pass
                            else:
                                break

        elif online_provider == "openai":
            url = self.active_config.get("online_url") or "https://api.openai.com/v1"
            messages = [{"role": "system", "content": self._system_prompt}]
            for turn in (conversation_history or []):
                messages.append(turn)
            if context:
                messages.append({"role": "user", "content": f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}"})
            else:
                messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": settings.OLLAMA_TEMPERATURE,
                "max_tokens": settings.OLLAMA_MAX_TOKENS,
                "stream": True
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            endpoint = f"{url.rstrip('/')}/chat/completions"
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data_obj = json.loads(data_str)
                                choices = data_obj.get("choices", [])
                                if choices:
                                    content = choices[0].get("delta", {}).get("content", "")
                                    if content:
                                        yield content
                            except Exception:
                                pass

        elif online_provider == "anthropic":
            url = self.active_config.get("online_url") or "https://api.anthropic.com/v1"
            messages = []
            for turn in (conversation_history or []):
                messages.append(turn)
            if context:
                messages.append({"role": "user", "content": f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}"})
            else:
                messages.append({"role": "user", "content": user_message})

            payload = {
                "model": model,
                "system": self._system_prompt,
                "messages": messages,
                "max_tokens": settings.OLLAMA_MAX_TOKENS,
                "temperature": settings.OLLAMA_TEMPERATURE,
                "stream": True
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            endpoint = f"{url.rstrip('/')}/messages"
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            try:
                                data_obj = json.loads(data_str)
                                event_type = data_obj.get("type")
                                if event_type == "content_block_delta":
                                    delta = data_obj.get("delta", {})
                                    text = delta.get("text", "")
                                    if text:
                                        yield text
                            except Exception:
                                pass

    def _build_messages(
        self,
        user_message: str,
        context: Optional[str],
        conversation_history: Optional[List[dict]],
    ) -> List[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]
        for turn in (conversation_history or []):
            messages.append(turn)
        if context:
            messages.append({
                "role": "user",
                "content": f"[Manufacturing data loaded]\n{context}\n\nUser Question: {user_message}",
            })
        else:
            messages.append({"role": "user", "content": user_message})
        return messages