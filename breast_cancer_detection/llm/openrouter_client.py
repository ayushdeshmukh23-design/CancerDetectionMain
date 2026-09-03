from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator, List, Optional

try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class OpenRouterLLMClient:
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_PRIMARY_MODEL = "meta-llama/llama-3.1-8b-instruct"
    DEFAULT_FALLBACK_MODEL = "mistralai/mistral-7b-instruct"

    _shared: Optional["OpenRouterLLMClient"] = None
    _shared_lock = threading.Lock()
    _last_verify = 0.0
    _verify_ttl = 120.0

    def __init__(self, verify: bool = True):
        self.api_key = self._discover_api_key()
        self.primary_model = os.getenv("OPENROUTER_MODEL", self.DEFAULT_PRIMARY_MODEL).strip()
        self.fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", self.DEFAULT_FALLBACK_MODEL).strip()
        self.is_available = False
        if verify:
            self._verify_api()

    @classmethod
    def get_shared(cls) -> "OpenRouterLLMClient":
        with cls._shared_lock:
            now = time.time()
            if cls._shared is None:
                cls._shared = cls(verify=True)
                cls._last_verify = now
            elif now - cls._last_verify > cls._verify_ttl:
                cls._shared.api_key = cls._shared._discover_api_key()
                cls._shared._verify_api()
                cls._last_verify = now
            return cls._shared

    def _discover_api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if key:
            return key
        # Check candidate .env locations
        current_dir = Path(__file__).resolve()
        candidate_paths = [
            current_dir.parents[2] / ".env",
            current_dir.parents[1] / ".env",
            Path.cwd() / ".env",
            Path.cwd() / "CancerDetectionProject-" / ".env",
        ]
        for env_path in candidate_paths:
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY="):
                            parsed = line.split("=", 1)[1].strip().strip("\"'")
                            if parsed:
                                os.environ["OPENROUTER_API_KEY"] = parsed
                                return parsed
                except Exception:
                    continue
        return ""

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ayushdeshmukh23-design/CancerDetectionProject-",
            "X-Title": "OncoVision AI",
        }

    def _verify_api(self):
        self.api_key = self._discover_api_key()
        if not self.api_key:
            self.is_available = False
            return
        # A key is provided and formatted correctly
        self.is_available = True

    def _call_http(self, payload: dict) -> dict:
        headers = self._headers()
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.OPENROUTER_API_URL,
            data=data_bytes,
            headers=headers,
            method="POST",
        )
        timeout = int(os.getenv("OPENROUTER_TIMEOUT", "15"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        model: Optional[str] = None,
        web_search: bool = False,
    ) -> str:
        if not self.is_available or not self.api_key:
            return (
                "OpenRouter API key is not configured or reachable. "
                "Please ensure OPENROUTER_API_KEY is set in your .env file."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        models_to_try = []
        if model:
            models_to_try.append(model)
        models_to_try.append(self.primary_model)
        if self.fallback_model and self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)

        last_error = None
        for target_model in models_to_try:
            try:
                payload: dict = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if web_search:
                    payload["plugins"] = [{"id": "web"}]

                res = self._call_http(payload)
                choices = res.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
            except Exception as exc:
                last_error = exc
                # Try once without web plugin if plugin failed
                if web_search:
                    try:
                        payload_no_web = {
                            "model": target_model,
                            "messages": messages,
                            "temperature": temperature,
                        }
                        res = self._call_http(payload_no_web)
                        choices = res.get("choices", [])
                        if choices:
                            return choices[0].get("message", {}).get("content", "")
                    except Exception:
                        pass
                continue

        return (
            f"LLM service temporarily unavailable ({last_error}). "
            "I can still provide static report metrics and findings."
        )

    def stream(
        self,
        prompt: str,
        system: str = "",
        history: List[dict] | None = None,
        model: Optional[str] = None,
        web_search: bool = False,
    ) -> Generator[str, None, None]:
        if not self.is_available or not self.api_key:
            yield (
                "OpenRouter API key is not configured. "
                "Please verify that OPENROUTER_API_KEY is defined in your .env file."
            )
            return

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})

        models_to_try = []
        if model:
            models_to_try.append(model)
        models_to_try.append(self.primary_model)
        if self.fallback_model and self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)

        for target_model in models_to_try:
            try:
                payload = {
                    "model": target_model,
                    "messages": messages,
                    "stream": True,
                }
                if web_search:
                    payload["plugins"] = [{"id": "web"}]
                headers = self._headers()

                if HAS_REQUESTS:
                    with requests.post(
                        self.OPENROUTER_API_URL,
                        json=payload,
                        headers=headers,
                        stream=True,
                        timeout=45,
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line:
                                continue
                            line_str = line.strip()
                            if line_str.startswith(":"):
                                continue  # SSE keepalive comment
                            if line_str.startswith("data:"):
                                data_part = line_str[5:].strip()
                                if data_part == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_part)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except Exception:
                                    continue
                        return
                else:
                    data_bytes = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        self.OPENROUTER_API_URL,
                        data=data_bytes,
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        while True:
                            line_bytes = resp.readline()
                            if not line_bytes:
                                break
                            line_str = line_bytes.decode("utf-8").strip()
                            if not line_str or line_str.startswith(":"):
                                continue
                            if line_str.startswith("data:"):
                                data_part = line_str[5:].strip()
                                if data_part == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_part)
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except Exception:
                                    continue
                        return
            except Exception:
                continue

        yield "LLM service was unable to stream response from OpenRouter at this time."


# Alias
LLMClient = OpenRouterLLMClient

