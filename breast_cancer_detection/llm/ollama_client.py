from __future__ import annotations

import os
import threading
import time
from typing import Generator, List

import ollama


class OllamaLLMClient:
    PRIMARY_MODEL = "mistral:latest"
    FALLBACK_MODEL = "llama3:latest"
    _shared = None
    _shared_lock = threading.Lock()
    _last_verify = 0.0
    _verify_ttl = 120.0

    def __init__(self, verify: bool = True):
        self.is_available = False
        self.client = None
        self.host = None
        if verify:
            self._verify_models()

    @classmethod
    def get_shared(cls) -> "OllamaLLMClient":
        with cls._shared_lock:
            now = time.time()
            if cls._shared is None:
                cls._shared = cls(verify=True)
                cls._last_verify = now
            elif now - cls._last_verify > cls._verify_ttl:
                cls._shared._verify_models()
                cls._last_verify = now
            return cls._shared

    def _candidate_hosts(self):
        env_host = os.getenv("OLLAMA_HOST", "").strip()
        hosts = []
        if env_host:
            hosts.append(env_host)
        hosts.extend(
            [
                "http://127.0.0.1:11434",
                "http://localhost:11434",
                "http://0.0.0.0:11434",
            ]
        )
        # preserve order and uniqueness
        seen = set()
        ordered = []
        for h in hosts:
            if h and h not in seen:
                seen.add(h)
                ordered.append(h)
        return ordered

    def _verify_models(self):
        for host in self._candidate_hosts():
            try:
                client = ollama.Client(host=host)
                listing = client.list()
                models = listing.get("models", []) if isinstance(listing, dict) else []
                available = set()
                for m in models:
                    if isinstance(m, dict):
                        name = m.get("name") or m.get("model")
                    else:
                        name = str(m)
                    if name:
                        name_str = str(name).strip()
                        available.add(name_str)
                        available.add(name_str.split(":")[0].strip())
                for model in [self.PRIMARY_MODEL, self.FALLBACK_MODEL]:
                    model_base = model.split(":")[0]
                    if model not in available and model_base not in available:
                        # Don't pull at runtime; avoids latency spikes. Use installed models only.
                        continue
                self.client = client
                self.host = host
                self.is_available = True
                return
            except Exception:
                continue
        self.client = None
        self.host = None
        self.is_available = False

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        if not self.is_available:
            return (
                "Ollama is currently unreachable from this process. "
                "Please verify the Ollama host/port and service visibility for this runtime."
            )
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
        for model in [self.PRIMARY_MODEL, self.FALLBACK_MODEL]:
            try:
                response = self.client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
                )
                return response["message"]["content"]
            except Exception as exc:
                if model == self.FALLBACK_MODEL:
                    return (
                        f"LLM unavailable right now ({exc}). "
                        "I can still help with a static explanation from the current report."
                    )
        raise RuntimeError("Unexpected LLM generation failure.")

    def stream(self, prompt: str, system: str = "", history: List[dict] | None = None) -> Generator[str, None, None]:
        if not self.is_available:
            yield (
                "I couldn't reach Ollama from this app process. "
                "Please ensure Ollama is running and reachable, then retry."
            )
            return
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})
        for model in [self.PRIMARY_MODEL, self.FALLBACK_MODEL]:
            try:
                stream = self.client.chat(
                    model=model, messages=messages, stream=True, options={"num_ctx": num_ctx, "num_predict": num_predict}
                )
                for chunk in stream:
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    if content:
                        yield content
                return
            except Exception:
                continue
        yield "Both primary and fallback Ollama models failed for this request."

