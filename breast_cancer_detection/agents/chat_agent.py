from __future__ import annotations

from typing import Generator, List

from breast_cancer_detection.llm.ollama_client import OllamaLLMClient
from breast_cancer_detection.llm.prompts import CHAT_SYSTEM_PROMPT


def stream_chat_response(user_message: str, report_context: str, chat_history: List[dict]) -> Generator[str, None, None]:
    try:
        client = OllamaLLMClient.get_shared()
        system = CHAT_SYSTEM_PROMPT.format(report_context=report_context[:8000])
        return client.stream(prompt=user_message, system=system, history=chat_history)
    except Exception:
        def fallback_stream():
            yield (
                "Chat service is temporarily unavailable. "
                "You can still review the report in the Results page and retry shortly."
            )
        return fallback_stream()

