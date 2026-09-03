from __future__ import annotations

from typing import Generator, List

from breast_cancer_detection.llm.openrouter_client import OpenRouterLLMClient
from breast_cancer_detection.llm.prompts import CHAT_SYSTEM_PROMPT


def stream_chat_response(
    user_message: str,
    report_context: str,
    chat_history: List[dict],
    model: str | None = None,
    web_search: bool = False,
) -> Generator[str, None, None]:
    try:
        client = OpenRouterLLMClient.get_shared()
        context_str = (report_context or "No specific patient scan loaded. Answer general medical and research inquiries.").strip()[:8000]
        system = CHAT_SYSTEM_PROMPT.format(report_context=context_str)
        return client.stream(
            prompt=user_message,
            system=system,
            history=chat_history,
            model=model,
            web_search=web_search,
        )
    except Exception:
        def fallback_stream():
            yield (
                "Chat service is temporarily unavailable. "
                "You can still review the report in the Results page and retry shortly."
            )
        return fallback_stream()

