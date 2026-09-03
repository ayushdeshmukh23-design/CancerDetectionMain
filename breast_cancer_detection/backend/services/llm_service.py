from __future__ import annotations

import json

from breast_cancer_detection.llm.openrouter_client import OpenRouterLLMClient
from breast_cancer_detection.llm.prompts import EXPLANATION_SYSTEM_PROMPT, EXPLANATION_USER_PROMPT_TEMPLATE


class LLMService:
    """LLM wrapper for deterministic patient-friendly explanations."""

    def __init__(self):
        self.client = OpenRouterLLMClient.get_shared()

    def explain(self, payload: dict) -> str:
        prompt = EXPLANATION_USER_PROMPT_TEMPLATE.format(payload_json=json.dumps(payload, indent=2))
        return self.client.generate(prompt=prompt, system=EXPLANATION_SYSTEM_PROMPT, temperature=0.2)
