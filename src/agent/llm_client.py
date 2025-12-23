"""
Unified LLM client for the disruption alert agent, with a strict
cost-aware backend priority so the project runs at zero cost by default:

  1. Claude API (Anthropic)  — only used if ANTHROPIC_API_KEY is set in
     .env. This is a paid API; it is entirely opt-in.
  2. Ollama (local, free)    — used automatically if a local Ollama server
     is reachable at OLLAMA_BASE_URL. Install from https://ollama.com and
     `ollama pull llama3.1` to enable this tier at zero cost.
  3. Rule-based fallback (free, always available) — a deterministic
     template-based narrative generator. Guarantees the agent always
     produces a complete, well-formed alert even with no LLM configured.

All three tiers implement the same `generate(system_prompt, user_prompt)`
interface. Every external call is wrapped in error handling; a failure at
one tier automatically falls through to the next.
"""
from __future__ import annotations

import time
import requests

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

OLLAMA_MODEL = "llama3.1"
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3


def _ollama_reachable() -> bool:
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


class LLMClient:
    def __init__(self):
        self.backend = self._select_backend()
        logger.info("LLMClient initialized with backend='%s'", self.backend)

    def _select_backend(self) -> str:
        if settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_api_key_here":
            return "claude"
        if _ollama_reachable():
            return "ollama"
        return "rule_based"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.backend == "claude":
            result = self._generate_claude(system_prompt, user_prompt)
            if result is not None:
                return result
            logger.warning("Claude call failed; falling back to next available backend")
            self.backend = "ollama" if _ollama_reachable() else "rule_based"

        if self.backend == "ollama":
            result = self._generate_ollama(system_prompt, user_prompt)
            if result is not None:
                return result
            logger.warning("Ollama call failed; falling back to rule-based backend")
            self.backend = "rule_based"

        return self._generate_rule_based(system_prompt, user_prompt)

    def _generate_claude(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)
        except Exception:
            logger.exception("Could not initialize the Claude API client")
            return None
        for attempt in range(MAX_ATTEMPTS):
            try:
                message = client.messages.create(
                    model=settings.claude_model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                response = "".join(block.text for block in message.content if hasattr(block, "text"))
                if response:
                    return response
                raise ValueError("Claude returned an empty response")
            except Exception as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    logger.exception("Claude API call failed after %d attempts", MAX_ATTEMPTS)
                    return None
                logger.warning("Claude API attempt %d/%d failed: %s", attempt + 1, MAX_ATTEMPTS, exc)
                time.sleep(2 ** attempt)
        return None

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str | None:
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = requests.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                response = resp.json().get("response", "").strip()
                if response:
                    return response
                raise ValueError("Ollama returned an empty response")
            except (requests.exceptions.RequestException, ValueError) as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    logger.exception("Ollama API call failed after %d attempts", MAX_ATTEMPTS)
                    return None
                logger.warning("Ollama API attempt %d/%d failed: %s", attempt + 1, MAX_ATTEMPTS, exc)
                time.sleep(2 ** attempt)
        return None

    def _generate_rule_based(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic, template-based narrative — no external calls, always succeeds."""
        logger.info("Generating narrative with rule-based fallback (no LLM configured)")
        return (
            "[Rule-based summary — no LLM backend configured] "
            "This alert was generated from quantitative model outputs (GNN criticality/impact "
            "scores, causal effect estimate, and graph cascade simulation) without LLM-generated "
            "prose. Configure ANTHROPIC_API_KEY or run a local Ollama server for richer natural "
            "language reasoning."
        )
