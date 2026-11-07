"""
DeepSeek Cost Tracking for LangSmith.

Provides:
1. DeepSeek model pricing constants (cache-hit / cache-miss / output).
2. A ChatOpenAI subclass (CostTrackingChatOpenAI) that injects cost data
   directly into AIMessage.usage_metadata BEFORE LangSmith's tracer reads it.
3. A helper (calculate_deepseek_cost) used by raw-API wrapper functions
   to report costs for calls that bypass LangChain.

Reference:
  https://docs.langchain.com/langsmith/cost-tracking#llm-calls-sending-costs-directly
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek model pricing (unit: RMB per token)
# ---------------------------------------------------------------------------
# Prices are stored as per-token values for direct multiplication.
# Original prices are in "RMB per 1 million tokens"; we divide by 1e6 here.

DEEPSEEK_MODEL_PRICES: Dict[str, Dict[str, float]] = {
    "deepseek-v4-pro": {
        "input_cache_hit": 1.0 / 1_000_000,      # ¥1  per 1M tokens
        "input_cache_miss": 12.0 / 1_000_000,     # ¥12 per 1M tokens
        "output": 24.0 / 1_000_000,               # ¥24 per 1M tokens
    },
    "deepseek-v3-2-251201": {
        "input_cache_hit": 0.4 / 1_000_000,       # ¥0.4 per 1M tokens
        "input_cache_miss": 2.0 / 1_000_000,      # ¥2   per 1M tokens
        "output": 3.0 / 1_000_000,                # ¥3   per 1M tokens
    },
}

# Fallback: if model name is not found, use deepseek-v3-2-251201 pricing
_DEFAULT_PRICE_KEY = "deepseek-v3-2-251201"

# ---------------------------------------------------------------------------
# Model → LangSmith provider mapping
# ---------------------------------------------------------------------------
DEEPSEEK_MODEL_PROVIDERS: Dict[str, str] = {
    "deepseek-v4-pro": "deepseek",
    "deepseek-v3-2-251201": "volcengine",
}

_DEFAULT_PROVIDER = "deepseek"


def _get_provider(model_name: str) -> str:
    """Return the LangSmith ls_provider string for *model_name*."""
    if model_name in DEEPSEEK_MODEL_PROVIDERS:
        return DEEPSEEK_MODEL_PROVIDERS[model_name]
    for key in DEEPSEEK_MODEL_PROVIDERS:
        if key in model_name:
            return DEEPSEEK_MODEL_PROVIDERS[key]
    return _DEFAULT_PROVIDER


def _get_prices(model_name: str) -> Dict[str, float]:
    """Return the pricing dict for *model_name*, falling back to default."""
    # Try exact match first
    if model_name in DEEPSEEK_MODEL_PRICES:
        return DEEPSEEK_MODEL_PRICES[model_name]
    # Try substring match (e.g. "deepseek-v4-pro" inside a longer endpoint id)
    for key in DEEPSEEK_MODEL_PRICES:
        if key in model_name:
            return DEEPSEEK_MODEL_PRICES[key]
    return DEEPSEEK_MODEL_PRICES[_DEFAULT_PRICE_KEY]


def calculate_deepseek_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_hit_tokens: int = 0,
) -> Dict[str, Any]:
    """Calculate cost and build a usage_metadata dict for LangSmith.

    Args:
        model_name: The DeepSeek model identifier.
        input_tokens: Total input (prompt) tokens.
        output_tokens: Total output (completion) tokens.
        cache_hit_tokens: Number of input tokens served from cache.

    Returns:
        A dict suitable for ``run_tree.set(usage_metadata=...)``.
        Contains input_tokens, output_tokens, input_cost, output_cost,
        input_token_details, and total_cost.
    """
    prices = _get_prices(model_name)
    cache_miss_tokens = max(input_tokens - cache_hit_tokens, 0)

    input_cost = (
        cache_hit_tokens * prices["input_cache_hit"]
        + cache_miss_tokens * prices["input_cache_miss"]
    )
    output_cost = output_tokens * prices["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_hit_tokens},
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


# ---------------------------------------------------------------------------
# ChatOpenAI subclass — injects cost into AIMessage.usage_metadata
# ---------------------------------------------------------------------------

class CostTrackingChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that injects DeepSeek cost data into usage_metadata.

    Overrides ``_generate()`` to add ``input_cost``, ``output_cost``, and
    ``total_cost`` fields to the ``AIMessage.usage_metadata`` dict BEFORE
    LangSmith's tracer reads it.  This guarantees that costs appear in the
    LangSmith UI regardless of callback ordering or run-tree context.

    Also enriches ``input_token_details`` with ``cache_read`` counts when
    DeepSeek returns ``prompt_cache_hit_tokens`` in the raw API response.

    Falls back gracefully: if anything goes wrong during cost injection,
    the original unmodified result is returned.
    """

    def _get_ls_params(self, stop=None, **kwargs):
        """Override to set ls_provider based on model name."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = _get_provider(self.model_name)
        return params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Call the parent _generate, then inject cost fields into the result."""
        result: ChatResult = super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        try:
            self._inject_cost(result)
        except Exception:
            logger.debug(
                "CostTrackingChatOpenAI: cost injection failed, "
                "returning original result",
                exc_info=True,
            )

        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async variant — same cost injection logic."""
        result: ChatResult = await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        try:
            self._inject_cost(result)
        except Exception:
            logger.debug(
                "CostTrackingChatOpenAI: async cost injection failed, "
                "returning original result",
                exc_info=True,
            )

        return result

    # ------------------------------------------------------------------
    def _inject_cost(self, result: ChatResult) -> None:
        """Mutate *result* in-place to add cost fields to usage_metadata."""
        # --- 1. Extract token counts from llm_output ---------------------
        token_usage = (result.llm_output or {}).get("token_usage", {})
        if not token_usage:
            return

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        if input_tokens == 0 and output_tokens == 0:
            return

        # DeepSeek-specific cache field (may be absent or zero)
        cache_hit = token_usage.get("prompt_cache_hit_tokens", 0)

        # --- 2. Calculate cost -------------------------------------------
        model_name = self.model_name  # from ChatOpenAI
        cost = calculate_deepseek_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit,
        )

        # --- 3. Inject into every generation's AIMessage.usage_metadata --
        for gen in result.generations:
            msg = gen.message
            if not isinstance(msg, AIMessage):
                continue
            um = msg.usage_metadata
            if um is None:
                continue
            # Add cost fields — these are standard LangSmith-recognized keys
            um["input_cost"] = cost["input_cost"]
            um["output_cost"] = cost["output_cost"]
            um["total_cost"] = cost["total_cost"]
            # Enrich input_token_details with cache_read if available
            if cache_hit > 0:
                details = um.get("input_token_details") or {}
                details["cache_read"] = cache_hit
                um["input_token_details"] = details
