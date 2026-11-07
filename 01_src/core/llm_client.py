"""
LLM Client Factory for TRNSYS DCK Generator.

Provides a unified interface for creating LLM clients that work with
DeepSeek API (OpenAI compatible) or OpenAI directly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings
from .langsmith_cost import CostTrackingChatOpenAI, calculate_deepseek_cost, _get_provider

import functools


def _ls_traceable_llm(func):
    """Decorator: wrap a raw-API LLM call so it appears as an LLM run in LangSmith.

    Uses ``@traceable(run_type="llm")`` under the hood.  The wrapper also
    ensures ``get_current_run_tree()`` works inside the wrapped function,
    because ``@traceable`` sets the ``_PARENT_RUN_TREE`` contextvar.

    When LangSmith tracing is disabled, ``@traceable`` is a no-op and
    the function runs unchanged.
    """
    try:
        from langsmith import traceable as _traceable
        wrapped = _traceable(run_type="llm", name=func.__name__)(func)
        functools.update_wrapper(wrapped, func)
        return wrapped
    except ImportError:
        return func


def create_llm_client(
    settings: Optional[Settings] = None,
    model_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
) -> ChatOpenAI:
    """
    Create an LLM client based on configuration.

    DeepSeek API is OpenAI-compatible, so we use ChatOpenAI for both providers.

    Args:
        settings: Settings instance (uses default if not provided)
        model_override: Override the configured model
        temperature_override: Override the configured temperature

    Returns:
        ChatOpenAI instance configured for the selected provider
    """
    if settings is None:
        settings = get_settings()

    config = settings.get_llm_config()

    # DeepSeek Official v4-pro requires special handling:
    # 1. Disable thinking mode (v4-pro enables it by default)
    # 2. Pass max_tokens via extra_body because DeepSeek v4-pro does not recognize
    #    max_completion_tokens (which LangChain auto-converts max_tokens into),
    #    causing output to be silently capped at 8K
    extra_body = None
    max_tokens_param = config["max_tokens"]
    if (settings.llm_provider == "deepseek"
            and settings.deepseek_api_source == "official"):
        extra_body = {"thinking": {"type": "disabled"}, "max_tokens": config["max_tokens"]}
        max_tokens_param = None

    actual_model = model_override or config["model"]

    return CostTrackingChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=actual_model,
        temperature=temperature_override if temperature_override is not None else config["temperature"],
        max_tokens=max_tokens_param,
        timeout=config["timeout"],
        max_retries=5,  # 自动重试5次
        default_headers={"X-Request-Source": "trnsys-dck-generator"},
        extra_body=extra_body,
    )


@lru_cache()
def get_chat_model(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> ChatOpenAI:
    """
    Get a cached chat model instance.

    Args:
        model_name: Optional model name override
        temperature: Temperature for generation

    Returns:
        Cached ChatOpenAI instance
    """
    settings = get_settings()

    return create_llm_client(
        settings=settings,
        model_override=model_name,
        temperature_override=temperature,
    )


def create_agent_llm(
    agent_role: str,
    settings: Optional[Settings] = None,
) -> ChatOpenAI:
    """
    Create an LLM client configured for a specific agent role.

    Different agents may benefit from different temperature settings:
    - Planner: Lower temperature for consistent planning
    - Validator: Lower temperature for accurate validation with web search enabled
    - Component Selector: Medium temperature for balanced selection
    - Equation Generator: Lower temperature for accurate equations

    Different agents may also have different timeout requirements:
    - Heavy agents (connection_generator, parameter_configurator): 360s
    - Other agents: 240s (default_agent_timeout)

    Args:
        agent_role: The role of the agent (from AgentRole enum)
        settings: Settings instance

    Returns:
        ChatOpenAI instance configured for the agent
    """
    if settings is None:
        settings = get_settings()

    # Agent-specific temperature settings
    temperature_map = {
        "supervisor": 0.5,
        "planner": 0.3,
        "component_selector": 0.5,
        "parameter_configurator": 0.15,
        "connection_generator": 0.15,
        "equation_generator": 0.2,
        "layout_designer": 0.5,
        "validator": 0.2,
    }

    # Agent-specific max_tokens settings
    # NOTE: DeepSeek v4-pro supports up to 128K output tokens
    # Most agents use 32768, but parameter_configurator and connection_generator
    # need 65536 (2x) due to large JSON output requirements
    max_tokens_map = {
        "supervisor": 32768,
        "planner": 32768,
        "component_selector": 32768,
        "parameter_configurator": 65536,  # 2x for large parameter JSON
        "connection_generator": 65536,    # 2x for large connection JSON
        "equation_generator": 32768,
        "layout_designer": 32768,
        "validator": 32768,
    }

    temperature = temperature_map.get(agent_role, settings.llm_temperature)
    max_tokens = max_tokens_map.get(agent_role, None)

    # Get agent-specific timeout from performance_config
    from .performance_config import get_performance_config
    perf_config = get_performance_config()

    # Priority: agent_timeout_map > default_agent_timeout > llm_timeout (fallback)
    agent_timeout = perf_config.agent_timeout_map.get(
        agent_role,
        perf_config.default_agent_timeout
    )

    # If max_tokens is specified, create a custom LLM instance
    if max_tokens:
        config = settings.get_llm_config()

        # DeepSeek Official v4-pro requires special handling:
        # 1. Disable thinking mode (v4-pro enables it by default)
        # 2. Pass max_tokens via extra_body because DeepSeek v4-pro does not recognize
        #    max_completion_tokens (which LangChain auto-converts max_tokens into),
        #    causing output to be silently capped at 8K
        extra_body = None
        max_tokens_param = max_tokens
        if (settings.llm_provider == "deepseek"
                and settings.deepseek_api_source == "official"):
            extra_body = {"thinking": {"type": "disabled"}, "max_tokens": max_tokens}
            max_tokens_param = None

        return CostTrackingChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens_param,
            timeout=agent_timeout,  # Use agent-specific timeout
            max_retries=5,  # 自动重试5次
            default_headers={"X-Request-Source": "trnsys-dck-generator"},
            extra_body=extra_body,
        )

    return create_llm_client(
        settings=settings,
        temperature_override=temperature,
    )


def create_embedding_client(
    settings: Optional[Settings] = None,
):
    """
    Create an embedding client for RAG.

    Args:
        settings: Settings instance

    Returns:
        Embedding model instance
    """
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        # For DeepSeek, we still use OpenAI embeddings
        # as DeepSeek doesn't provide embedding API
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key or settings.deepseek_api_key,
            model=settings.embedding_model,
        )
    else:
        # Local embeddings (e.g., sentence-transformers)
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )


# === Utility Functions ===

def test_llm_connection(settings: Optional[Settings] = None) -> bool:
    """
    Test if the LLM connection is working.

    Args:
        settings: Settings instance

    Returns:
        True if connection is successful
    """
    try:
        llm = create_llm_client(settings)
        response = llm.invoke("Say 'Hello' if you can hear me.")
        return "hello" in response.content.lower()
    except Exception as e:
        print(f"LLM connection test failed: {e}")
        return False


def get_model_info(settings: Optional[Settings] = None) -> dict:
    """
    Get information about the configured model.

    Returns:
        Dictionary with model information
    """
    if settings is None:
        settings = get_settings()

    config = settings.get_llm_config()

    return {
        "provider": settings.llm_provider,
        "model": config["model"],
        "base_url": config["base_url"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }


@_ls_traceable_llm
def call_validator_with_search(
    messages: List[Dict[str, str]],
    settings: Optional[Settings] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Volcano Engine /responses endpoint with web search enabled for validator.

    Uses the /responses endpoint instead of /chat/completions to support web_search tool.

    IMPORTANT: This function ALWAYS uses Volcano Engine API regardless of
    deepseek_api_source setting, because only Volcano Engine supports web_search tool.

    Args:
        messages: List of message dicts with 'role' and 'content'
        settings: Settings instance
        temperature: Temperature for LLM generation (if None, uses config default)
        max_output_tokens: Maximum output tokens (if None, uses config max_tokens)
        instructions: System-level instructions (equivalent to system prompt)

    Returns:
        Response dict from Volcano Engine API
    """
    if settings is None:
        settings = get_settings()

    # ALWAYS use Volcano Engine configuration for web_search
    config = settings.get_volcano_llm_config()

    # Use /responses endpoint for web search support
    url = f"{config['base_url']}/responses"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config["model"],
        "input": messages,
        "tools": [{"type": "web_search", "max_keyword": 5}],
        "temperature": temperature if temperature is not None else config["temperature"],
        "max_output_tokens": max_output_tokens if max_output_tokens is not None else config["max_tokens"],
    }

    # Add system-level instructions if provided
    if instructions:
        payload["instructions"] = instructions

    response = requests.post(url, headers=headers, json=payload, timeout=config["timeout"])
    response.raise_for_status()
    result = response.json()

    # --- LangSmith cost tracking for raw API call --------------------------
    _report_volcano_cost_to_langsmith(result, config["model"])

    return result


@_ls_traceable_llm
def call_validator_without_search(
    messages: List[Dict[str, str]],
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Call Volcano Engine /responses endpoint WITHOUT web search for validator.

    Uses the /responses endpoint but without tools parameter.

    IMPORTANT: This function ALWAYS uses Volcano Engine API regardless of
    deepseek_api_source setting, to maintain consistency with call_validator_with_search.

    Args:
        messages: List of message dicts with 'role' and 'content'
        settings: Settings instance

    Returns:
        Response dict from Volcano Engine API
    """
    if settings is None:
        settings = get_settings()

    # ALWAYS use Volcano Engine configuration for consistency
    config = settings.get_volcano_llm_config()

    url = f"{config['base_url']}/responses"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config["model"],
        "input": messages,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=config["timeout"])
    response.raise_for_status()
    result = response.json()

    # --- LangSmith cost tracking for raw API call --------------------------
    _report_volcano_cost_to_langsmith(result, config["model"])

    return result


@_ls_traceable_llm
def call_validator_fusion(
    analysis_with_search: Dict[str, Any],
    analysis_without_search: Dict[str, Any],
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Fuse two validator analyses (with and without search) using LLM.

    IMPORTANT: This function ALWAYS uses Volcano Engine API regardless of
    deepseek_api_source setting, to maintain consistency with validator analysis calls.

    Args:
        analysis_with_search: Analysis result from search-enabled LLM
        analysis_without_search: Analysis result from search-disabled LLM
        settings: Settings instance

    Returns:
        Fused analysis result in the same JSON format
    """
    if settings is None:
        settings = get_settings()

    # ALWAYS use Volcano Engine configuration for consistency
    config = settings.get_volcano_llm_config()

    fusion_prompt = f"""You are an error analysis fusion expert. Your task is to fuse two independent TRNSYS validation error analyses.

【Input Information】
You will receive two independent error analysis results (JSON format):
1. Analysis without web search (based on local knowledge)
2. Analysis with web search (may contain latest information)

Analysis without search:
{json.dumps(analysis_without_search, ensure_ascii=False)}

Analysis with search:
{json.dumps(analysis_with_search, ensure_ascii=False)}

【Fusion Principles】
1. Deduplication and merging: For the same type of error, merge the insights from both analyses and avoid repetition.
2. Priority judgment:
   - If both analyses agree, keep that analysis
   - If search results provide more recent or accurate information, prioritize using it
   - If search results contradict local knowledge, perform reasonableness check and choose the more reliable one
3. Clear and concise: Avoid verbosity, maintain accuracy and actionability of analysis

【Output Format Requirement】
You must strictly follow the original JSON format without any deviation. Output ONLY valid JSON with no other text.

【Important Notes】
- Output ONLY JSON, no other text
- Ensure JSON format is completely valid
- Do not modify field names or structure
- All fields are required"""

    url = f"{config['base_url']}/responses"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config["model"],
        "input": [{"role": "user", "content": fusion_prompt}],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=config["timeout"])
    response.raise_for_status()
    result = response.json()

    # --- LangSmith cost tracking for raw API call --------------------------
    _report_volcano_cost_to_langsmith(result, config["model"])

    return result


def _report_volcano_cost_to_langsmith(
    api_response: Dict[str, Any],
    model_name: str,
) -> None:
    """Report token usage and cost to LangSmith for a raw Volcano Engine API call.

    The Volcano Engine /responses API returns a ``usage`` dict at the top level:
        {"usage": {"prompt_tokens": ..., "completion_tokens": ...,
                   "total_tokens": ..., "prompt_cache_hit_tokens": ...}}

    This function extracts those counts, calculates cost, and patches the
    current LangSmith run tree (if active).  Does nothing when LangSmith
    tracing is disabled.
    """
    try:
        usage = api_response.get("usage", {})
        if not usage:
            return

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cache_hit = usage.get("prompt_cache_hit_tokens", 0)

        if input_tokens == 0 and output_tokens == 0:
            return

        usage_meta = calculate_deepseek_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit,
        )

        from langsmith import get_current_run_tree
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.set(usage_metadata=usage_meta)
            run_tree.set(metadata={
                "ls_provider": _get_provider(model_name),
                "ls_model_name": model_name,
            })

    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "_report_volcano_cost_to_langsmith: failed",
            exc_info=True,
        )
