"""
Configuration Management for TRNSYS DCK Generator.

Uses pydantic-settings for type-safe configuration with environment variable support.
Configuration hierarchy:
1. Environment variables (highest priority)
2. .env file
3. Default values
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Environment variables are automatically mapped:
    - DEEPSEEK_API_KEY -> deepseek_api_key
    - DEEPSEEK_BASE_URL -> deepseek_base_url
    etc.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # === DeepSeek API Source Selection ===
    deepseek_api_source: Literal["official", "volcano"] = Field(
        default="official",
        description="DeepSeek API source: 'official' (DeepSeek Official) or 'volcano' (Volcano Engine)",
    )

    # === DeepSeek Official API Configuration ===
    deepseek_api_key_official: str = Field(
        default="",
        description="DeepSeek Official API key",
    )
    deepseek_base_url_official: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek Official API base URL",
    )
    deepseek_model_official: str = Field(
        default="deepseek-v4-pro",
        description="DeepSeek Official model name (e.g., deepseek-v4-pro)",
    )

    # === DeepSeek Volcano Engine API Configuration ===
    deepseek_api_key_volcano: str = Field(
        default="",
        description="DeepSeek Volcano Engine API key",
    )
    deepseek_base_url_volcano: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="DeepSeek Volcano Engine API base URL",
    )
    deepseek_model_volcano: str = Field(
        default="deepseek-v3-2-251201",
        description="DeepSeek Volcano Engine model name",
    )

    # === Legacy Configuration (Deprecated - kept for backward compatibility) ===
    deepseek_api_key: str = Field(
        default="",
        description="[DEPRECATED] DeepSeek API key (mapped to volcano if new fields not set)",
    )
    deepseek_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="[DEPRECATED] DeepSeek API base URL (mapped to volcano if new fields not set)",
    )
    deepseek_model: str = Field(
        default="deepseek-v3-2-251201",
        description="[DEPRECATED] DeepSeek model name (mapped to volcano if new fields not set)",
    )

    # === Alternative LLM (OpenAI) ===
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (alternative provider)",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model name",
    )

    # === LLM Provider Selection ===
    llm_provider: Literal["deepseek", "openai"] = Field(
        default="deepseek",
        description="Which LLM provider to use",
    )

    # === LLM Parameters ===
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens for LLM response",
    )
    llm_timeout: int = Field(
        default=120,
        ge=10,
        description="Timeout for LLM calls in seconds",
    )

    # === Embedding Configuration ===
    embedding_provider: Literal["openai", "dashscope", "local"] = Field(
        default="dashscope",
        description="Embedding provider for RAG (dashscope recommended for Chinese)",
    )
    embedding_model: str = Field(
        default="text-embedding-v4",
        description="Embedding model name (Qwen text-embedding-v4 for DashScope)",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding vector dimension (1024 for Qwen text-embedding-v4)",
    )

    # === DashScope Configuration (for RAG Embedding) ===
    dashscope_api_key: str = Field(
        default="",
        description="DashScope API key for Qwen embedding (Alibaba Cloud)",
    )

    # === Project Paths ===
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Project root directory (trnsys_dck_generator)",
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data",
        description="Data directory (trnsys_dck_generator/data)",
    )

    # === TRNSYS Paths ===
    trnsys_exe_path: Path = Field(
        default=Path(r"D:\TRNSYS18_04\Exe\TrnEXE64.exe"),
        description="Path to TrnEXE64.exe",
    )
    trnsys_weather_dir: Path = Field(
        default=Path(r"D:\TRNSYS18_04\Weather"),
        description="Path to TRNSYS weather data directory",
    )

    # === Original Data Paths ===
    original_data_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent / "original_trnsys_data",
        description="Root directory for original TRNSYS data",
    )

    # === Generation Settings ===
    max_validation_iterations: int = Field(
        default=5,
        ge=1,
        description="Maximum validation/retry iterations",
    )
    enable_memory: bool = Field(
        default=True,
        description="Enable Mem0 memory system",
    )
    enable_rag: bool = Field(
        default=True,
        description="Enable RAG retrieval",
    )

    # === Logging ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_to_file: bool = Field(
        default=True,
        description="Enable file logging",
    )

    # === LangSmith (Optional Tracing) ===
    langsmith_api_key: str = Field(
        default="",
        description="LangSmith API key for tracing",
    )
    langsmith_project: str = Field(
        default="trnsys-dck-generator",
        description="LangSmith project name",
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langsmith_endpoint: str = Field(
        default="",
        description="LangSmith API endpoint (leave empty for SaaS, use http://localhost:1984 for self-hosted)",
    )

    @field_validator("project_root", "data_dir", "original_data_root", mode="before")
    @classmethod
    def validate_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def train_dck_dir(self) -> Path:
        """Path to training DCK files."""
        return self.original_data_root / "206_dck_files" / "train_set"

    @property
    def test_dck_dir(self) -> Path:
        """Path to test DCK files."""
        return self.original_data_root / "206_dck_files" / "test_set"

    @property
    def train_dck_parsed_path(self) -> Path:
        """Path to parsed training DCK JSON."""
        return self.original_data_root / "parsed_206_dck_files" / "train_set_dck_parsed.json"

    @property
    def test_dck_parsed_path(self) -> Path:
        """Path to parsed test DCK JSON."""
        return self.original_data_root / "parsed_206_dck_files" / "test_set_dck_parsed.json"

    @property
    def tmf_parsed_path(self) -> Path:
        """Path to parsed TMF JSON."""
        return self.original_data_root / "parsed_765_tmf_files.json"

    @property
    def equation_doc_path(self) -> Path:
        """Path to Equation usage documentation."""
        return self.original_data_root / "TRNSYS_Equation_User_Guide.doc"

    @property
    def task_descriptions_dir(self) -> Path:
        """Path to task descriptions directory."""
        return self.data_dir / "test_task_descriptions"

    @property
    def vector_store_dir(self) -> Path:
        """Path to vector store directory."""
        return self.data_dir / "vector_store"

    @property
    def checkpoints_dir(self) -> Path:
        """Path to LangGraph checkpoints directory."""
        return self.data_dir / "checkpoints"

    @property
    def results_dir(self) -> Path:
        """Path to results directory."""
        return self.data_dir / "results"

    def get_llm_config(self) -> dict:
        """Get LLM configuration based on selected provider.

        Note: max_tokens and timeout are read from performance_config for centralized management.
        """
        # Import performance_config for centralized LLM settings
        from .performance_config import get_performance_config
        perf_config = get_performance_config()

        if self.llm_provider == "deepseek":
            # Select DeepSeek API source (official or volcano)
            if self.deepseek_api_source == "official":
                # Use DeepSeek Official API (with backward compatibility fallback)
                api_key = self.deepseek_api_key_official or self.deepseek_api_key
                base_url = self.deepseek_base_url_official or self.deepseek_base_url
                model = self.deepseek_model_official or self.deepseek_model
            else:  # volcano
                # Use Volcano Engine API (with backward compatibility fallback)
                api_key = self.deepseek_api_key_volcano or self.deepseek_api_key
                base_url = self.deepseek_base_url_volcano or self.deepseek_base_url
                model = self.deepseek_model_volcano or self.deepseek_model

            return {
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
                "temperature": self.llm_temperature,
                "max_tokens": perf_config.llm_max_tokens,
                "timeout": perf_config.llm_timeout,
            }
        else:  # openai
            return {
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "model": self.openai_model,
                "temperature": self.llm_temperature,
                "max_tokens": perf_config.llm_max_tokens,
                "timeout": perf_config.llm_timeout,
            }

    def get_volcano_llm_config(self) -> dict:
        """Get Volcano Engine LLM configuration (for validator's web_search).

        This method always returns Volcano Engine configuration regardless of
        deepseek_api_source setting, because only Volcano Engine supports web_search tool.

        Note: max_tokens and timeout are read from performance_config for centralized management.
        """
        from .performance_config import get_performance_config
        perf_config = get_performance_config()

        # Always use Volcano Engine configuration (with backward compatibility fallback)
        api_key = self.deepseek_api_key_volcano or self.deepseek_api_key
        base_url = self.deepseek_base_url_volcano or self.deepseek_base_url
        model = self.deepseek_model_volcano or self.deepseek_model

        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": self.llm_temperature,
            "max_tokens": perf_config.llm_max_tokens,
            "timeout": perf_config.llm_timeout,
        }

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.data_dir / "train" / "dck_files",
            self.data_dir / "train" / "dck_parsed",
            self.data_dir / "test" / "dck_files",
            self.data_dir / "test" / "dck_parsed",
            self.data_dir / "test" / "task_descriptions",
            self.data_dir / "tmf" / "tmf_parsed",
            self.vector_store_dir / "mem0",
            self.vector_store_dir / "rag",
            self.checkpoints_dir,
            self.results_dir / "proposed",
            self.results_dir / "b1_direct_llm",
            self.results_dir / "b2_rag",
            self.results_dir / "b3_multiagent_rag",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload.
    """
    return Settings()


def setup_environment() -> None:
    """
    Set up environment variables for LangSmith and other services.

    Call this at application startup.
    """
    settings = get_settings()

    # LangSmith tracing
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_TRACING"] = "true"

        # Support self-hosted endpoint
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

        # Verify LangSmith connection and log cost tracking status
        _verify_langsmith_cost_tracking(settings)

    # Ensure directories exist
    settings.ensure_directories()


def _verify_langsmith_cost_tracking(settings) -> None:
    """Verify LangSmith connection and log cost tracking configuration.

    Attempts to connect to LangSmith using the SDK client and logs the
    status of cost tracking setup.  Also logs the DeepSeek model pricing
    that will be used for client-side cost calculation.

    This is informational only — failure does not block application startup.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        from langsmith import Client
        from .langsmith_cost import DEEPSEEK_MODEL_PRICES

        # Verify connection by instantiating the client
        Client()

        # Log connection info
        logger.info(
            "LangSmith cost tracking enabled — project: %s",
            settings.langsmith_project,
        )

        # Log the DeepSeek pricing that will be applied
        for model, prices in DEEPSEEK_MODEL_PRICES.items():
            logger.info(
                "  Model %-30s | input(cache_miss)=¥%.1f/1M  "
                "input(cache_hit)=¥%.1f/1M  output=¥%.1f/1M",
                model,
                prices["input_cache_miss"] * 1_000_000,
                prices["input_cache_hit"] * 1_000_000,
                prices["output"] * 1_000_000,
            )

        # Suggest UI pricing table configuration for redundancy
        logger.info(
            "  Tip: For redundancy, you can also add these prices in the "
            "LangSmith UI at Settings > Models "
            "(https://smith.langchain.com/settings/workspaces/models). "
            "Client-side costs (sent via usage_metadata) take precedence."
        )

    except Exception as e:
        logger.warning(
            "LangSmith cost tracking verification failed (non-fatal): %s", e
        )
