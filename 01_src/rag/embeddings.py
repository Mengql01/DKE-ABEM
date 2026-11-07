"""
Embedding Configuration for TRNSYS DCK Generator RAG.

Uses LlamaIndex official DashScopeEmbedding API.
NO hardcoded bypass classes (removed QwenEmbedding from old project).

This module provides embedding configuration for the RAG system used by:
- B2: RAG method (single LLM + RAG)
- B3: MultiAgent_RAG method (multi-agent + RAG)

Note: Proposed: MultiAgent_Memory method does NOT use RAG.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding

if TYPE_CHECKING:
    from ..core.config import Settings as AppSettings

logger = logging.getLogger(__name__)


class RAGEmbeddingConfig:
    """
    RAG embedding configuration manager using the official LlamaIndex API.

    Supports multiple embedding providers:
    - dashscope: Qwen text-embedding-v4 (recommended for Chinese)
    - openai: OpenAI text-embedding-3-small
    - local: Local HuggingFace models (future support)

    NO hardcoded bypass classes - uses official LlamaIndex integrations only.
    """

    def __init__(self, settings: Optional[AppSettings] = None):
        """
        Initialize RAG Embedding configuration.

        Args:
            settings: Application settings. If None, loads from get_settings().
        """
        if settings is None:
            from ..core.config import get_settings
            settings = get_settings()

        self.settings = settings
        self._embed_model: Optional[BaseEmbedding] = None
        self._initialized = False

    def get_embed_model(self) -> BaseEmbedding:
        """
        Get the configured embedding model.

        Uses LlamaIndex official API based on configured provider:
        - dashscope: DashScopeEmbedding (official LlamaIndex integration)
        - openai: OpenAIEmbedding (official LlamaIndex integration)

        Returns:
            Configured embedding model instance.

        Raises:
            ValueError: If provider is not supported or API key is missing.
        """
        if self._embed_model is not None:
            return self._embed_model

        provider = self.settings.embedding_provider

        if provider == "dashscope":
            self._embed_model = self._create_dashscope_embedding()
        elif provider == "openai":
            self._embed_model = self._create_openai_embedding()
        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported providers: dashscope, openai"
            )

        logger.info(
            f"Initialized {provider} embedding model: {self.settings.embedding_model}"
        )
        return self._embed_model

    def _create_dashscope_embedding(self) -> BaseEmbedding:
        """
        Create DashScope embedding using official LlamaIndex integration.

        Uses llama-index-embeddings-dashscope package.
        Model: Qwen text-embedding-v4 (1024 dimensions)

        Returns:
            DashScopeEmbedding instance.

        Raises:
            ValueError: If DASHSCOPE_API_KEY is not configured.
            ImportError: If llama-index-embeddings-dashscope is not installed.
        """
        if not self.settings.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required for DashScope embedding. "
                "Please set it in .env file or environment variable."
            )

        try:
            from llama_index.embeddings.dashscope import DashScopeEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-dashscope is required for DashScope embedding. "
                "Install with: pip install llama-index-embeddings-dashscope"
            ) from e

        # DashScope API limits batch size to 10, but llama-index default is 25
        return DashScopeEmbedding(
            model_name=self.settings.embedding_model,
            api_key=self.settings.dashscope_api_key,
            embed_batch_size=10,
        )

    def _create_openai_embedding(self) -> BaseEmbedding:
        """
        Create OpenAI embedding using official LlamaIndex integration.

        Uses llama-index-embeddings-openai package.
        Can also be used with OpenAI-compatible APIs via api_base.

        Returns:
            OpenAIEmbedding instance.

        Raises:
            ValueError: If OPENAI_API_KEY is not configured.
            ImportError: If llama-index-embeddings-openai is not installed.
        """
        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI embedding. "
                "Please set it in .env file or environment variable."
            )

        # Check if using OpenAI-compatible API (e.g., DashScope)
        is_compatible_api = (
            self.settings.openai_base_url and
            "openai.com" not in self.settings.openai_base_url
        )

        if is_compatible_api:
            # For OpenAI-compatible APIs like DashScope, use OpenAI client directly
            logger.info(
                f"Using OpenAI-compatible API at {self.settings.openai_base_url} "
                f"with model {self.settings.embedding_model}"
            )
            return self._create_compatible_embedding()

        # Standard OpenAI embedding
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-openai is required for OpenAI embedding. "
                "Install with: pip install llama-index-embeddings-openai"
            ) from e

        return OpenAIEmbedding(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key,
            api_base=self.settings.openai_base_url,
            dimensions=self.settings.embedding_dimension,
        )

    def _create_compatible_embedding(self) -> BaseEmbedding:
        """
        Create embedding for OpenAI-compatible APIs (e.g., DashScope).

        Uses OpenAI client directly to avoid model name validation issues.

        Returns:
            OpenAICompatibleEmbedding instance.
        """
        from typing import List

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for compatible embedding. "
                "Install with: pip install openai"
            ) from e

        # Create a custom embedding class for compatible APIs
        class OpenAICompatibleEmbedding(BaseEmbedding):
            """OpenAI-compatible embedding for DashScope and similar APIs."""

            def __init__(
                self,
                api_key: str,
                api_base: str,
                model: str,
                dimensions: int = 1024,
            ):
                super().__init__()
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                )
                self._model = model
                self._dimensions = dimensions

            @classmethod
            def class_name(cls) -> str:
                return "OpenAICompatibleEmbedding"

            def _get_query_embedding(self, query: str) -> List[float]:
                """Get embedding for a query."""
                response = self._client.embeddings.create(
                    model=self._model,
                    input=query,
                    dimensions=self._dimensions,
                )
                return response.data[0].embedding

            def _get_text_embedding(self, text: str) -> List[float]:
                """Get embedding for a text."""
                return self._get_query_embedding(text)

            async def _aget_query_embedding(self, query: str) -> List[float]:
                """Async get embedding for a query."""
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> List[float]:
                """Async get embedding for a text."""
                return self._get_query_embedding(text)

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                """Get embeddings for multiple texts."""
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                )
                return [item.embedding for item in response.data]

        return OpenAICompatibleEmbedding(
            api_key=self.settings.openai_api_key,
            api_base=self.settings.openai_base_url,
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimension,
        )

    def configure_global_settings(self) -> None:
        """
        Configure the global LlamaIndex Settings.

        Sets the embed_model in LlamaIndex's global Settings object.
        This affects all LlamaIndex operations that use embeddings.
        """
        embed_model = self.get_embed_model()
        Settings.embed_model = embed_model
        self._initialized = True
        logger.info("Configured LlamaIndex global embedding settings")

    @property
    def is_initialized(self) -> bool:
        """Check if global settings have been configured."""
        return self._initialized

    def get_embedding_info(self) -> dict:
        """
        Get information about the configured embedding.

        Returns:
            Dictionary with embedding configuration details.
        """
        return {
            "provider": self.settings.embedding_provider,
            "model": self.settings.embedding_model,
            "dimension": self.settings.embedding_dimension,
            "initialized": self._initialized,
        }


# === Module-level convenience functions ===

_config_instance: Optional[RAGEmbeddingConfig] = None


def get_rag_embedding_config() -> RAGEmbeddingConfig:
    """
    Get the RAG embedding configuration instance.

    Returns a singleton instance of RAGEmbeddingConfig.

    Returns:
        RAGEmbeddingConfig instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGEmbeddingConfig()
    return _config_instance


def configure_rag_embeddings() -> None:
    """
    Configure RAG embeddings globally.

    Call this at application startup to configure LlamaIndex
    global embedding settings.

    Example:
        >>> from src.rag.embeddings import configure_rag_embeddings
        >>> configure_rag_embeddings()
    """
    config = get_rag_embedding_config()
    config.configure_global_settings()


def get_embed_model() -> BaseEmbedding:
    """
    Get the configured embedding model.

    Convenience function to get the embedding model directly.

    Returns:
        Configured embedding model instance.
    """
    config = get_rag_embedding_config()
    return config.get_embed_model()


def reset_embedding_config() -> None:
    """
    Reset the embedding configuration.

    Clears the cached configuration instance.
    Useful for testing or reconfiguration.
    """
    global _config_instance
    _config_instance = None
    logger.info("Reset RAG embedding configuration")


# === Validation functions ===

def validate_embedding_config() -> dict:
    """
    Validate whether the embedding configuration is correct.

    Checks:
    - Required API keys are set
    - Provider is supported
    - Model can be initialized

    Returns:
        Dictionary with validation results.
    """
    from ..core.config import get_settings
    settings = get_settings()

    results = {
        "valid": True,
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "errors": [],
        "warnings": [],
    }

    # Check API key based on provider
    if settings.embedding_provider == "dashscope":
        if not settings.dashscope_api_key:
            results["valid"] = False
            results["errors"].append("DASHSCOPE_API_KEY is not set")
    elif settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            results["valid"] = False
            results["errors"].append("OPENAI_API_KEY is not set")
    else:
        results["valid"] = False
        results["errors"].append(f"Unsupported provider: {settings.embedding_provider}")

    # Check dimension
    if settings.embedding_provider == "dashscope" and settings.embedding_dimension != 1024:
        results["warnings"].append(
            f"DashScope text-embedding-v4 uses 1024 dimensions, "
            f"but {settings.embedding_dimension} is configured"
        )

    return results
