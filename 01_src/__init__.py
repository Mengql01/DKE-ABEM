"""TRNSYS DCK Generator - Memory-Enhanced Multi-Agent Framework."""

__version__ = "1.0.0"


def _initialize():
    """Initialize the application environment including LangSmith tracing."""
    from .core.config import setup_environment
    setup_environment()


# Auto-initialize on import
_initialize()
