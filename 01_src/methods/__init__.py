"""
Methods module - Three experimental methods for DCK generation.

Three Methods (Control Variable Principle - Updated 2026-02-03):
1. B1: MultiAgent_RAG - LLM + RAG + Multi-Agent (no KG, no Memory)
2. B2: MultiAgent_KG - LLM + KG + Multi-Agent (no RAG, no Memory)
3. Proposed: MultiAgent_Memory - LLM + Memory + Multi-Agent (no RAG, no KG, ESR=100% required)

| Method                          | LLM        | Multi-Agent | RAG  | KG   | Memory | ESR Target |
| ------------------------------- | ---------- | ----------- | ---- | ---- | ------ | ---------- |
| **B1: MultiAgent_RAG**          | ✅ DeepSeek | ✅           | ✅    | ❌    | ❌      | Normal     |
| **B2: MultiAgent_KG**           | ✅ DeepSeek | ✅           | ❌    | ✅    | ❌      | Normal     |
| **Proposed: MultiAgent_Memory** | ✅ DeepSeek | ✅           | ❌    | ❌    | ✅      | **100%**   |

Key Classes:
- BaseMethod: Abstract base class for all methods
- GenerationResult: Result of DCK generation
- MethodRegistry: Registry for method classes
- ProposedMultiAgentMemory: The proposed method implementation
"""

from .base_method import BaseMethod, GenerationResult, MethodRegistry

from .proposed_full import (
    ProposedMultiAgentMemory,
    create_proposed_method,
    get_proposed_method,
)

__all__ = [
    # Base
    "BaseMethod",
    "GenerationResult",
    "MethodRegistry",
    # Proposed Method
    "ProposedMultiAgentMemory",
    "create_proposed_method",
    "get_proposed_method",
]
