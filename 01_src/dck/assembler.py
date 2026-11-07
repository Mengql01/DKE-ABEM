"""
DCK Assembler for TRNSYS DCK Generator.

This module provides LLM-driven DCK file assembly.
Implements the "copy when possible" and "copy first, then modify" principles.

Key Features:
1. Template-based DCK assembly
2. Format validation during assembly
3. LLM-driven format optimization
4. Support for Memory/RAG retrieval of similar examples

Design Principles:
- NO hardcoded DCK generation rules
- LLM-driven customization and optimization
- Templates derived from training set
- Strict format validation
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .format_checker import (
    DCKFormatChecker,
    FormatError,
    ValidationResult,
    get_format_checker,
)
from .templates import (
    DCKTemplateManager,
    HeaderTemplate,
    ControlCardsTemplate,
    UnitTemplate,
    EquationsTemplate,
    LinkStyleTemplate,
    get_template_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class AssemblyResult:
    """Result of DCK assembly."""

    dck_content: str
    sections: Dict[str, str] = field(default_factory=dict)
    format_errors: List[FormatError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    assembly_time: float = 0.0
    llm_optimized: bool = False

    @property
    def error_count(self) -> int:
        """Number of format errors."""
        return len([e for e in self.format_errors if e.severity.value == "error"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dck_content": self.dck_content,
            "sections": self.sections,
            "format_errors": [e.to_dict() for e in self.format_errors],
            "warnings": self.warnings,
            "success": self.success,
            "error_count": self.error_count,
            "assembly_time": self.assembly_time,
            "llm_optimized": self.llm_optimized,
        }


class DCKAssembler:
    """
    LLM-driven DCK Assembler.

    Assembles complete DCK files from component data using templates
    and LLM optimization.
    """

    def __init__(
        self,
        method: str = "proposed",
        template_manager: DCKTemplateManager = None,
        format_checker: DCKFormatChecker = None,
    ):
        """
        Initialize the assembler.

        Args:
            method: The method being used ("proposed" or "b3_multiagent_rag").
            template_manager: Template manager instance.
            format_checker: Format checker instance.
        """
        self.method = method
        self.template_manager = template_manager or get_template_manager()
        self.format_checker = format_checker or get_format_checker()
        self._llm = None

    @property
    def llm(self):
        """Get or create LLM instance."""
        if self._llm is None:
            try:
                from ..core.llm_client import create_llm_client
                self._llm = create_llm_client()
            except Exception as e:
                logger.warning(f"Could not create LLM: {e}")
        return self._llm

    def assemble(
        self,
        components: List[Dict[str, Any]],
        equations: List[Dict[str, Any]] = None,
        constants: List[Dict[str, Any]] = None,
        simulation_settings: Dict[str, Any] = None,
        layout_styles: List[Dict[str, Any]] = None,
        link_style_lines: List[str] = None,
        task_description: str = "",
        optimize_with_llm: bool = True,
    ) -> AssemblyResult:
        """
        Assemble a complete DCK file.

        Args:
            components: List of component dictionaries.
            equations: List of equation block dictionaries.
            constants: List of constant dictionaries.
            simulation_settings: Simulation settings dictionary.
            layout_styles: List of layout style dictionaries.
            link_style_lines: Pre-generated LINK_STYLE lines from layout_designer.
            task_description: Task description for header.
            optimize_with_llm: Whether to use LLM for format optimization.

        Returns:
            AssemblyResult with DCK content and validation results.
        """
        import time
        start_time = time.time()

        equations = equations or []
        constants = constants or []
        simulation_settings = simulation_settings or {}
        layout_styles = layout_styles or []
        link_style_lines = link_style_lines or []

        warnings = []
        sections = {}

        try:
            # 1. Build header section
            sections["header"] = self._build_header_section(task_description)

            # 2. Build control cards section
            sections["control"] = self._build_control_section(simulation_settings, constants)

            # 3. Build all unit blocks (components + equations) sorted by unit_number
            sections["all_units"] = self._build_all_units_sorted(components, equations)

            # 4. Build end marker
            sections["end"] = "END"

            # 5. Build link styles section (if any)
            # Prefer pre-generated link_style_lines from layout_designer
            if link_style_lines:
                # link_style_lines can be list of dicts or list of strings
                if link_style_lines and isinstance(link_style_lines[0], dict):
                    # New dict format: each item has LINK and CONNECTION_SET fields
                    ls_lines = ["*!LINK_STYLE"]
                    for item in link_style_lines:
                        ls_lines.append(f"*!LINK {item.get('LINK', '')}")
                        ls_lines.append(f"*!CONNECTION_SET {item.get('CONNECTION_SET', '')}")
                    ls_lines.append("*!LINK_STYLE_END")
                    sections["link_styles"] = "\n".join(ls_lines)
                else:
                    # Legacy string format
                    sections["link_styles"] = "\n".join(link_style_lines)
            elif layout_styles:
                sections["link_styles"] = self._build_link_styles_section(layout_styles, components)
            else:
                sections["link_styles"] = ""

            # 7. Combine all sections
            dck_content = self._combine_sections(sections)

            # 8. Validate format
            validation_result = self.format_checker.validate_complete_dck(dck_content)
            format_errors = validation_result.all_issues

            # 9. If errors and LLM optimization enabled, try to fix
            # IMPORTANT: LLM optimization must NOT alter EQUATIONS blocks or LINK_STYLE blocks
            llm_optimized = False
            if format_errors and optimize_with_llm and self.llm:
                try:
                    # Retrieve format examples for reference
                    type_ids = [c.get("type_number", "") for c in components]
                    format_examples = self._retrieve_format_examples(type_ids)

                    # Use LLM to optimize
                    optimized_content = self._llm_optimize_format(
                        dck_content,
                        format_errors,
                        format_examples,
                    )

                    if optimized_content and optimized_content != dck_content:
                        # Safety check: reject if LLM destroyed EQUATIONS or LINK_STYLE
                        orig_eq_count = dck_content.count('* EQUATIONS "')
                        opt_eq_count = optimized_content.count('* EQUATIONS "')
                        orig_link_style = "*!LINK_STYLE" in dck_content
                        opt_link_style = "*!LINK_STYLE" in optimized_content

                        if opt_eq_count < orig_eq_count:
                            logger.warning(
                                f"LLM optimization rejected: EQUATIONS blocks reduced "
                                f"from {orig_eq_count} to {opt_eq_count}"
                            )
                        elif orig_link_style and not opt_link_style:
                            logger.warning(
                                "LLM optimization rejected: LINK_STYLE block was removed"
                            )
                        else:
                            # Re-validate
                            new_validation = self.format_checker.validate_complete_dck(optimized_content)
                            if new_validation.error_count < validation_result.error_count:
                                dck_content = optimized_content
                                format_errors = new_validation.all_issues
                                llm_optimized = True
                                warnings.append("DCK content was optimized by LLM")

                except Exception as e:
                    logger.warning(f"LLM optimization failed: {e}")
                    warnings.append(f"LLM optimization failed: {e}")

            assembly_time = time.time() - start_time

            return AssemblyResult(
                dck_content=dck_content,
                sections=sections,
                format_errors=format_errors,
                warnings=warnings,
                success=validation_result.error_count == 0 or llm_optimized,
                assembly_time=assembly_time,
                llm_optimized=llm_optimized,
            )

        except Exception as e:
            logger.error(f"Assembly error: {e}")
            return AssemblyResult(
                dck_content="",
                sections=sections,
                format_errors=[FormatError(
                    error_type="assembly_error",
                    message=str(e),
                    severity="error",
                )],
                warnings=warnings,
                success=False,
                assembly_time=time.time() - start_time,
            )

    def _build_header_section(self, task_description: str) -> str:
        """Build the header section."""
        return HeaderTemplate.render(task_description=task_description)

    def _build_control_section(
        self,
        simulation_settings: Dict[str, Any],
        constants: List[Dict[str, Any]],
    ) -> str:
        """Build the control cards section."""
        # LIMITS statement defaults come from performance_config (single source of truth)
        from ..core.performance_config import get_performance_config
        perf_config = get_performance_config()
        return ControlCardsTemplate.render(
            start_time=simulation_settings.get("start_time", 0),
            stop_time=simulation_settings.get("stop_time", 8760),
            time_step=simulation_settings.get("time_step", 1),
            additional_constants=constants,
            integration_tol=simulation_settings.get("integration_tolerance", 0.001),
            convergence_tol=simulation_settings.get("convergence_tolerance", 0.001),
            max_iterations=simulation_settings.get("max_iterations", perf_config.limits_iterations_before_warning),
            max_warnings=simulation_settings.get("max_warnings", perf_config.limits_warnings_before_error),
            trace_limit=simulation_settings.get("trace_limit", perf_config.limits_iterations_before_trace),
            dfq=simulation_settings.get("dfq", 1),
            width=simulation_settings.get("width", 80),
            solver_type=simulation_settings.get("solver_type", 0),
            min_relaxation=simulation_settings.get("min_relaxation", 1),
            max_relaxation=simulation_settings.get("max_relaxation", 1),
            nan_check=simulation_settings.get("nan_check", 0),
            overwrite_check=simulation_settings.get("overwrite_check", 0),
            time_report=simulation_settings.get("time_report", 0),
            eqsolver=simulation_settings.get("eqsolver", 0),
        )

    def _build_equations_section(self, equations: List[Dict[str, Any]]) -> str:
        """Build the equations section."""
        sections = []

        for eq_block in equations:
            section = EquationsTemplate.render(
                unit_number=eq_block.get("unit_number", 10),
                unit_name=eq_block.get("unit_name", "Equations"),
                equations=eq_block.get("equations", []),
                position_x=eq_block.get("position_x", 100),
                position_y=eq_block.get("position_y", 100),
                layer=eq_block.get("layer", "Main"),
            )
            sections.append(section)

        return "\n\n".join(sections)

    def _build_all_units_sorted(
        self,
        components: List[Dict[str, Any]],
        equations: List[Dict[str, Any]],
    ) -> str:
        """Build all unit blocks (components + equations) sorted by unit_number."""
        # Collect all items with their unit_number and rendered content
        items: List[Tuple[int, str]] = []

        for comp in components:
            unit_num = int(comp.get("unit_number", 0))
            section = self._build_single_unit(comp)
            items.append((unit_num, section))

        for eq_block in equations:
            unit_num = int(eq_block.get("unit_number", 0))
            section = EquationsTemplate.render(
                unit_number=eq_block.get("unit_number", 10),
                unit_name=eq_block.get("unit_name", "Equations"),
                equations=eq_block.get("equations", []),
                position_x=eq_block.get("position_x", 100),
                position_y=eq_block.get("position_y", 100),
                layer=eq_block.get("layer", "Main"),
            )
            items.append((unit_num, section))

        # Sort by unit_number
        items.sort(key=lambda x: x[0])

        return "\n\n".join(section for _, section in items)

    def _build_units_section(self, components: List[Dict[str, Any]]) -> str:
        """Build the units section."""
        sections = []

        for comp in components:
            section = self._build_single_unit(comp)
            sections.append(section)

        return "\n\n".join(sections)

    def _build_single_unit(self, component: Dict[str, Any]) -> str:
        """Build a single UNIT block."""
        return UnitTemplate.render(
            unit_number=component.get("unit_number", 1),
            type_number=component.get("type_number", "1"),
            unit_name=component.get("unit_name", "Unit"),
            parameters=component.get("parameters", []),
            inputs=component.get("inputs", []),
            initial_values=component.get("initial_values", []),
            position_x=component.get("position_x", 100),
            position_y=component.get("position_y", 100),
            layer=component.get("layer", "Main"),
            model_path=component.get("model_path", ""),
            external_files=component.get("external_files", []),
            labels=component.get("labels", []),
            specific_properties=component.get("specific_properties", []),
        )

    def _build_link_styles_section(
        self,
        layout_styles: List[Dict[str, Any]],
        components: List[Dict[str, Any]],
    ) -> str:
        """Build the link styles section."""
        return LinkStyleTemplate.render(layout_styles)

    def _combine_sections(self, sections: Dict[str, str]) -> str:
        """Combine all sections into a complete DCK file."""
        parts = []

        # Header
        if sections.get("header"):
            parts.append(sections["header"])

        # Units section header
        parts.append("*******************************************************************************\n*** Units \n*******************************************************************************")

        # Control cards
        if sections.get("control"):
            parts.append(sections["control"])

        # All units (components + equations) sorted by unit_number
        if sections.get("all_units"):
            parts.append(sections["all_units"])

        # End marker
        if sections.get("end"):
            parts.append(sections["end"])

        # Combine with double newlines
        result = "\n\n".join(parts)

        # Link styles (after END) - special handling for spacing
        if sections.get("link_styles"):
            # END and LINK_STYLE should have exactly one blank line between them
            result += "\n" + sections["link_styles"]

        return result

    def _retrieve_format_examples(self, type_ids: List[str]) -> Dict[str, List[str]]:
        """
        Retrieve format examples from training set or Memory/RAG.

        This implements the "copy when possible" principle.

        Args:
            type_ids: List of Type IDs to retrieve examples for.

        Returns:
            Dictionary mapping type_id to list of example strings.
        """
        examples = {}

        # Try to get examples from template manager
        try:
            examples = self.template_manager.get_similar_unit_examples(type_ids, limit=2)
        except Exception as e:
            logger.debug(f"Could not get examples from template manager: {e}")

        # If using b3 method, also try RAG
        if self.method == "b3_multiagent_rag":
            try:
                from ..agents.tools.rag_tools import rag_search_dck_format_examples
                for type_id in type_ids:
                    if type_id not in examples:
                        result = rag_search_dck_format_examples.invoke({
                            "type_ids": [type_id],
                            "limit": 2,
                        })
                        if result:
                            examples[type_id] = [result]
            except Exception as e:
                logger.debug(f"Could not get examples from RAG: {e}")

        return examples

    def _llm_optimize_format(
        self,
        dck_content: str,
        format_errors: List[FormatError],
        format_examples: Dict[str, List[str]],
    ) -> str:
        """
        Use LLM to optimize DCK format.

        This implements the "copy first, then modify" principle.

        Args:
            dck_content: Current DCK content.
            format_errors: List of format errors to fix.
            format_examples: Reference examples from training set.

        Returns:
            Optimized DCK content.
        """
        if not self.llm:
            return dck_content

        # Build error description
        error_desc = "\n".join([
            f"- {e.error_type}: {e.message}" + (f" (suggestion: {e.suggestion})" if e.suggestion else "")
            for e in format_errors[:10]  # Limit to 10 errors
        ])

        # Build examples description
        examples_desc = ""
        if format_examples:
            examples_list = []
            for type_id, exs in list(format_examples.items())[:3]:
                if exs:
                    examples_list.append(f"Type{type_id} example:\n{exs[0][:500]}")
            examples_desc = "\n\n".join(examples_list)

        # Build prompt
        system_prompt = """You are a TRNSYS DCK file formatting expert. Your task is to repair formatting errors in the DCK file.

[Important Rules]
1. Only repair formatting errors; do not change the functional logic
2. Keep all UNIT numbers, TYPE numbers, and parameter values unchanged
3. Ensure the PARAMETERS count matches its declaration
4. Ensure the INPUTS count matches its declaration
5. Ensure the number of initial values matches the INPUTS count
6. Refer to the provided formatting examples

[Output Requirement]
Directly output the complete repaired DCK file content without adding any explanation or markdown markers."""

        user_prompt = f"""Please repair the formatting errors in the following DCK file:

[Formatting Errors]
{error_desc}

[Reference Formatting Examples]
{examples_desc if examples_desc else "No examples available"}

[DCK File to Repair]
{dck_content[:8000]}

Please directly output the complete repaired DCK file:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            optimized = response.content.strip()

            # Basic validation - should still look like a DCK file
            if "VERSION" in optimized and "END" in optimized:
                return optimized

        except Exception as e:
            logger.warning(f"LLM optimization failed: {e}")

        return dck_content

    def assemble_from_state(self, state: Dict[str, Any]) -> AssemblyResult:
        """
        Assemble DCK from LangGraph state.

        Args:
            state: DCKGeneratorState dictionary.

        Returns:
            AssemblyResult.
        """
        return self.assemble(
            components=state.get("components", []),
            equations=state.get("equations", []),
            constants=state.get("constants", []),
            simulation_settings=state.get("simulation_settings", {}),
            layout_styles=state.get("layout_styles", []),
            link_style_lines=state.get("link_style_lines", []),
            task_description=state.get("task_description", ""),
        )


# Global assembler instance
_assembler: Optional[DCKAssembler] = None


def get_assembler(method: str = "proposed") -> DCKAssembler:
    """Get the global assembler instance."""
    global _assembler
    if _assembler is None or _assembler.method != method:
        _assembler = DCKAssembler(method=method)
    return _assembler


def reset_assembler():
    """Reset the global assembler (for testing)."""
    global _assembler
    _assembler = None


def assemble_dck(
    components: List[Dict[str, Any]],
    equations: List[Dict[str, Any]] = None,
    constants: List[Dict[str, Any]] = None,
    simulation_settings: Dict[str, Any] = None,
    layout_styles: List[Dict[str, Any]] = None,
    task_description: str = "",
    method: str = "proposed",
) -> AssemblyResult:
    """
    Convenience function to assemble DCK content.

    Args:
        components: List of component dictionaries.
        equations: List of equation block dictionaries.
        constants: List of constant dictionaries.
        simulation_settings: Simulation settings dictionary.
        layout_styles: List of layout style dictionaries.
        task_description: Task description for header.
        method: Method being used.

    Returns:
        AssemblyResult.
    """
    assembler = get_assembler(method)
    return assembler.assemble(
        components=components,
        equations=equations,
        constants=constants,
        simulation_settings=simulation_settings,
        layout_styles=layout_styles,
        task_description=task_description,
    )
