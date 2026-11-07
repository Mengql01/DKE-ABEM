"""
DCK Module for TRNSYS DCK Generator.

This module provides DCK file generation and validation capabilities.

Components:
- templates: DCK template system based on training set analysis
- format_checker: Strict format validation
- assembler: LLM-driven DCK assembly
- validator: TrnEXE64.exe integration for validation

Usage:
    from src.dck import (
        DCKAssembler,
        DCKValidator,
        DCKFormatChecker,
        DCKTemplateManager,
        assemble_dck,
        validate_dck,
    )

    # Assemble DCK
    result = assemble_dck(
        components=[...],
        equations=[...],
        simulation_settings={...},
        task_description="...",
    )

    # Validate DCK
    validation = validate_dck(result.dck_content)

Design Principles:
- NO hardcoded DCK generation rules
- LLM-driven customization and optimization
- Templates derived from training set
- Strict format validation based on TMF specs
"""

from .templates import (
    DCKSection,
    DCKTemplate,
    DCKTemplateManager,
    HeaderTemplate,
    ControlCardsTemplate,
    UnitTemplate,
    EquationsTemplate,
    LinkStyleTemplate,
    TemplateVariable,
    get_template_manager,
    reset_template_manager,
)

from .format_checker import (
    DCKFormatChecker,
    FormatError,
    ValidationResult as FormatValidationResult,
    ErrorSeverity,
    ErrorType,
    TMFDataLoader,
    get_format_checker,
    reset_format_checker,
    validate_dck as validate_dck_format,
)

from .assembler import (
    DCKAssembler,
    AssemblyResult,
    get_assembler,
    reset_assembler,
    assemble_dck,
)

from .validator import (
    DCKValidator,
    DCKValidationResult,
    TRNSYSError,
    ErrorAnalysis,
    get_validator,
    reset_validator,
    validate_dck,
)

__all__ = [
    # Templates
    "DCKSection",
    "DCKTemplate",
    "DCKTemplateManager",
    "HeaderTemplate",
    "ControlCardsTemplate",
    "UnitTemplate",
    "EquationsTemplate",
    "LinkStyleTemplate",
    "TemplateVariable",
    "get_template_manager",
    "reset_template_manager",
    # Format Checker
    "DCKFormatChecker",
    "FormatError",
    "FormatValidationResult",
    "ErrorSeverity",
    "ErrorType",
    "TMFDataLoader",
    "get_format_checker",
    "reset_format_checker",
    "validate_dck_format",
    # Assembler
    "DCKAssembler",
    "AssemblyResult",
    "get_assembler",
    "reset_assembler",
    "assemble_dck",
    # Validator
    "DCKValidator",
    "DCKValidationResult",
    "TRNSYSError",
    "ErrorAnalysis",
    "get_validator",
    "reset_validator",
    "validate_dck",
]
