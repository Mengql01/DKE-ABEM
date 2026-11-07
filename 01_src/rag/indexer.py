"""
RAG Index Builder for TRNSYS DCK Generator.

Builds three specialized indexes using LlamaIndex official SDK:
1. dck_systems - parsed training-set DCK data (system-level knowledge)
2. tmf_components - parsed TMF component data (component-level knowledge)
3. dck_format_examples - raw training-set DCK files (format learning)

This module supports the "copy when possible" and "copy first, then modify" principles:
- copy when possible: Retrieve similar system designs as references
- copy first, then modify: Retrieve format templates to adapt

Used by:
- B2: RAG method (single LLM + RAG)
- B3: MultiAgent_RAG method (multi-agent + RAG)

Note: Proposed: MultiAgent_Memory method does NOT use RAG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

if TYPE_CHECKING:
    from ..core.config import Settings

logger = logging.getLogger(__name__)


class RAGIndexer:
    """
    TRNSYS RAG index builder.

    Builds and manages three specialized vector indexes for TRNSYS DCK generation:
    - DCK Systems Index: System-level knowledge from 176 training DCK files
    - TMF Components Index: Component-level knowledge from 765 TMF files
    - DCK Format Index: Format examples from raw DCK files

    Uses LlamaIndex official SDK with ChromaDB for vector storage.
    NO hardcoded bypass classes.
    """

    # Index collection names
    COLLECTION_PARSED_DCK_SYSTEMS = "trnsys_parsed_dck_systems"
    COLLECTION_TMF_COMPONENTS = "trnsys_tmf_components"
    COLLECTION_DCK_SYSTEMS = "trnsys_dck_systems"

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize RAG Indexer.

        Args:
            settings: Application settings. If None, loads from get_settings().
        """
        if settings is None:
            from ..core.config import get_settings
            settings = get_settings()

        self.settings = settings
        self.rag_dir = self.settings.vector_store_dir / "rag"
        self.rag_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = None
        self._indexes: Dict[str, VectorStoreIndex] = {}

        # Configure embedding before building indexes
        from .embeddings import configure_rag_embeddings
        configure_rag_embeddings()

    @property
    def chroma_client(self):
        """Get or create ChromaDB persistent client."""
        if self._chroma_client is None:
            try:
                import chromadb
            except ImportError as e:
                raise ImportError(
                    "chromadb is required for RAG indexing. "
                    "Install with: pip install chromadb"
                ) from e

            self._chroma_client = chromadb.PersistentClient(
                path=str(self.rag_dir)
            )
            logger.info(f"Initialized ChromaDB client at {self.rag_dir}")

        return self._chroma_client

    # === Index Building Methods ===

    def build_dck_systems_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        Build the parsed DCK systems index (parsed training-set DCK data).

        This index contains system-level knowledge:
        - System configurations and component combinations
        - Connection patterns between components
        - Parameter configurations

        Supports the "copy when possible" principle: retrieve similar system designs.

        Args:
            force_rebuild: If True, rebuild even if index exists.

        Returns:
            VectorStoreIndex for parsed DCK systems.
        """
        collection_name = self.COLLECTION_PARSED_DCK_SYSTEMS

        # Check if index already exists
        if not force_rebuild and collection_name in self._indexes:
            return self._indexes[collection_name]

        # Try to load existing index
        if not force_rebuild:
            existing_index = self._load_index(collection_name)
            if existing_index is not None:
                self._indexes[collection_name] = existing_index
                return existing_index

        logger.info("Building DCK systems index from training data...")

        # Load training DCK parsed data
        documents = self._create_dck_system_documents()

        if not documents:
            logger.warning("No DCK system documents found. Creating empty index.")
            documents = [Document(text="Empty index placeholder", metadata={"type": "placeholder"})]

        # Create index with ChromaDB
        index = self._create_index(collection_name, documents)
        self._indexes[collection_name] = index

        logger.info(f"Built DCK systems index with {len(documents)} documents")
        return index

    def build_tmf_components_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        Build the TMF components index (parsed TMF component data).

        This index contains component-level knowledge:
        - Component specifications and parameters
        - Input/output definitions
        - Usage guidelines

        Supports component selection and parameter validation.

        Args:
            force_rebuild: If True, rebuild even if index exists.

        Returns:
            VectorStoreIndex for TMF components.
        """
        collection_name = self.COLLECTION_TMF_COMPONENTS

        # Check if index already exists
        if not force_rebuild and collection_name in self._indexes:
            return self._indexes[collection_name]

        # Try to load existing index
        if not force_rebuild:
            existing_index = self._load_index(collection_name)
            if existing_index is not None:
                self._indexes[collection_name] = existing_index
                return existing_index

        logger.info("Building TMF components index...")

        # Load TMF parsed data
        documents = self._create_tmf_component_documents()

        if not documents:
            logger.warning("No TMF component documents found. Creating empty index.")
            documents = [Document(text="Empty index placeholder", metadata={"type": "placeholder"})]

        # Create index with ChromaDB
        index = self._create_index(collection_name, documents)
        self._indexes[collection_name] = index

        logger.info(f"Built TMF components index with {len(documents)} documents")
        return index

    def build_dck_format_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        Build the DCK format examples index (raw training-set DCK files).

        This index contains format examples:
        - Raw DCK file content for format learning
        - Section structures (VERSION, SIMULATION, UNITS, etc.)
        - Connection syntax examples

        Supports the "copy first, then modify" principle: retrieve format templates.

        Args:
            force_rebuild: If True, rebuild even if index exists.

        Returns:
            VectorStoreIndex for DCK format examples.
        """
        collection_name = self.COLLECTION_DCK_SYSTEMS

        # Check if index already exists
        if not force_rebuild and collection_name in self._indexes:
            return self._indexes[collection_name]

        # Try to load existing index
        if not force_rebuild:
            existing_index = self._load_index(collection_name)
            if existing_index is not None:
                self._indexes[collection_name] = existing_index
                return existing_index

        logger.info("Building DCK format examples index...")

        # Load raw DCK files
        documents = self._create_dck_format_documents()

        if not documents:
            logger.warning("No DCK format documents found. Creating empty index.")
            documents = [Document(text="Empty index placeholder", metadata={"type": "placeholder"})]

        # Create index with ChromaDB
        index = self._create_index(collection_name, documents)
        self._indexes[collection_name] = index

        logger.info(f"Built DCK format index with {len(documents)} documents")
        return index

    def build_all_indexes(self, force_rebuild: bool = False) -> Dict[str, VectorStoreIndex]:
        """
        Build all three indexes.

        Args:
            force_rebuild: If True, rebuild all indexes even if they exist.

        Returns:
            Dictionary mapping collection names to indexes.
        """
        logger.info("Building all RAG indexes...")

        indexes = {
            self.COLLECTION_PARSED_DCK_SYSTEMS: self.build_dck_systems_index(force_rebuild),
            self.COLLECTION_TMF_COMPONENTS: self.build_tmf_components_index(force_rebuild),
            self.COLLECTION_DCK_SYSTEMS: self.build_dck_format_index(force_rebuild),
        }

        logger.info(f"Built {len(indexes)} indexes successfully")
        return indexes

    # === Index Loading Methods ===

    def _load_index(self, collection_name: str) -> Optional[VectorStoreIndex]:
        """
        Load an existing index from ChromaDB.

        Args:
            collection_name: Name of the collection to load.

        Returns:
            VectorStoreIndex if exists, None otherwise.
        """
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore

            # Check if collection exists
            existing_collections = [c.name for c in self.chroma_client.list_collections()]
            if collection_name not in existing_collections:
                logger.debug(f"Collection {collection_name} does not exist")
                return None

            # Load existing collection
            chroma_collection = self.chroma_client.get_collection(collection_name)

            # Check if collection has data
            if chroma_collection.count() == 0:
                logger.debug(f"Collection {collection_name} is empty")
                return None

            # Create vector store and index
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)

            logger.info(f"Loaded existing index: {collection_name} ({chroma_collection.count()} vectors)")
            return index

        except Exception as e:
            logger.warning(f"Failed to load index {collection_name}: {e}")
            return None

    def load_all_indexes(self) -> Dict[str, VectorStoreIndex]:
        """
        Load all existing indexes.

        Returns:
            Dictionary mapping collection names to loaded indexes.
        """
        indexes = {}

        for collection_name in [
            self.COLLECTION_PARSED_DCK_SYSTEMS,
            self.COLLECTION_TMF_COMPONENTS,
            self.COLLECTION_DCK_SYSTEMS,
        ]:
            index = self._load_index(collection_name)
            if index is not None:
                indexes[collection_name] = index
                self._indexes[collection_name] = index

        logger.info(f"Loaded {len(indexes)} existing indexes")
        return indexes

    def get_index(self, name: str) -> Optional[VectorStoreIndex]:
        """
        Get a specific index.

        Args:
            name: Collection name (use class constants).

        Returns:
            VectorStoreIndex if available, None otherwise.
        """
        if name in self._indexes:
            return self._indexes[name]

        # Try to load from disk
        index = self._load_index(name)
        if index is not None:
            self._indexes[name] = index

        return index

    # === Document Creation Methods ===

    def _create_dck_system_documents(self) -> List[Document]:
        """
        Create documents from parsed DCK system data.

        Loads 176 training DCK parsed JSON and creates documents
        for system-level knowledge retrieval.

        Returns:
            List of Document objects.
        """
        documents = []
        parsed_path = self.settings.train_dck_parsed_path

        if not parsed_path.exists():
            logger.warning(f"Training DCK parsed file not found: {parsed_path}")
            return documents

        try:
            with open(parsed_path, "r", encoding="utf-8") as f:
                parsed_data = json.load(f)

            # Handle the actual JSON structure: {"statistics": ..., "results": [...]}
            if isinstance(parsed_data, dict) and "results" in parsed_data:
                systems = parsed_data["results"]
            elif isinstance(parsed_data, list):
                systems = parsed_data
            else:
                # Fallback: treat dict values as systems
                systems = list(parsed_data.values())

            logger.info(f"Found {len(systems)} DCK systems to index")

            for idx, system in enumerate(systems):
                doc = self._create_single_dck_system_document(system, idx)
                if doc is not None:
                    documents.append(doc)

            logger.info(f"Created {len(documents)} DCK system documents")

        except Exception as e:
            logger.error(f"Error loading DCK parsed data: {e}")

        return documents

    def _create_single_dck_system_document(
        self, system: Dict[str, Any], idx: int
    ) -> Optional[Document]:
        """
        Create a document from a single parsed DCK system.

        Extracts all 15 unit sub-fields, all 9 connection sub-fields,
        all 3 constant sub-fields, plus system-level fields:
        deck_id, source_file, global_settings, statistics.

        Args:
            system: Parsed DCK system data.
            idx: Index for identification.

        Returns:
            Document object or None if invalid.
        """
        try:
            # Extract key information
            filename = system.get("filename", f"system_{idx}")
            deck_id = system.get("deck_id", "")
            source_file = system.get("source_file", "")
            units = system.get("units", [])
            connections = system.get("connections", [])
            constants = system.get("constants", [])
            global_settings = system.get("global_settings", {})
            statistics = system.get("statistics", {})

            # Build document text
            text_parts = [
                f"DCK System: {filename}",
                f"Deck ID: {deck_id}",
                f"Source File: {source_file}",
            ]

            # Add global_settings (all sub-fields)
            if global_settings:
                text_parts.append("\nGlobal Settings:")
                sim = global_settings.get("simulation", {})
                if sim:
                    text_parts.append(f"  Simulation: start={sim.get('start','')}, stop={sim.get('stop','')}, step={sim.get('step','')}, start_expr={sim.get('start_expr','')}, stop_expr={sim.get('stop_expr','')}, step_expr={sim.get('step_expr','')}")
                tol = global_settings.get("tolerances", {})
                if tol:
                    text_parts.append(f"  Tolerances: integration={tol.get('integration','')}, convergence={tol.get('convergence','')}")
                lim = global_settings.get("limits", {})
                if lim:
                    text_parts.append(f"  Limits: max_iterations={lim.get('max_iterations','')}, max_warnings={lim.get('max_warnings','')}, trace_limit={lim.get('trace_limit','')}")
                sol = global_settings.get("solver", {})
                if sol:
                    text_parts.append(f"  Solver: solver_statement={sol.get('solver_statement','')}, min_relaxation={sol.get('min_relaxation','')}, max_relaxation={sol.get('max_relaxation','')}")
                other = global_settings.get("other_cards", {})
                if other:
                    other_str = ", ".join(f"{k}={v}" for k, v in other.items())
                    text_parts.append(f"  Other Cards: {other_str}")

            # Add statistics (all sub-fields)
            if statistics:
                text_parts.append("\nStatistics:")
                for stat_key, stat_val in statistics.items():
                    if stat_key == "global_type_usage" and isinstance(stat_val, dict):
                        usage_str = ", ".join(f"{k}:{v}" for k, v in stat_val.items())
                        text_parts.append(f"  {stat_key}: {usage_str}")
                    else:
                        text_parts.append(f"  {stat_key}: {stat_val}")

            # Add unit/component information - all units, all 15 sub-fields
            text_parts.append(f"\nUnits/Components ({len(units)}):")
            for unit in units:
                unit_type = unit.get("full_type_id", unit.get("type_number", "Unknown"))
                unit_number = unit.get("unit_number", "")
                unit_id = unit.get("unit_id", "")
                u_type = unit.get("unit_type", "")
                metadata_info = unit.get("metadata", {})
                unit_name = metadata_info.get("unit_name", metadata_info.get("name", "")) if isinstance(metadata_info, dict) else ""
                model_path = metadata_info.get("model_path", "") if isinstance(metadata_info, dict) else ""
                position = metadata_info.get("position", {}) if isinstance(metadata_info, dict) else {}
                layer = metadata_info.get("layer", "") if isinstance(metadata_info, dict) else ""

                text_parts.append(f"  - {unit_type} (Unit {unit_number}): {unit_name}")
                text_parts.append(f"    unit_id={unit_id}, unit_type={u_type}, model_path={model_path}")
                if position:
                    text_parts.append(f"    position=({position.get('x','')},{position.get('y','')}), layer={layer}")

                # Parameters
                params = unit.get("parameters", [])
                params_count = unit.get("parameters_count", len(params))
                if params:
                    text_parts.append(f"    Parameters ({params_count}):")
                    for p in params:
                        text_parts.append(f"      [{p.get('index','')}] value={p.get('value','')}, raw={p.get('raw_line','')}")

                # Inputs
                inputs = unit.get("inputs", [])
                inputs_count = unit.get("inputs_count", len(inputs))
                if inputs:
                    text_parts.append(f"    Inputs ({inputs_count}):")
                    for inp in inputs:
                        text_parts.append(f"      [{inp.get('index','')}] value={inp.get('value','')}, raw={inp.get('raw_line','')}")

                # Initial input values
                init_vals = unit.get("initial_input_values", [])
                if init_vals:
                    text_parts.append(f"    Initial Input Values: {', '.join(str(v) for v in init_vals)}")

                # Equations
                equations = unit.get("equations", [])
                eq_count = unit.get("equations_count", len(equations))
                if equations:
                    text_parts.append(f"    Equations ({eq_count}):")
                    for eq in equations:
                        text_parts.append(f"      {eq.get('variable','')} = {eq.get('expression','')}")

                # External files
                ext_files = unit.get("external_files", [])
                if ext_files:
                    text_parts.append(f"    External Files:")
                    for ef in ext_files:
                        if isinstance(ef, dict):
                            text_parts.append(f"      {ef}")
                        else:
                            text_parts.append(f"      {ef}")

                # Labels
                labels = unit.get("labels", [])
                if labels:
                    text_parts.append(f"    Labels: {', '.join(str(l) for l in labels)}")

            # Add connection details - all connections, all 9 sub-fields
            text_parts.append(f"\nConnections ({len(connections)}):")
            for conn in connections:
                if isinstance(conn, dict):
                    conn_type = conn.get("connection_type", "")

                    # Extract all 9 possible fields
                    source_unit = conn.get("source_unit", "")
                    source_output = conn.get("source_output", "")
                    source_equation_unit = conn.get("source_equation_unit", "")
                    source_variable = conn.get("source_variable", "")
                    target_unit = conn.get("target_unit", "")
                    target_input = conn.get("target_input", "")
                    target_equation_unit = conn.get("target_equation_unit", "")
                    target_variable = conn.get("target_variable", "")

                    # Format based on connection type
                    if conn_type == "unit_to_unit":
                        conn_str = f"  - [unit_to_unit] Unit {source_unit}:{source_output} -> Unit {target_unit}:{target_input}"
                    elif conn_type == "equation_to_unit":
                        conn_str = f"  - [equation_to_unit] Equation {source_equation_unit}:{source_variable} -> Unit {target_unit}:{target_input}"
                    elif conn_type == "unit_to_equation":
                        conn_str = f"  - [unit_to_equation] Unit {source_unit}:{source_output} -> Equation {target_equation_unit}:{target_input}"
                    elif conn_type == "equation_to_equation":
                        conn_str = f"  - [equation_to_equation] Equation {source_equation_unit}:{source_variable} -> Equation {target_equation_unit}:{target_variable}"
                    else:
                        # Fallback for unknown types
                        conn_str = f"  - [{conn_type}] {conn}"

                    text_parts.append(conn_str)

            # Add constants - all constants, all 3 sub-fields
            if constants:
                const_list = constants if isinstance(constants, list) else []
                text_parts.append(f"\nConstants ({len(const_list)}):")
                for const in const_list:
                    if isinstance(const, dict):
                        c_name = const.get("name", "")
                        c_expr = const.get("expression", "")
                        c_val = const.get("evaluated_value", "")
                        text_parts.append(f"  - {c_name} = {c_expr} (value={c_val})")

            text = "\n".join(text_parts)

            # Extract type IDs for metadata
            type_ids = list(set(
                unit.get("full_type_id", unit.get("type_number", "")).replace("Type", "").replace("TYPE", "")
                for unit in units
                if unit.get("full_type_id") or unit.get("type_number")
            ))

            metadata = {
                "source": "dck_systems",
                "filename": filename,
                "deck_id": deck_id,
                "unit_count": len(units),
                "connection_count": len(connections),
                "type_ids": ",".join(type_ids[:10]),  # Limit metadata size
            }

            return Document(text=text, metadata=metadata)

        except Exception as e:
            logger.warning(f"Error creating DCK system document {idx}: {e}")
            return None

    def _create_tmf_component_documents(self) -> List[Document]:
        """
        Create documents from parsed TMF component data.

        Loads 765 TMF parsed JSON and creates documents
        for component-level knowledge retrieval.

        Returns:
            List of Document objects.
        """
        documents = []
        tmf_path = self.settings.tmf_parsed_path

        if not tmf_path.exists():
            logger.warning(f"TMF parsed file not found: {tmf_path}")
            return documents

        try:
            with open(tmf_path, "r", encoding="utf-8") as f:
                tmf_data = json.load(f)

            # Handle the actual JSON structure: {"results": {...}, ...metadata...}
            if isinstance(tmf_data, dict) and "results" in tmf_data:
                results = tmf_data["results"]
                if isinstance(results, dict):
                    # results is a dict like {"Type519b.tmf": {...}, ...}
                    for tmf_name, component in results.items():
                        doc = self._create_single_tmf_document(tmf_name, component)
                        if doc is not None:
                            documents.append(doc)
                elif isinstance(results, list):
                    for idx, component in enumerate(results):
                        tmf_name = component.get("name", f"tmf_{idx}")
                        doc = self._create_single_tmf_document(tmf_name, component)
                        if doc is not None:
                            documents.append(doc)
            elif isinstance(tmf_data, dict):
                # Fallback: iterate over dict items
                for tmf_name, component in tmf_data.items():
                    if isinstance(component, dict):
                        doc = self._create_single_tmf_document(tmf_name, component)
                        if doc is not None:
                            documents.append(doc)
            elif isinstance(tmf_data, list):
                for idx, component in enumerate(tmf_data):
                    tmf_name = component.get("name", f"tmf_{idx}")
                    doc = self._create_single_tmf_document(tmf_name, component)
                    if doc is not None:
                        documents.append(doc)

            logger.info(f"Created {len(documents)} TMF component documents")

        except Exception as e:
            logger.error(f"Error loading TMF data: {e}")

        return documents

    def _create_single_tmf_document(
        self, tmf_name: str, component: Dict[str, Any]
    ) -> Optional[Document]:
        """
        Create a document from a single parsed TMF component.

        Extracts all 11 sub-fields for parameters/inputs/outputs,
        plus component-level fields: tmf_path, variable_counts,
        specific_properties, external_files, tags.

        Args:
            tmf_name: TMF file name.
            component: Parsed TMF component data.

        Returns:
            Document object or None if invalid.
        """
        try:
            # Extract key information
            type_number = component.get("type_number", "")
            description = component.get("description", "")
            parameters = component.get("parameters", [])
            inputs = component.get("inputs", [])
            outputs = component.get("outputs", [])
            tmf_path = component.get("tmf_path", "")
            variable_counts = component.get("variable_counts", {})
            specific_properties = component.get("specific_properties", [])
            external_files = component.get("external_files", [])
            tags = component.get("tags", [])

            # Build document text
            text_parts = [
                f"TMF Component: {tmf_name}",
                f"Type: {type_number}",
                f"Description: {description}",
                f"TMF Path: {tmf_path}",
            ]

            # Add variable_counts (all sub-fields)
            if variable_counts:
                vc_str = ", ".join(f"{k}={v}" for k, v in variable_counts.items())
                text_parts.append(f"Variable Counts: {vc_str}")

            # Add tags
            if tags:
                text_parts.append(f"Tags: {', '.join(str(t) for t in tags)}")

            # Helper to format a variable item with all 11 sub-fields
            def _format_variable(var: Dict[str, Any]) -> str:
                idx = var.get("index", "")
                name = var.get("name", "")
                desc = var.get("description", "")
                data_type = var.get("data_type", "")
                io_type = var.get("io_type", "")
                default_val = var.get("default_value", "")
                unit = var.get("unit", "")
                dimension = var.get("dimension", "")
                min_val = var.get("min_value", "")
                max_val = var.get("max_value", "")
                category = var.get("category", "")
                parts = [f"  - [{idx}] {name} [{unit}]"]
                detail_parts = []
                if default_val:
                    detail_parts.append(f"default={default_val}")
                if desc:
                    detail_parts.append(f"desc={desc}")
                if data_type:
                    detail_parts.append(f"type={data_type}")
                if io_type:
                    detail_parts.append(f"io={io_type}")
                if dimension:
                    detail_parts.append(f"dim={dimension}")
                if min_val:
                    detail_parts.append(f"min={min_val}")
                if max_val:
                    detail_parts.append(f"max={max_val}")
                if category:
                    detail_parts.append(f"cat={category}")
                if detail_parts:
                    parts.append(f"    ({', '.join(detail_parts)})")
                return "\n".join(parts)

            # Add parameters - all items, all 11 sub-fields
            if parameters:
                text_parts.append(f"\nParameters ({len(parameters)}):")
                for param in parameters:
                    text_parts.append(_format_variable(param))

            # Add inputs - all items, all 11 sub-fields
            if inputs:
                text_parts.append(f"\nInputs ({len(inputs)}):")
                for inp in inputs:
                    text_parts.append(_format_variable(inp))

            # Add outputs - all items, all 11 sub-fields
            if outputs:
                text_parts.append(f"\nOutputs ({len(outputs)}):")
                for out in outputs:
                    text_parts.append(_format_variable(out))

            # Add specific_properties (all sub-fields)
            if specific_properties:
                text_parts.append(f"\nSpecific Properties ({len(specific_properties)}):")
                for sp in specific_properties:
                    if isinstance(sp, dict):
                        sp_prop = sp.get("specific_property", "")
                        sp_note = sp.get("note", "")
                        sp_default = sp.get("default", "")
                        text_parts.append(f"  - {sp_prop}")
                        if sp_note:
                            text_parts.append(f"    note={sp_note}")
                        if sp_default:
                            text_parts.append(f"    default={sp_default}")

            # Add external_files (all sub-fields)
            if external_files:
                text_parts.append(f"\nExternal Files ({len(external_files)}):")
                for ef in external_files:
                    if isinstance(ef, dict):
                        ef_prompt = ef.get("prompt", "")
                        ef_path = ef.get("default_path", "")
                        text_parts.append(f"  - prompt={ef_prompt}, path={ef_path}")

            text = "\n".join(text_parts)

            metadata = {
                "source": "tmf_components",
                "tmf_name": tmf_name,
                "type_number": str(type_number),
                "parameter_count": len(parameters),
                "input_count": len(inputs),
                "output_count": len(outputs),
            }

            return Document(text=text, metadata=metadata)

        except Exception as e:
            logger.warning(f"Error creating TMF document {tmf_name}: {e}")
            return None

    def _create_dck_format_documents(self) -> List[Document]:
        """
        Create format-learning documents from raw DCK files.

        Loads 176 training DCK raw files and creates documents
        for format learning and template retrieval.

        Returns:
            List of Document objects.
        """
        documents = []
        dck_dir = self.settings.train_dck_dir

        if not dck_dir.exists():
            logger.warning(f"Training DCK directory not found: {dck_dir}")
            return documents

        try:
            # Deduplicate: on Windows *.dck and *.DCK match the same files
            seen = set()
            dck_files = []
            for p in list(dck_dir.glob("*.dck")) + list(dck_dir.glob("*.DCK")):
                key = p.resolve()
                if key not in seen:
                    seen.add(key)
                    dck_files.append(p)

            for dck_path in dck_files:
                doc = self._create_single_dck_format_document(dck_path)
                if doc is not None:
                    documents.append(doc)

            logger.info(f"Created {len(documents)} DCK format documents")

        except Exception as e:
            logger.error(f"Error loading DCK files: {e}")

        return documents

    def _create_single_dck_format_document(self, dck_path: Path) -> Optional[Document]:
        """
        Create a format-learning document from a raw DCK file.

        Extracts content between '*** Units' and 'END' (the END before
        *!LINK_STYLE), inclusive of both markers.

        Args:
            dck_path: Path to DCK file.

        Returns:
            Document object or None if invalid.
        """
        try:
            # Read DCK file content
            full_content = dck_path.read_text(encoding="utf-8", errors="ignore")

            # Extract content between '*** Units' and 'END' (before *!LINK_STYLE)
            content = self._extract_units_to_end(full_content)
            if not content:
                # Fallback: use full content if markers not found
                logger.warning(f"Could not find '*** Units'...'END' markers in {dck_path.name}, using full content")
                content = full_content

            # Extract type IDs from content
            type_ids = self._extract_type_ids_from_dck(content)

            metadata = {
                "source": "dck_format",
                "filename": dck_path.name,
                "file_size": dck_path.stat().st_size,
                "type_ids": ",".join(type_ids[:10]),
            }

            return Document(text=content, metadata=metadata)

        except Exception as e:
            logger.warning(f"Error creating DCK format document {dck_path}: {e}")
            return None

    def _extract_units_to_end(self, content: str) -> Optional[str]:
        """
        Extract content between '*** Units' and 'END' (before *!LINK_STYLE).

        Includes both the '*** Units' line and the 'END' line.

        Args:
            content: Full DCK file content.

        Returns:
            Extracted content string, or None if markers not found.
        """
        lines = content.splitlines(keepends=True)
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if start_idx is None and stripped.startswith("*** Units"):
                start_idx = i
            elif start_idx is not None and stripped == "END":
                end_idx = i
                break

        if start_idx is not None and end_idx is not None:
            return "".join(lines[start_idx:end_idx + 1])
        return None

    def _extract_type_ids_from_dck(self, content: str) -> List[str]:
        """
        Extract Type IDs from DCK content.

        Args:
            content: DCK file content.

        Returns:
            List of type IDs found.
        """
        import re

        type_ids = set()

        # Match patterns like "TYPE 15", "Type15", "type 15-3"
        patterns = [
            r"TYPE\s*(\d+)",
            r"Type\s*(\d+)",
            r"type\s*(\d+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            type_ids.update(matches)

        return sorted(list(type_ids))

    # === Index Creation Helper ===

    def _create_index(
        self, collection_name: str, documents: List[Document]
    ) -> VectorStoreIndex:
        """
        Create a vector index using ChromaDB.

        Args:
            collection_name: Name for the ChromaDB collection.
            documents: List of documents to index.

        Returns:
            Created VectorStoreIndex.
        """
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError as e:
            raise ImportError(
                "llama-index-vector-stores-chroma is required. "
                "Install with: pip install llama-index-vector-stores-chroma"
            ) from e

        # Delete existing collection if exists
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.debug(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist

        # Create new collection
        chroma_collection = self.chroma_client.create_collection(collection_name)

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index with sentence splitter for better chunking
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
            show_progress=True,
        )

        logger.info(f"Created index: {collection_name} with {len(documents)} documents")
        return index

    # === Utility Methods ===

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all indexes.

        Returns:
            Dictionary with index statistics.
        """
        stats = {}

        for collection_name in [
            self.COLLECTION_PARSED_DCK_SYSTEMS,
            self.COLLECTION_TMF_COMPONENTS,
            self.COLLECTION_DCK_SYSTEMS,
        ]:
            try:
                existing_collections = [c.name for c in self.chroma_client.list_collections()]
                if collection_name in existing_collections:
                    collection = self.chroma_client.get_collection(collection_name)
                    stats[collection_name] = {
                        "exists": True,
                        "count": collection.count(),
                    }
                else:
                    stats[collection_name] = {
                        "exists": False,
                        "count": 0,
                    }
            except Exception as e:
                stats[collection_name] = {
                    "exists": False,
                    "error": str(e),
                }

        return stats

    def delete_all_indexes(self) -> None:
        """
        Delete all indexes.

        Use with caution - this will remove all indexed data.
        """
        for collection_name in [
            self.COLLECTION_PARSED_DCK_SYSTEMS,
            self.COLLECTION_TMF_COMPONENTS,
            self.COLLECTION_DCK_SYSTEMS,
        ]:
            try:
                self.chroma_client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_name}: {e}")

        self._indexes.clear()
        logger.info("Deleted all RAG indexes")


# === Module-level convenience functions ===

_indexer_instance: Optional[RAGIndexer] = None


def get_rag_indexer() -> RAGIndexer:
    """
    Get the RAG indexer instance.

    Returns a singleton instance of RAGIndexer.

    Returns:
        RAGIndexer instance.
    """
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = RAGIndexer()
    return _indexer_instance


def build_all_rag_indexes(force_rebuild: bool = False) -> Dict[str, VectorStoreIndex]:
    """
    Build all RAG indexes.

    Convenience function to build all indexes.

    Args:
        force_rebuild: If True, rebuild all indexes.

    Returns:
        Dictionary mapping collection names to indexes.
    """
    indexer = get_rag_indexer()
    return indexer.build_all_indexes(force_rebuild)


def reset_indexer() -> None:
    """
    Reset the indexer instance.

    Clears the cached indexer instance.
    """
    global _indexer_instance
    _indexer_instance = None
    logger.info("Reset RAG indexer")
