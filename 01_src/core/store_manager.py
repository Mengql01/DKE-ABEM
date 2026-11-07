"""
LangGraph Store Manager for External Data Storage.

Provides utilities to store large objects in LangGraph Store
instead of graph state, reducing checkpoint size.

This module helps reduce memory usage by:
1. Storing retrieved documents externally (only IDs in state)
2. Storing validation file contents externally (only refs in state)
3. Archiving old message history (only recent N messages in state)

Usage:
    from src.core.store_manager import StoreManager

    # In agent node function
    store = config.get("store")
    if store:
        store_mgr = StoreManager(store, thread_id)
        doc_ids = store_mgr.store_documents(documents)
        # Later retrieve
        documents = store_mgr.get_documents(doc_ids)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StoreManager:
    """
    Manager for storing large objects outside graph state.

    This class provides a clean interface to LangGraph Store,
    helping reduce checkpoint size by externalizing large data.
    """

    def __init__(self, store: Any, thread_id: str):
        """
        Initialize store manager.

        Args:
            store: LangGraph Store instance (from config.store)
            thread_id: Current thread ID for namespacing
        """
        self.store = store
        self.thread_id = thread_id
        self._logger = logger

    def store_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Store retrieved documents and return IDs.

        Args:
            documents: List of document dicts with content

        Returns:
            List of document IDs that can be used to retrieve later

        Example:
            doc_ids = store_mgr.store_documents([
                {"content": "...", "metadata": {...}},
                {"content": "...", "metadata": {...}}
            ])
            # Store doc_ids in state instead of full documents
        """
        if not documents:
            return []

        doc_ids = []
        try:
            for idx, doc in enumerate(documents):
                doc_id = f"{self.thread_id}_doc_{idx}"
                namespace = ("retrieved_docs", self.thread_id)
                self.store.put(namespace, doc_id, doc)
                doc_ids.append(doc_id)
                self._logger.debug(f"Stored document {doc_id} in Store")
        except Exception as e:
            self._logger.warning(f"Failed to store documents in Store: {e}")
            # Return empty list to signal fallback to state storage
            return []

        return doc_ids

    def get_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by IDs.

        Args:
            doc_ids: List of document IDs from store_documents()

        Returns:
            List of document dicts
        """
        if not doc_ids:
            return []

        documents = []
        try:
            for doc_id in doc_ids:
                namespace = ("retrieved_docs", self.thread_id)
                item = self.store.get(namespace, doc_id)
                if item:
                    documents.append(item.value)
                else:
                    self._logger.warning(f"Document {doc_id} not found in Store")
        except Exception as e:
            self._logger.warning(f"Failed to retrieve documents from Store: {e}")
            return []

        return documents

    def store_validation_files(
        self,
        version_tag: str,
        files: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Store validation files (dck/log/lst content).

        Args:
            version_tag: Version identifier (e.g., "V1_20260224_060900")
            files: Dict with keys "dck_content", "log_content", "lst_content"

        Returns:
            Reference dict with store_namespace and store_key

        Example:
            files = {
                "dck_content": "VERSION 18\n...",
                "log_content": "TRNSYS log...",
                "lst_content": "TRNSYS listing..."
            }
            ref = store_mgr.store_validation_files("V1_20260508_120000", files)
            # Store ref in state instead of full file contents
        """
        try:
            namespace = ("validation_files", self.thread_id)
            key = version_tag
            self.store.put(namespace, key, files)
            self._logger.debug(f"Stored validation files {version_tag} in Store")

            return {
                "version_tag": version_tag,
                "store_namespace": f"validation_files/{self.thread_id}",
                "store_key": key
            }
        except Exception as e:
            self._logger.warning(f"Failed to store validation files in Store: {e}")
            # Return empty dict to signal fallback to state storage
            return {}

    def get_validation_files(self, version_tag: str) -> Optional[Dict[str, str]]:
        """
        Retrieve validation files by version tag.

        Args:
            version_tag: Version identifier

        Returns:
            Dict with "dck_content", "log_content", "lst_content" or None
        """
        try:
            namespace = ("validation_files", self.thread_id)
            item = self.store.get(namespace, version_tag)
            return item.value if item else None
        except Exception as e:
            self._logger.warning(f"Failed to retrieve validation files from Store: {e}")
            return None

    def store_similar_dcks(self, dcks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Store similar DCK examples and return references.

        Args:
            dcks: List of similar DCK dicts with full content

        Returns:
            List of reference dicts with dck_id and store_key
        """
        if not dcks:
            return []

        refs = []
        try:
            for idx, dck in enumerate(dcks):
                dck_id = f"{self.thread_id}_dck_{idx}"
                namespace = ("similar_dcks", self.thread_id)
                self.store.put(namespace, dck_id, dck)
                refs.append({
                    "dck_id": dck_id,
                    "store_key": dck_id
                })
                self._logger.debug(f"Stored similar DCK {dck_id} in Store")
        except Exception as e:
            self._logger.warning(f"Failed to store similar DCKs in Store: {e}")
            return []

        return refs

    def get_similar_dcks(self, refs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Retrieve similar DCKs by references.

        Args:
            refs: List of reference dicts from store_similar_dcks()

        Returns:
            List of DCK dicts
        """
        if not refs:
            return []

        dcks = []
        try:
            for ref in refs:
                dck_id = ref.get("dck_id")
                if not dck_id:
                    continue
                namespace = ("similar_dcks", self.thread_id)
                item = self.store.get(namespace, dck_id)
                if item:
                    dcks.append(item.value)
        except Exception as e:
            self._logger.warning(f"Failed to retrieve similar DCKs from Store: {e}")
            return []

        return dcks

    def store_message_history(self, messages: List[Any]) -> str:
        """
        Store old messages and return reference key.

        Args:
            messages: List of messages to archive

        Returns:
            Store key for retrieval (empty string if failed)
        """
        if not messages:
            return ""

        try:
            namespace = ("message_history", self.thread_id)
            key = f"history_{len(messages)}"
            self.store.put(namespace, key, {"messages": messages})
            self._logger.debug(f"Stored {len(messages)} messages in Store")
            return key
        except Exception as e:
            self._logger.warning(f"Failed to store message history in Store: {e}")
            return ""

    def get_message_history(self, key: str) -> List[Any]:
        """
        Retrieve archived messages.

        Args:
            key: Store key from store_message_history()

        Returns:
            List of messages (empty if not found or failed)
        """
        if not key:
            return []

        try:
            namespace = ("message_history", self.thread_id)
            item = self.store.get(namespace, key)
            return item.value.get("messages", []) if item else []
        except Exception as e:
            self._logger.warning(f"Failed to retrieve message history from Store: {e}")
            return []


def trim_messages_to_store(
    messages: List[Any],
    store: Any,
    thread_id: str,
    keep_recent: int = 10
) -> tuple[List[Any], str]:
    """
    Trim messages, keeping only recent N in state and archiving old ones to Store.

    This is a utility function that can be called from any agent node
    to reduce message list size in state.

    Args:
        messages: Current message list from state
        store: LangGraph Store instance (from config.store)
        thread_id: Current thread ID
        keep_recent: Number of recent messages to keep in state

    Returns:
        Tuple of (trimmed_messages, history_ref)
        - trimmed_messages: Recent N messages to keep in state
        - history_ref: Store key for archived messages (empty if no archiving)

    Example:
        # In agent node
        messages = state.get("messages", [])
        store = config.get("store")
        thread_id = config.get("configurable", {}).get("thread_id", state["task_id"])

        if store and len(messages) > 10:
            trimmed, ref = trim_messages_to_store(messages, store, thread_id, keep_recent=10)
            return {"messages": trimmed, "message_history_ref": ref}
    """
    if len(messages) <= keep_recent:
        return messages, ""  # No trimming needed

    if not store:
        return messages, ""  # Store not available, keep all

    try:
        store_mgr = StoreManager(store, thread_id)

        # Archive old messages
        old_messages = messages[:-keep_recent]
        history_key = store_mgr.store_message_history(old_messages)

        if history_key:
            # Successfully archived, return trimmed list
            return messages[-keep_recent:], history_key
        else:
            # Failed to archive, keep all
            return messages, ""
    except Exception as e:
        logger.warning(f"Failed to trim messages: {e}")
        return messages, ""
