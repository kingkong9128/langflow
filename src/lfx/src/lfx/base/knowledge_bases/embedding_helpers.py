"""Shared helpers for Knowledge Base / Memory Base retrieval components.

Extracted from KnowledgeBaseComponent so MemoryBaseComponent can reuse the
embedding-builder, metadata loader, and global-variable resolution code
without duplicating it. All helpers take ``user_id`` as a parameter instead
of reading ``self.user_id`` so they remain component-agnostic.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from cryptography.fernet import InvalidToken
from langflow.services.auth.utils import decrypt_api_key

from lfx.base.models.unified_models import (
    get_model_provider_variable_mapping,
    get_provider_all_variables,
)
from lfx.log.logger import logger
from lfx.services.deps import get_settings_service, get_variable_service, session_scope

_KNOWLEDGE_BASES_ROOT_PATH: Path | None = None


def get_knowledge_bases_root_path() -> Path:
    """Lazy-load the knowledge-bases root path from settings."""
    global _KNOWLEDGE_BASES_ROOT_PATH  # noqa: PLW0603
    if _KNOWLEDGE_BASES_ROOT_PATH is None:
        settings = get_settings_service().settings
        knowledge_directory = settings.knowledge_bases_dir
        if not knowledge_directory:
            msg = "Knowledge bases directory is not set in the settings."
            raise ValueError(msg)
        _KNOWLEDGE_BASES_ROOT_PATH = Path(knowledge_directory).expanduser()
    return _KNOWLEDGE_BASES_ROOT_PATH


def get_kb_metadata(kb_path: Path) -> dict:
    """Load and decrypt embedding metadata for a KB directory."""
    metadata: dict[str, Any] = {}
    metadata_file = kb_path / "embedding_metadata.json"
    if not metadata_file.exists():
        logger.warning(f"Embedding metadata file not found at {metadata_file}")
        return metadata

    try:
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {metadata_file}")
        return {}

    if "api_key" in metadata and metadata.get("api_key"):
        settings_service = get_settings_service()
        try:
            metadata["api_key"] = decrypt_api_key(metadata["api_key"], settings_service)
        except (InvalidToken, TypeError, ValueError) as e:
            logger.error(f"Could not decrypt API key. Please provide it manually. Error: {e}")
            metadata["api_key"] = None
    return metadata


def _coerce_user_uuid(user_id: str | uuid.UUID | None) -> uuid.UUID | None:
    if not user_id:
        return None
    return user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))


async def resolve_provider_variables(provider: str, user_id: str | uuid.UUID | None) -> dict[str, str]:
    """Resolve all global variables for a provider in the current async context."""
    result: dict[str, str] = {}
    provider_vars = get_provider_all_variables(provider)
    user_uuid = _coerce_user_uuid(user_id)
    if not provider_vars or not user_uuid:
        return result

    async with session_scope() as session:
        variable_service = get_variable_service()
        if variable_service is None:
            return result

        for var_info in provider_vars:
            var_key = var_info.get("variable_key")
            if not var_key:
                continue
            try:
                value = await variable_service.get_variable(
                    user_id=user_uuid,
                    name=var_key,
                    field="",
                    session=session,
                )
                if value and str(value).strip():
                    result[var_key] = str(value)
            except (ValueError, KeyError, AttributeError) as e:
                logger.debug(f"Variable service lookup failed for '{var_key}', falling back to environment: {e}")
                env_value = os.environ.get(var_key)
                if env_value and env_value.strip():
                    result[var_key] = env_value
    return result


async def resolve_api_key(provider: str, user_id: str | uuid.UUID | None) -> str | None:
    """Resolve the provider API key from global variables (fallback chain)."""
    provider_variable_map = get_model_provider_variable_mapping()
    variable_name = provider_variable_map.get(provider)
    user_uuid = _coerce_user_uuid(user_id)
    if not variable_name or not user_uuid:
        return None

    async with session_scope() as session:
        variable_service = get_variable_service()
        if variable_service is None:
            return None
        try:
            return await variable_service.get_variable(
                user_id=user_uuid,
                name=variable_name,
                field="",
                session=session,
            )
        except (ValueError, KeyError, AttributeError):
            return None


def build_embeddings(metadata: dict, *, api_key: str | None = None, provider_vars: dict | None = None):
    """Build an embedding model from KB metadata. Raises for unsupported providers."""
    provider = metadata.get("embedding_provider")
    model = metadata.get("embedding_model")
    chunk_size = metadata.get("chunk_size")

    if provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        if not api_key:
            msg = (
                "OpenAI API key is required. Provide it in the component's advanced settings"
                " or configure it globally."
            )
            raise ValueError(msg)
        openai_kwargs: dict = {"model": model, "api_key": api_key}
        if chunk_size is not None:
            openai_kwargs["chunk_size"] = chunk_size
        return OpenAIEmbeddings(**openai_kwargs)
    if provider == "HuggingFace":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model=model)
    if provider == "Cohere":
        from langchain_cohere import CohereEmbeddings

        if not api_key:
            msg = "Cohere API key is required when using Cohere provider"
            raise ValueError(msg)
        return CohereEmbeddings(model=model, cohere_api_key=api_key)
    if provider == "Google Generative AI":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not api_key:
            msg = (
                "Google API key is required. Provide it in the component's advanced settings"
                " or configure it globally."
            )
            raise ValueError(msg)
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    if provider == "Ollama":
        from langchain_ollama import OllamaEmbeddings

        all_vars = provider_vars or {}
        base_url = all_vars.get("OLLAMA_BASE_URL")
        kwargs: dict = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbeddings(**kwargs)
    if provider == "IBM WatsonX":
        from langchain_ibm import WatsonxEmbeddings

        all_vars = provider_vars or {}
        watsonx_apikey = api_key or all_vars.get("WATSONX_APIKEY")
        watsonx_project_id = all_vars.get("WATSONX_PROJECT_ID")
        watsonx_url = all_vars.get("WATSONX_URL")
        if not watsonx_apikey:
            msg = (
                "IBM WatsonX API key is required. Provide it in the component's advanced settings"
                " or configure it globally."
            )
            raise ValueError(msg)
        kwargs = {"model_id": model, "apikey": watsonx_apikey}
        if watsonx_project_id:
            kwargs["project_id"] = watsonx_project_id
        if watsonx_url:
            kwargs["url"] = watsonx_url
        return WatsonxEmbeddings(**kwargs)
    if provider == "Custom":
        msg = "Custom embedding models not yet supported"
        raise NotImplementedError(msg)
    msg = f"Embedding provider '{provider}' is not supported for retrieval."
    raise NotImplementedError(msg)
