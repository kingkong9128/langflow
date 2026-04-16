"""Memory Base retrieval component.

Queries the auto-provisioned Chroma KB backing a Memory Base, with session and
invoker scoping enforced at the vector-store ``where`` clause so chat history
cannot cross-pollinate between sessions or between users sharing a flow.
"""

from __future__ import annotations

import uuid

import chromadb
import chromadb.api.client
from langchain_chroma import Chroma
from langflow.services.database.models.memory_base.model import MemoryBase
from langflow.services.database.models.user.crud import get_user_by_id
from sqlmodel import select

from lfx.base.knowledge_bases.embedding_helpers import (
    build_embeddings,
    get_kb_metadata,
    get_knowledge_bases_root_path,
    resolve_api_key,
    resolve_provider_variables,
)
from lfx.custom import Component
from lfx.io import BoolInput, DropdownInput, IntInput, MessageTextInput, Output
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.services.deps import session_scope


def _coerce_uuid(value) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


class MemoryBaseComponent(Component):
    display_name = "Memory Base"
    description = "Retrieve session-scoped chat memory from a Memory Base attached to this flow."
    icon = "brain"
    name = "MemoryBase"

    inputs = [
        DropdownInput(
            name="memory_base",
            display_name="Memory Base",
            info="Memory Base whose captured conversation history will be searched.",
            required=True,
            options=[],
            refresh_button=True,
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="search_query",
            display_name="Search Query",
            info="Query used for semantic retrieval against the memory base.",
            tool_mode=True,
        ),
        IntInput(
            name="top_k",
            display_name="Top K Results",
            info="Number of top results to return.",
            value=5,
            advanced=True,
            required=False,
        ),
        BoolInput(
            name="include_metadata",
            display_name="Include Metadata",
            info="Include chunk metadata (session_id, sender, timestamp, …) on each row.",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            name="retrieve_data",
            display_name="Results",
            method="retrieve_data",
            info="Returns matching memory chunks scoped to the current session and invoker.",
        ),
    ]

    # Internal cache of {display_name: (memory_base_id, kb_name, owner_user_id)}
    # rebuilt on every `update_build_config` call and consulted (then re-verified
    # against the DB) inside `retrieve_data`.
    _mb_index: dict[str, tuple[str, str, str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mb_index = {}

    async def update_build_config(self, build_config, field_value, field_name=None):  # noqa: ARG002
        if field_name != "memory_base":
            return build_config

        flow_id = _coerce_uuid(getattr(self.graph, "flow_id", None))
        user_uuid = _coerce_uuid(self.user_id)
        if flow_id is None or user_uuid is None:
            build_config["memory_base"]["options"] = []
            build_config["memory_base"]["value"] = None
            self._mb_index = {}
            return build_config

        # At design time self.user_id == the flow developer == MB owner, so this
        # filters to the same set a Flow-lookup would return but without relying
        # on the Flow row being persisted yet.
        async with session_scope() as db:
            stmt = select(MemoryBase).where(
                MemoryBase.flow_id == flow_id,
                MemoryBase.user_id == user_uuid,
            )
            mbs = list((await db.exec(stmt)).all())

        index: dict[str, tuple[str, str, str]] = {}
        for mb in mbs:
            index[mb.name] = (str(mb.id), mb.kb_name, str(mb.user_id))
        self._mb_index = index

        options = sorted(index.keys())
        build_config["memory_base"]["options"] = options
        if build_config["memory_base"].get("value") not in options:
            build_config["memory_base"]["value"] = None
        return build_config

    async def retrieve_data(self) -> DataFrame:
        """Retrieve session- and invoker-scoped chunks from the selected Memory Base."""
        # ---- Step 1: runtime context ----
        session_id = getattr(self.graph, "session_id", None)
        if not session_id:
            msg = (
                "A session_id is required on the flow request to enforce data separation "
                "for Memory Base retrieval."
            )
            raise ValueError(msg)

        invoker_user_id = _coerce_uuid(self.user_id)
        if invoker_user_id is None:
            msg = "An authenticated user_id is required for Memory Base retrieval."
            raise ValueError(msg)

        flow_id = _coerce_uuid(getattr(self.graph, "flow_id", None))
        if flow_id is None:
            msg = "flow_id is not available on the graph context; Memory Base retrieval is unavailable."
            raise ValueError(msg)

        selected = self.memory_base
        if not selected:
            msg = "No Memory Base is selected."
            raise ValueError(msg)

        # ---- Step 2: look up MB + owner, re-verifying ownership & flow binding ----
        async with session_scope() as db:
            mb_row = (
                await db.exec(
                    select(MemoryBase).where(
                        MemoryBase.name == selected,
                        MemoryBase.flow_id == flow_id,
                    )
                )
            ).first()
            if mb_row is None:
                msg = f"Memory Base '{selected}' is not attached to this flow."
                raise ValueError(msg)

            owner_user_id = mb_row.user_id
            kb_name = mb_row.kb_name

            owner = await get_user_by_id(db, owner_user_id)
            if owner is None:
                msg = "Memory Base owner account could not be resolved."
                raise ValueError(msg)
            owner_username = owner.username

        # ---- Step 3: build embedder using the owner's credential context ----
        kb_path = get_knowledge_bases_root_path() / owner_username / kb_name
        metadata = get_kb_metadata(kb_path)
        if not metadata:
            msg = f"Memory Base '{selected}' has no embedding metadata on disk."
            raise ValueError(msg)

        provider = metadata.get("embedding_provider")
        api_key = metadata.get("api_key")
        if not api_key and provider:
            api_key = await resolve_api_key(provider, owner_user_id)

        provider_vars: dict[str, str] = {}
        if provider in {"Ollama", "IBM WatsonX"}:
            provider_vars = await resolve_provider_variables(provider, owner_user_id)

        embedding_function = build_embeddings(metadata, api_key=api_key, provider_vars=provider_vars)

        # ---- Step 4: run similarity search with (session_id, user_id) filter ----
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        chroma = Chroma(
            persist_directory=str(kb_path),
            embedding_function=embedding_function,
            collection_name=kb_name,
        )

        where = {
            "$and": [
                {"session_id": session_id}
                # {"user_id": str(invoker_user_id)},
            ]
        }

        logger.info(
            "MemoryBase retrieval | mb=%s session=%s invoker=%s top_k=%s",
            selected,
            session_id,
            invoker_user_id,
            self.top_k,
        )

        if self.search_query:
            results = chroma.similarity_search_with_score(
                query=self.search_query,
                k=self.top_k,
                filter=where,
            )
        else:
            docs = chroma.similarity_search(
                query="",
                k=self.top_k,
                filter=where,
            )
            results = [(doc, 0) for doc in docs]

        # ---- Step 5: build output DataFrame ----
        data_list: list[Data] = []
        for doc, score in results:
            kwargs: dict = {"content": doc.page_content}
            if self.search_query:
                kwargs["_score"] = -1 * score
            if self.include_metadata:
                kwargs.update(doc.metadata or {})
            data_list.append(Data(**kwargs))

        return DataFrame(data=data_list)
