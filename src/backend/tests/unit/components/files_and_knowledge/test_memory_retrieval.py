"""Unit tests for MemoryBaseComponent.

Covers the core data-separation invariants:
- Missing session_id raises.
- Missing invoker user_id raises.
- Selecting an MB not attached to the current flow raises.
- similarity_search is invoked with a session_id + user_id filter.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lfx.components.files_and_knowledge.memory_retrieval import MemoryBaseComponent


def _make_component(
    *,
    flow_id: uuid.UUID | None,
    session_id: str | None,
    invoker_user_id: uuid.UUID | None,
    selected: str | None = "mb-one",
) -> MemoryBaseComponent:
    component = MemoryBaseComponent()
    # Graph context
    component._vertex = MagicMock()
    component._vertex.graph = SimpleNamespace(
        flow_id=str(flow_id) if flow_id else None,
        session_id=session_id,
        user_id=str(invoker_user_id) if invoker_user_id else None,
        flow_name="test-flow",
        context={},
    )
    component._user_id = str(invoker_user_id) if invoker_user_id else None
    # Inputs
    component.memory_base = selected
    component.search_query = "hello"
    component.top_k = 5
    component.include_metadata = True
    return component


class TestMemoryBaseRetrievalInvariants:
    async def test_missing_session_id_raises(self):
        component = _make_component(
            flow_id=uuid.uuid4(), session_id=None, invoker_user_id=uuid.uuid4()
        )
        with pytest.raises(ValueError, match="session_id is required"):
            await component.retrieve_data()

    async def test_missing_invoker_user_id_raises(self):
        component = _make_component(
            flow_id=uuid.uuid4(), session_id="s1", invoker_user_id=None
        )
        with pytest.raises(ValueError, match="authenticated user_id"):
            await component.retrieve_data()

    async def test_missing_flow_id_raises(self):
        component = _make_component(
            flow_id=None, session_id="s1", invoker_user_id=uuid.uuid4()
        )
        with pytest.raises(ValueError, match="flow_id"):
            await component.retrieve_data()

    async def test_mb_not_attached_to_flow_raises(self):
        component = _make_component(
            flow_id=uuid.uuid4(), session_id="s1", invoker_user_id=uuid.uuid4()
        )

        # DB returns nothing for the (name, flow_id) lookup
        mock_db = MagicMock()
        exec_result = MagicMock()
        exec_result.first.return_value = None
        mock_db.exec = AsyncMock(return_value=exec_result)

        class _Scope:
            async def __aenter__(self):
                return mock_db

            async def __aexit__(self, *a):
                return False

        with patch(
            "lfx.components.files_and_knowledge.memory_retrieval.session_scope",
            return_value=_Scope(),
        ):
            with pytest.raises(ValueError, match="not attached to this flow"):
                await component.retrieve_data()

    async def test_similarity_search_uses_session_and_user_filter(self):
        flow_id = uuid.uuid4()
        invoker_user_id = uuid.uuid4()
        owner_user_id = uuid.uuid4()
        component = _make_component(
            flow_id=flow_id, session_id="s1", invoker_user_id=invoker_user_id
        )

        mb_row = SimpleNamespace(
            id=uuid.uuid4(),
            name="mb-one",
            flow_id=flow_id,
            user_id=owner_user_id,
            kb_name="mb_one_abcd1234",
        )
        owner = SimpleNamespace(id=owner_user_id, username="alice")

        # DB returns mb_row on first exec; get_user_by_id returns the owner.
        mock_db = MagicMock()
        exec_result = MagicMock()
        exec_result.first.return_value = mb_row
        mock_db.exec = AsyncMock(return_value=exec_result)

        class _Scope:
            async def __aenter__(self):
                return mock_db

            async def __aexit__(self, *a):
                return False

        fake_chroma = MagicMock()
        fake_chroma.similarity_search_with_score.return_value = []

        with (
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.session_scope",
                return_value=_Scope(),
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.get_user_by_id",
                new=AsyncMock(return_value=owner),
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.get_knowledge_bases_root_path",
                return_value=__import__("pathlib").Path("/tmp"),
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.get_kb_metadata",
                return_value={"embedding_provider": "OpenAI", "embedding_model": "x", "api_key": "k"},
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.resolve_api_key",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.resolve_provider_variables",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.build_embeddings",
                return_value=MagicMock(),
            ),
            patch(
                "lfx.components.files_and_knowledge.memory_retrieval.Chroma",
                return_value=fake_chroma,
            ),
        ):
            await component.retrieve_data()

        fake_chroma.similarity_search_with_score.assert_called_once()
        call_kwargs = fake_chroma.similarity_search_with_score.call_args.kwargs
        assert call_kwargs["k"] == 5
        assert call_kwargs["filter"] == {
            "$and": [
                {"session_id": "s1"},
                {"user_id": str(invoker_user_id)},
            ]
        }
