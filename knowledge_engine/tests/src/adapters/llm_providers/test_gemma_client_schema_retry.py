"""complete_structured's schema-parse retry (attempt 2/2): without
``on_schema_retry`` it resends the identical payload and hopes Gemma
succeeds by luck; with it, callers holding the original data (e.g. a
REDUCE-synthesis fact list) can hand back a smaller payload instead —
verified here by asserting the SECOND HTTP call actually used the reduced
prompt, not the original one."""

from __future__ import annotations

import asyncio
import copy
import json

from pydantic import BaseModel

from knowledge_engine.src.adapters.llm_providers.gemma_client import GemmaCloudClient


class _Schema(BaseModel):
    executive_summary: str
    key_takeaways: list[str]


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _gemma_payload(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }


class _FakeHttpClient:
    """Records every posted payload; returns queued responses in order."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.posted_payloads: list[dict] = []

    async def post(self, url: str, *, json: dict, headers: dict) -> _FakeResponse:
        # complete_structured mutates its outer `payload` dict in place on
        # retry rather than rebuilding it — snapshot, don't keep a reference.
        self.posted_payloads.append(copy.deepcopy(json))
        return self._responses.pop(0)


def test_schema_retry_uses_reduced_payload_from_callback() -> None:
    malformed = _gemma_payload(
        json.dumps(
            {"article_passport": {"executive_summary": "x", "key_takeaways": []}}
        )
    )
    valid = _gemma_payload(
        json.dumps({"executive_summary": "ok", "key_takeaways": ["a"]})
    )
    fake_client = _FakeHttpClient(
        [_FakeResponse(200, malformed), _FakeResponse(200, valid)]
    )

    def _shrink() -> tuple[str, str]:
        return "system (reduced)", "prompt with fewer facts"

    client = GemmaCloudClient(api_key="test-key")
    result = asyncio.run(
        client.complete_structured(
            "system (full)",
            "prompt with many facts",
            _Schema,
            client=fake_client,
            use_token_budget=False,
            on_schema_retry=_shrink,
        )
    )

    assert isinstance(result, _Schema)
    assert result.executive_summary == "ok"
    assert len(fake_client.posted_payloads) == 2
    first_user = fake_client.posted_payloads[0]["messages"][1]["content"]
    second_user = fake_client.posted_payloads[1]["messages"][1]["content"]
    assert "prompt with many facts" in first_user
    assert "prompt with fewer facts" in second_user
    assert "prompt with many facts" not in second_user


def test_schema_retry_without_callback_resends_same_payload() -> None:
    """No on_schema_retry: behavior unchanged — attempt 2 resends attempt 1's

    exact payload (existing "maybe Gemma gets it right this time" retry)."""
    malformed = _gemma_payload(
        json.dumps(
            {"article_passport": {"executive_summary": "x", "key_takeaways": []}}
        )
    )
    valid = _gemma_payload(
        json.dumps({"executive_summary": "ok", "key_takeaways": ["a"]})
    )
    fake_client = _FakeHttpClient(
        [_FakeResponse(200, malformed), _FakeResponse(200, valid)]
    )

    client = GemmaCloudClient(api_key="test-key")
    result = asyncio.run(
        client.complete_structured(
            "system",
            "same prompt",
            _Schema,
            client=fake_client,
            use_token_budget=False,
        )
    )

    assert isinstance(result, _Schema)
    assert len(fake_client.posted_payloads) == 2
    assert fake_client.posted_payloads[0] == fake_client.posted_payloads[1]
