"""Contract tests for the LLM and embedding abstraction layers.

Offline tests run everywhere and assert the guarantees the rest of the codebase
relies on: response normalisation, the JSON-mode guard, dimension validation,
and the fail-closed egress policy.

Tests marked ``live`` make real backend calls and are deselected by default:

    uv run pytest # offline only
    uv run pytest -m live # include live backend calls
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import config, embeddings, llm


# ── Egress policy ─────────────────────────────────────────────────────────────


class TestEgressPolicy:
    """The proxy-only policy must be enforced in code, not by convention."""

    @pytest.mark.parametrize("backend", ["openai"])
    def test_blocks_external_llm_backends(self, monkeypatch, backend):
        monkeypatch.setattr(config, "LLM_EGRESS_POLICY", "proxy_only")
        llm.reset_backend()
        with pytest.raises(PermissionError, match="Egress blocked"):
            llm.get_backend(backend)

    @pytest.mark.parametrize("backend", ["openai", "local"])
    def test_blocks_external_embedding_backends(self, monkeypatch, backend):
        monkeypatch.setattr(config, "LLM_EGRESS_POLICY", "proxy_only")
        embeddings.reset_backend()
        with pytest.raises(PermissionError, match="Egress blocked"):
            embeddings.get_backend(backend)

    def test_local_embeddings_blocked_before_model_download(self, monkeypatch):
        """The guard must fire before any weights are fetched."""
        monkeypatch.setattr(config, "LLM_EGRESS_POLICY", "proxy_only")
        embeddings.reset_backend()
        with pytest.raises(PermissionError):
            embeddings.get_backend("local")

    def test_open_policy_permits_external(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_EGRESS_POLICY", "open")
        # Reaches construction (and fails on the missing key) rather than
        # being blocked by policy.
        with pytest.raises(llm.LLMError, match="OPENAI_API_KEY"):
            monkeypatch.setattr(config, "OPENAI_API_KEY", "")
            llm.reset_backend()
            llm.get_backend("openai")

    def test_unknown_backend_rejected(self):
        with pytest.raises(llm.LLMError, match="Unknown LLM_BACKEND"):
            llm.get_backend("does-not-exist")


# ── JSON mode ─────────────────────────────────────────────────────────────────


class TestJsonModeGuard:
    """OpenAI-compatible APIs reject JSON mode unless 'json' appears in the
    messages. Catching this locally turns an opaque HTTP 500 into a clear error.
    """

    def test_raises_without_json_hint(self):
        with pytest.raises(llm.LLMError, match="JSON mode requires"):
            llm._assert_json_hint("You are terse.", "Say hello.")

    @pytest.mark.parametrize(
        "system,user",
        [
            ("Respond with valid JSON only.", "Say hello."),
            ("You are terse.", "Reply as a json object."),
            ("Return JSON.", "anything"), # case-insensitive
        ],
    )
    def test_accepts_hint_in_either_message(self, system, user):
        llm._assert_json_hint(system, user) # must not raise


class TestJsonParsing:
    """Native JSON mode makes fences unnecessary, but providers vary."""

    def test_plain_json(self):
        assert llm.parse_json_content('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert llm.parse_json_content('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bare_fence(self):
        assert llm.parse_json_content('```\n{"a": 1}\n```') == {"a": 1}

    def test_json_embedded_in_prose(self):
        assert llm.parse_json_content('Here you go: {"a": 1} hope that helps') == {"a": 1}

    def test_raises_when_no_object_present(self):
        with pytest.raises(llm.LLMError, match="No JSON object"):
            llm.parse_json_content("no json here at all")

    def test_raises_on_malformed_object(self):
        with pytest.raises(llm.LLMError, match="Malformed JSON"):
            llm.parse_json_content('{"a": 1,,,}')


# ── Response normalisation ────────────────────────────────────────────────────


class TestResponseNormalisation:
    """Both backends must produce identical shapes so monitoring is uniform."""

    def test_pulse_flat_body_is_normalised(self):
        backend = object.__new__(llm.PulseProxyBackend)
        body = {
            "content": "hello",
            "model": "gpt-4o",
            "latency_ms": 1234.5,
            "cached": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = backend._normalise(body, elapsed_ms=999.0)
        assert resp.content == "hello"
        assert (resp.prompt_tokens, resp.completion_tokens, resp.total_tokens) == (10, 5, 15)
        assert resp.cached is True
        assert resp.latency_ms == 1234.5 # proxy timing preferred over local
        assert resp.backend == "pulse"

    def test_pulse_missing_usage_defaults_to_zero(self):
        backend = object.__new__(llm.PulseProxyBackend)
        resp = backend._normalise({"content": "x"}, elapsed_ms=50.0)
        assert resp.total_tokens == 0
        assert resp.latency_ms == 50.0

    def test_cost_estimate_uses_configured_rates(self, monkeypatch):
        monkeypatch.setattr(config, "COST_PROMPT_PER_1M", 2.50)
        monkeypatch.setattr(config, "COST_COMPLETION_PER_1M", 10.00)
        resp = llm.LLMResponse(
            content="x", prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        assert resp.estimated_cost_usd == pytest.approx(12.50)

    def test_zero_tokens_cost_nothing(self):
        assert llm.LLMResponse(content="x").estimated_cost_usd == 0.0


# ── Embeddings ────────────────────────────────────────────────────────────────


class TestEmbeddingContract:
    def test_dimension_mismatch_fails_fast(self, monkeypatch):
        """A mismatch must surface immediately, not as a pgvector insert error."""
        monkeypatch.setattr(config, "PULSE_PROXY_URL", "http://mock-proxy-url")
        monkeypatch.setattr(config, "EMBED_DIM", 1536)
        embeddings._dim_verified = False
        with pytest.raises(embeddings.EmbeddingError, match="dimension mismatch"):
            embeddings._verify_dim([[0.0] * 384])

    def test_matching_dimension_passes(self, monkeypatch):
        monkeypatch.setattr(config, "EMBED_DIM", 384)
        embeddings._dim_verified = False
        embeddings._verify_dim([[0.0] * 384]) # must not raise

    def test_pulse_extracts_singular_and_plural_shapes(self):
        extract = embeddings.PulseEmbedBackend._extract
        assert extract({"embedding": [1.0, 2.0]}) == [[1.0, 2.0]]
        assert extract({"embeddings": [[1.0], [2.0]]}) == [[1.0], [2.0]]

    def test_pulse_rejects_unexpected_shape(self):
        with pytest.raises(embeddings.EmbeddingError, match="Unexpected"):
            embeddings.PulseEmbedBackend._extract({"data": []})

    def test_empty_input_short_circuits(self):
        """Must not construct a backend or make a call for an empty list."""
        assert embeddings.embed_texts([]) == []

    def test_batching_covers_all_inputs_exactly_once(self):
        texts = [f"t{i}" for i in range(125)]
        batches = embeddings._batches(texts, 50)
        assert [len(b) for b in batches] == [50, 50, 25]
        assert [t for b in batches for t in b] == texts

    def test_unknown_backend_rejected(self):
        with pytest.raises(embeddings.EmbeddingError, match="Unknown EMBED_BACKEND"):
            embeddings.get_backend("does-not-exist")


# ── Live backend tests ────────────────────────────────────────────────────────


@pytest.mark.live
class TestLiveBackends:
    """Real calls against whichever backend is configured in .env."""

    def test_chat_returns_content_and_tokens(self):
        resp = llm.chat("You are terse.", "Say exactly: hello")
        assert resp.content.strip()
        assert resp.total_tokens > 0
        assert resp.latency_ms > 0

    def test_json_mode_returns_parsable_object(self):
        parsed, resp = llm.chat_json(
            "You are a helpful assistant. Respond with valid JSON only.",
            'Return a json object with keys "ok" (boolean) and "sum" (2+2).',
        )
        assert isinstance(parsed, dict)
        assert resp.total_tokens > 0

    def test_embeddings_match_configured_dimension(self):
        vectors = embeddings.embed_texts(["nutella spread", "low sugar cereal"])
        assert len(vectors) == 2
        assert all(len(v) == config.EMBED_DIM for v in vectors)

    def test_probe_reports_backend(self):
        stats = embeddings.probe()
        assert stats.count == 1
        assert stats.backend == config.EMBED_BACKEND


class TestOpenAIResponseNormalisation:
    """The Groq/OpenAI path is the reviewer's default and is never exercised here.

    Its response is absorbed by ``_normalise`` before anything downstream sees
    it, so these tests pin the shapes a compatible provider is allowed to
    return. Usage fields are already defaulted; ``choices`` is the one field
    that cannot be, and an empty list used to raise ``IndexError`` - which
    ``rag.py`` does not catch, turning a filtered completion into a 500 that
    discarded the retrieved products.
    """

    @staticmethod
    def _backend():
        backend = object.__new__(llm.OpenAICompatibleBackend)
        backend.name = "openai"
        return backend

    @staticmethod
    def _response(*, choices, usage=None, model="llama-3.3-70b-versatile"):
        return SimpleNamespace(choices=choices, usage=usage, model=model, id="resp_1")

    @staticmethod
    def _choice(content):
        return SimpleNamespace(message=SimpleNamespace(content=content))

    def test_normalises_a_well_formed_response(self):
        resp = self._response(
            choices=[self._choice("hello")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        out = self._backend()._normalise(resp, 12.5)
        assert out.content == "hello"
        assert (out.prompt_tokens, out.completion_tokens, out.total_tokens) == (10, 5, 15)
        assert out.model == "llama-3.3-70b-versatile"
        assert out.backend == "openai"

    def test_empty_choices_raises_llm_error_not_index_error(self):
        with pytest.raises(llm.LLMError) as excinfo:
            self._backend()._normalise(self._response(choices=[]), 1.0)
        assert "no completion choices" in str(excinfo.value)

    def test_empty_choices_error_is_not_retried(self):
        """An empty completion is a verdict, not a transient fault."""
        from app import retry

        assert not retry.is_retryable(llm.LLMError("no completion choices"))

    def test_missing_usage_defaults_to_zero_rather_than_failing(self):
        """Some providers omit usage entirely; cost accounting must not crash."""
        out = self._backend()._normalise(
            self._response(choices=[self._choice("hi")], usage=None), 1.0
        )
        assert (out.prompt_tokens, out.completion_tokens, out.total_tokens) == (0, 0, 0)
        assert out.estimated_cost_usd == 0.0

    def test_null_content_becomes_empty_string(self):
        """A tool-call or filtered message can carry content=None."""
        out = self._backend()._normalise(
            self._response(choices=[self._choice(None)]), 1.0
        )
        assert out.content == ""