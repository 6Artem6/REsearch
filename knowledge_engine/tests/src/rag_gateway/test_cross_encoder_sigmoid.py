"""Cross-Encoder logits must be sigmoid-calibrated once (not double-squashed)."""

from __future__ import annotations

import math

from knowledge_engine.src.rag_gateway import cross_encoder as ce


def test_sigmoid_unit_interval():
    assert abs(ce._sigmoid(0.0) - 0.5) < 1e-12
    assert ce._sigmoid(20.0) > 0.999999
    assert ce._sigmoid(-20.0) < 1e-8
    # numerically stable for large |x|
    assert 0.0 <= ce._sigmoid(1e9) <= 1.0
    assert 0.0 <= ce._sigmoid(-1e9) <= 1.0


def test_score_relevance_pairs_applies_sigmoid_to_raw_logits(monkeypatch):
    class FakeModel:
        def predict(self, pairs, batch_size=16, activation_fct=None):
            assert activation_fct is not None
            assert type(activation_fct).__name__ == "Identity"
            assert len(pairs) == 3
            return [0.0, 2.0, -2.0]

    monkeypatch.setattr(ce, "_load_cross_encoder", lambda: FakeModel())
    monkeypatch.setattr(ce, "_touch_ce_use", lambda: None)
    monkeypatch.setattr(ce, "_release_torch_cache", lambda: None)
    scores = ce.score_relevance_pairs("criterion text", ["a" * 24, "b" * 24, "c" * 24])
    assert len(scores) == 3
    assert abs(scores[0] - 0.5) < 1e-9
    assert abs(scores[1] - (1.0 / (1.0 + math.exp(-2.0)))) < 1e-9
    assert abs(scores[2] - (1.0 / (1.0 + math.exp(2.0)))) < 1e-9
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Double-sigmoid would land near 0.73, not 0.88.
    assert scores[1] > 0.85
