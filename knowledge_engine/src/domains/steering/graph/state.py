"""SteeringState — состояние изолированного LangGraph-микрографа «Штурвала».

Этап 3 мастер-плана, план/статус этапов — docs/STEERING_AND_TOPIC_QNA_ROADMAP.md.
Add-only: параллельно ``TutorGraphState``
(``src/node_deep_dive/graph/state.py``), никак с ним не пересекается и его
не импортирует.
"""

from __future__ import annotations

from typing import TypedDict

from knowledge_engine.src.domains.steering.steering_contracts import (
    SurfaceDigestResponse,
    TaxonomyDiscoveryResponse,
)


class SteeringState(TypedDict):
    target_goal: str
    work_job_id: str
    # Отладка "утечка Consensus/arXiv/S2 в Practical": source_policy раньше
    # нигде не доходил от SteeringCurriculumGenerateInput до
    # TaxonomyService/Light Discovery — оба всегда искали без ограничения
    # policy. Читается через .get(..., "hybrid") в builder.py, поэтому
    # добавление поля не ломает код, который его ещё не передаёт.
    source_policy: str
    taxonomy: TaxonomyDiscoveryResponse | None
    approved_gate1_urls: list[str]
    surface_digests: SurfaceDigestResponse | None
    final_approved_urls: list[str]


__all__ = ["SteeringState"]
