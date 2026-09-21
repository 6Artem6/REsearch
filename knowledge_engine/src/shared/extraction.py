"""Knowledge Triangulation — scope tags for extraction → tutor packing.

5-category taxonomy (CONCEPT/MECHANIC/PRACTICE/EDGE_CASE/ANTI_PATTERN),
promoted from the original 3-category taxonomy (PRINCIPLE/MECHANIC/
INSTANCE). The pre-migration 3-category implementation is preserved
byte-for-byte as ``extraction_3_scopes.py`` for rollback — see
``dialog_atoms_rag_3_scopes.py`` / ``pedagogical_reranker_3_scopes.py`` /
``blog_spatial_summarizer_3_scopes.py`` for its matching sibling snapshots.
Rollback is atomic: restore all four ``*_3_scopes.py`` files back over their
primary names together, since they were snapshotted as one consistent set
and their internal imports reference the (then-current) primary module
names, not each other's ``_3_scopes`` names.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

# Legacy 3-category names (PRINCIPLE / INSTANCE) stay recognised here so tags
# already persisted in stored key_takeaways keep parsing — coerce_scope_type()
# maps them to CONCEPT / PRACTICE.
_SCOPE_NAMES = "CONCEPT|MECHANIC|PRACTICE|EDGE_CASE|ANTI_PATTERN|PRINCIPLE|INSTANCE"

_SCOPE_TAG_RE = re.compile(
    rf"^\s*\[SCOPE:\s*({_SCOPE_NAMES})\s*\]\s*(.*)$",
    re.I | re.S,
)


class ScopeType(str, Enum):
    CONCEPT = "CONCEPT"
    MECHANIC = "MECHANIC"
    PRACTICE = "PRACTICE"
    EDGE_CASE = "EDGE_CASE"
    ANTI_PATTERN = "ANTI_PATTERN"


def coerce_scope_type(
    value: object, *, default: ScopeType = ScopeType.CONCEPT
) -> ScopeType:
    """Best-effort ScopeType parse (case / typos / legacy 3-category values)
    for defensive Pydantic. Legacy PRINCIPLE/INSTANCE values (atoms tagged
    before this migration, or a rollback to extraction_3_scopes.py and back)
    resolve to their direct semantic descendants — CONCEPT / PRACTICE."""
    if isinstance(value, ScopeType):
        return value
    if value is None:
        return default
    raw = str(value).strip()
    if not raw:
        return default
    # Strip optional [SCOPE: …] wrapper
    m = _SCOPE_TAG_RE.match(raw)
    if m:
        raw = m.group(1)
    upper = raw.upper().replace("-", "_").replace(" ", "_")
    for scope in ScopeType:
        if upper == scope.value or upper == scope.name:
            return scope
    # Legacy 3-category taxonomy (pre-migration atoms / rollback data).
    legacy = {
        "PRINCIPLE": ScopeType.CONCEPT,
        "PRINCIPLES": ScopeType.CONCEPT,
        "BASIS": ScopeType.CONCEPT,
        "FUNDAMENTAL": ScopeType.CONCEPT,
        "INSTANCE": ScopeType.PRACTICE,
        "INSTANCES": ScopeType.PRACTICE,
        "EVIDENCE": ScopeType.PRACTICE,
        "EXAMPLE": ScopeType.PRACTICE,
        "CASE": ScopeType.PRACTICE,
        "METRIC": ScopeType.PRACTICE,
    }
    if upper in legacy:
        return legacy[upper]
    # New-taxonomy aliases.
    aliases = {
        "ALGORITHM": ScopeType.MECHANIC,
        "MECHANISM": ScopeType.MECHANIC,
        "MECHANICS": ScopeType.MECHANIC,
        "ERROR": ScopeType.ANTI_PATTERN,
        "BAD_PRACTICE": ScopeType.ANTI_PATTERN,
        "ANTIPATTERN": ScopeType.ANTI_PATTERN,
        "LIMIT": ScopeType.EDGE_CASE,
        "CORNER_CASE": ScopeType.EDGE_CASE,
        "EDGECASE": ScopeType.EDGE_CASE,
    }
    if upper in aliases:
        return aliases[upper]
    for key, scope in {**legacy, **aliases}.items():
        if key in upper or upper in key:
            return scope
    return default


def merge_source_chunk_ids(*lists: Iterable[str] | None) -> list[str]:
    """Stable unique merge of chunk id lists (order: first occurrence wins)."""
    out: list[str] = []
    seen: set[str] = set()
    for lst in lists:
        for raw in lst or []:
            cid = str(raw or "").strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
    return out


class KnowledgeAtom(BaseModel):
    """Knowledge atom with generalization level (Knowledge Triangulation)."""

    scope: ScopeType = Field(
        ...,
        description=(
            "CONCEPT — fundamental principle/pattern (WHY/WHAT); "
            "MECHANIC — generalized algorithm/stage (HOW, normal path); "
            "PRACTICE — routine specific case, numbers, libraries, metrics; "
            "EDGE_CASE — boundary/degradation behavior, limits, races; "
            "ANTI_PATTERN — known mistake / explicit 'do not do this'"
        ),
    )
    statement: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Claim without URL; PRACTICE/EDGE_CASE may include numbers and library names",
    )
    context_quote: str | None = Field(
        default=None,
        max_length=800,
        description="Short supporting fragment from the source paragraph (optional)",
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of MAP windows / rag_chunks that mention this fact",
    )
    cluster_key: str = Field(
        default="general",
        max_length=64,
        description="Short entity or topic tag, e.g. 'cpylex', 'gil_lock'",
    )
    core_relevance_score: float | None = Field(
        default=None,
        description=(
            "EXPERIMENTAL (Two-Stage Fact Relevance & Anchor Filtering). "
            "Static relevance score against Article Core Anchor, in [0, 1]. "
            "None means not yet scored — existing consumers (grounding, "
            "entity_consensus_engine) MUST treat missing as unscored, never as 0.0."
        ),
    )
    """ RU: экспериментальный статический скор релевантности факта относительно
    сути статьи (см. anchor_relevance.py). Optional/дефолт None — Add-only,
    не ломает существующих потребителей, которые об этом поле не знают. """
    id: str | None = Field(
        default=None,
        description=(
            "Stable vector-store row id (KA_COL_ID), populated when the atom "
            "was reconstructed from a DB row (dialog_atoms_rag.py). None for "
            "atoms built directly from a fresh MAP/REDUCE LLM response."
        ),
    )
    """ RU: id строки в векторном хранилище; отсутствует для «свежих» атомов
    из MAP/REDUCE, до записи в БД. """
    article_id: str | None = Field(
        default=None,
        description=(
            "Owning document/article id (KA_COL_DOC_ID) — used for source "
            "rotation (Pedagogical-Aware RAG Reranking, pedagogical_reranker.py)."
        ),
    )
    """ RU: id статьи-источника факта; используется для ротации источников
    в педагогическом ре-ранкинге. """

    @field_validator("cluster_key", mode="before")
    @classmethod
    def _coerce_cluster_key(cls, v: object) -> str:
        raw = str(v or "general").strip().lower() or "general"
        return raw[:64]

    @field_validator("source_chunk_ids", mode="before")
    @classmethod
    def _coerce_source_chunk_ids(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if isinstance(v, (list, tuple, set)):
            return merge_source_chunk_ids([str(x) for x in v])
        return []

    @field_validator("scope", mode="before")
    @classmethod
    def _coerce_scope(cls, v: object) -> object:
        return coerce_scope_type(v)

    @field_validator("statement", mode="before")
    @classmethod
    def _strip_statement(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("context_quote", mode="before")
    @classmethod
    def _strip_context_quote(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @field_validator("statement", mode="before")
    @classmethod
    def _pad_short_statement(cls, v: object) -> object:
        """Avoid hard ValidationError on slightly-short Gemma statements."""
        if isinstance(v, str):
            s = v.strip()
            if 0 < len(s) < 8:
                return (s + " — (see context)").strip()[:2000]
            return s
        return v

    def format_tagged(self) -> str:
        body = (self.statement or "").strip()
        return f"[SCOPE: {self.scope.value}] {body}".strip()

    @classmethod
    def from_tagged_line(cls, line: str) -> KnowledgeAtom | None:
        raw = (line or "").strip()
        if not raw:
            return None
        m = _SCOPE_TAG_RE.match(raw)
        if m:
            scope = coerce_scope_type(m.group(1))
            statement = (m.group(2) or "").strip()
            if len(statement) < 8:
                return None
            return cls(scope=scope, statement=statement)
        # без тега — консервативно CONCEPT (базис), не PRACTICE
        if len(raw) < 8:
            return None
        return cls(scope=ScopeType.CONCEPT, statement=raw[:2000])


class ParagraphInspectionResult(BaseModel):
    """Structured Output Lite/Map: paragraph/window inspection → knowledge atoms."""

    atoms: list[KnowledgeAtom] = Field(
        default_factory=list,
        max_length=24,
        description="Extracted claims with mandatory scope",
    )

    @field_validator("atoms", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> object:
        return v if v is not None else []


class AggregatedKnowledgeBase(BaseModel):
    """Агрегация атомов для передачи Тьютору (5 раздельных блоков)."""

    concepts: list[KnowledgeAtom] = Field(default_factory=list)
    mechanics: list[KnowledgeAtom] = Field(default_factory=list)
    practices: list[KnowledgeAtom] = Field(default_factory=list)
    edge_cases: list[KnowledgeAtom] = Field(default_factory=list)
    anti_patterns: list[KnowledgeAtom] = Field(default_factory=list)

    @classmethod
    def from_atoms(cls, atoms: Iterable[KnowledgeAtom]) -> AggregatedKnowledgeBase:
        buckets: dict[ScopeType, list[KnowledgeAtom]] = {s: [] for s in ScopeType}
        by_key: dict[str, KnowledgeAtom] = {}
        for atom in atoms:
            key = f"{atom.scope.value}|{atom.statement.lower()}"
            existing = by_key.get(key)
            if existing is not None:
                existing.source_chunk_ids = merge_source_chunk_ids(
                    existing.source_chunk_ids,
                    atom.source_chunk_ids,
                )
                if (
                    not (existing.context_quote or "").strip()
                    and (atom.context_quote or "").strip()
                ):
                    existing.context_quote = atom.context_quote
                continue
            by_key[key] = atom
            buckets[atom.scope].append(atom)
        return cls(
            concepts=buckets[ScopeType.CONCEPT],
            mechanics=buckets[ScopeType.MECHANIC],
            practices=buckets[ScopeType.PRACTICE],
            edge_cases=buckets[ScopeType.EDGE_CASE],
            anti_patterns=buckets[ScopeType.ANTI_PATTERN],
        )

    @classmethod
    def from_tagged_strings(cls, lines: Iterable[str]) -> AggregatedKnowledgeBase:
        atoms: list[KnowledgeAtom] = []
        for line in lines:
            atom = KnowledgeAtom.from_tagged_line(line)
            if atom is not None:
                atoms.append(atom)
        return cls.from_atoms(atoms)

    def all_atoms(self) -> list[KnowledgeAtom]:
        return (
            list(self.concepts)
            + list(self.mechanics)
            + list(self.practices)
            + list(self.edge_cases)
            + list(self.anti_patterns)
        )

    def to_tagged_takeaways(self, *, max_items: int = 24) -> list[str]:
        out = [a.format_tagged() for a in self.all_atoms()]
        return out[: max(1, max_items)]

    def format_tutor_blocks(self, *, max_per_bucket: int = 16) -> str:
        """Five explicit blocks for tutor system/user context (English labels)."""

        def _bucket(title: str, items: list[KnowledgeAtom]) -> str:
            if not items:
                return f"### {title}\n(no extracted claims)"
            lines = [f"### {title}"]
            for atom in items[:max_per_bucket]:
                line = f"- {atom.format_tagged()}"
                q = (atom.context_quote or "").strip()
                if q:
                    line += f"\n  quote: «{q[:240]}»"
                lines.append(line)
            return "\n".join(lines)

        return "\n\n".join(
            [
                _bucket(
                    "FUNDAMENTAL CONCEPTS (Basis) [SCOPE: CONCEPT]",
                    self.concepts,
                ),
                _bucket(
                    "GENERALIZED MECHANICS [SCOPE: MECHANIC]",
                    self.mechanics,
                ),
                _bucket(
                    "ROUTINE PRACTICE (Footnotes and examples) [SCOPE: PRACTICE]",
                    self.practices,
                ),
                _bucket(
                    "EDGE CASES (Boundary/degradation behavior) [SCOPE: EDGE_CASE]",
                    self.edge_cases,
                ),
                _bucket(
                    "ANTI-PATTERNS (Known mistakes, do NOT do this) [SCOPE: ANTI_PATTERN]",
                    self.anti_patterns,
                ),
            ]
        )


SCOPE_TAGGING_PROMPT_RULES = (
    "=== KNOWLEDGE TRIANGULATION (mandatory scope tagging, 5 categories) ===\n"
    "Tag every extracted claim with exactly one generalization level. These are "
    "STRICT META-INVARIANTS about the STATEMENT'S OWN abstraction level — never "
    "decide the tag by which technology/topic the source article is about.\n\n"
    "- [SCOPE: CONCEPT] — conceptual basis (WHY / WHAT). Positive invariant: "
    "states a fundamental problem, architectural pattern, concept, law, or "
    "system-level trade-off — answers 'why is this needed / what does it protect "
    "against / what is the core idea', independent of any specific implementation. "
    "STRICT NEGATIVE CUTOFF: if the statement names a concrete programming-language "
    "construct, a specific API/function call, a specific configuration-parameter "
    "name, a numeric metric, a version, or an environment-specific setting, it "
    "CANNOT be CONCEPT — rephrase at a higher level of abstraction, or use "
    "MECHANIC/PRACTICE/EDGE_CASE/ANTI_PATTERN instead.\n\n"
    "- [SCOPE: MECHANIC] — generalized mechanism (HOW). Positive invariant: "
    "describes a generalized algorithm, pipeline phase/stage, lifecycle, "
    "interaction scheme, or class of data structure — how it works internally, in "
    "theory, as a reusable pattern, under NORMAL/expected operating conditions. "
    "Flexible terminology boundary: common, textbook data-structure/pattern terms "
    "are ALLOWED and ENCOURAGED here (e.g. an index, a write-ahead log, a buffer "
    "pool, hashing, a ring buffer) as long as the term names a GENERAL class of "
    "mechanism — never a specific library's API call, function signature, or an "
    "implementation/version-specific detail.\n\n"
    "- [SCOPE: PRACTICE] — routine specific case (WHERE / METRICS, happy path). "
    "Positive invariant: concrete numeric measurements, experiment parameters, "
    "standard configuration, code fragments, library/platform versions, or "
    "example function calls/syntax that are particular to THIS source or study "
    "and describe NORMAL, intended usage — not portable to other systems of the "
    "same class as-is.\n\n"
    "- [SCOPE: EDGE_CASE] — boundary/degradation behavior. Positive invariant: "
    "describes what happens at or beyond a limit — saturation, overload, "
    "concurrency race windows, resource exhaustion, unusual input shape, or any "
    "state the system enters only away from the common/expected path. STRICT "
    "NEGATIVE CUTOFF: routine behavior under normal load/inputs is never "
    "EDGE_CASE, even if it names a specific number or parameter (use PRACTICE) "
    "and even if it is a generalized boundary CONDITION rather than a boundary "
    "VALUE (use MECHANIC for a generalized description of how limits are "
    "handled in the abstract).\n\n"
    "- [SCOPE: ANTI_PATTERN] — known mistake / what NOT to do. Positive "
    "invariant: explicitly frames a choice, configuration, or usage as wrong, "
    "risky, deprecated, or a common source of bugs — a warning or "
    "deconstruction of bad practice, not neutral description of behavior. "
    "STRICT NEGATIVE CUTOFF: a plain description of a limitation or a failure "
    "mode with no explicit 'don't do this / this is wrong' framing is EDGE_CASE, "
    "not ANTI_PATTERN — ANTI_PATTERN requires the source to be prescriptive "
    "about avoiding a choice, not just descriptive about a boundary.\n\n"
    "=== GROUNDING & NO-PADDING INVARIANTS (STRICT NO-HALLUCINATION) ===\n"
    "1. ZERO SYNTHETIC PADDING: Never invent or fabricate abstract CONCEPT/"
    "MECHANIC statements if the source text contains only implementation code "
    "or metrics. If a chunk contains only concrete details, output ONLY "
    "PRACTICE or EDGE_CASE atoms.\n"
    "2. ANCHOR VERIFICATION: Every atom MUST have a verbatim `context_quote` "
    "from the source chunk. If the statement adds information not present in "
    "`context_quote`, IT IS STRICTLY FORBIDDEN.\n\n"
    "Do not drop tags when passing results downstream (Map → Reduce). In JSON fill "
    "`knowledge_atoms` / `atoms` as {scope, statement, context_quote, source_chunk_ids}.\n"
    "source_chunk_ids is usually filled by the pipeline from CHUNK_ID; preserve it if present.\n"
    "In textual takeaways, duplicate the `[SCOPE: …]` prefix at the start of each line."
)

KNOWLEDGE_TRIANGULATION_TUTOR_RULES = (
    "### KNOWLEDGE TRIANGULATION (semantic hierarchy, 5 categories)\n\n"
    "When source material carries `[SCOPE: CONCEPT|MECHANIC|PRACTICE|EDGE_CASE|"
    "ANTI_PATTERN]` tags (or separate CONCEPT / MECHANIC / PRACTICE / EDGE_CASE / "
    "ANTI_PATTERN blocks), obey:\n\n"
    "1. Lecture basis MUST be built ONLY from `[SCOPE: CONCEPT]` and "
    "`[SCOPE: MECHANIC]`.\n"
    "   - Explain fundamentals, architectural problems, and patterns.\n"
    "   - Generalize private names: if sources say «AJV» or «Pydantic», "
    "prefer the category «schema validation tools» in the main narrative.\n\n"
    "2. `[SCOPE: PRACTICE]` and `[SCOPE: EDGE_CASE]` particulars go ONLY into "
    "illustrative blocks:\n"
    "   - FORBIDDEN: present experimental parameters "
    "(e.g. «8.3 ms latency», «32 nesting levels», «library X») as industry-wide standards.\n"
    "   - Format every PRACTICE claim as a case/footnote block in Russian user output:\n"
    "     `> 📊 Практический пример / Показатели исследования [S1]/[Rn]: …`\n"
    "   - Format every EDGE_CASE claim as an explicit boundary-behavior warning:\n"
    "     `> ⚠️ Граничный случай / Поведение при нагрузке [S1]/[Rn]: …`\n\n"
    "3. `[SCOPE: ANTI_PATTERN]` claims MUST be presented as explicit "
    "'do NOT do this' warnings, never as neutral alternatives:\n"
    "     `> 🚫 Антипаттерн / Так делать не стоит [S1]/[Rn]: …`\n\n"
    "4. Conflicting PRACTICE metrics across sources → lift to MECHANIC level "
    "(e.g. «latency ranges from single-digit to tens of milliseconds depending on "
    "validation depth») instead of picking one number as truth.\n\n"
    "5. NO PARAMETRIC INJECTION: never state a fact that is not grounded in the "
    "provided RAG context (CONCEPT/MECHANIC/PRACTICE/EDGE_CASE/ANTI_PATTERN "
    "blocks above) — do not fill gaps from general training knowledge and "
    "present it as if it came from the sources. If the RAG context does not "
    "cover the question, say so explicitly rather than inventing an answer.\n"
)


_INLINE_SCOPE_RE = re.compile(
    rf"\[SCOPE:\s*({_SCOPE_NAMES})\s*\]\s*[^\n\[]+",
    re.I,
)


def extract_tagged_lines(text: str) -> list[str]:
    """Вытащить строки/фрагменты с явным [SCOPE: …] из произвольного текста."""
    raw = (text or "").strip()
    if not raw:
        return []
    found: list[str] = []
    for line in raw.splitlines():
        s = line.strip().lstrip("-•* ").strip()
        if _SCOPE_TAG_RE.match(s):
            found.append(s)
    if found:
        return found
    return [m.group(0).strip() for m in _INLINE_SCOPE_RE.finditer(raw)]


def normalize_knowledge_atoms(
    atoms: Iterable[KnowledgeAtom | dict | str] | None,
    *,
    fallback_lines: Iterable[str] | None = None,
) -> list[KnowledgeAtom]:
    """Валидация/нормализация атомов; fallback — tagged takeaway-строки."""
    out: list[KnowledgeAtom] = []
    by_key: dict[str, KnowledgeAtom] = {}

    def _push(atom: KnowledgeAtom | None) -> None:
        if atom is None:
            return
        key = f"{atom.scope.value}|{atom.statement.lower()}"
        existing = by_key.get(key)
        if existing is not None:
            existing.source_chunk_ids = merge_source_chunk_ids(
                existing.source_chunk_ids,
                atom.source_chunk_ids,
            )
            if (
                not (existing.context_quote or "").strip()
                and (atom.context_quote or "").strip()
            ):
                existing.context_quote = atom.context_quote
            return
        by_key[key] = atom
        out.append(atom)

    for item in atoms or []:
        if isinstance(item, KnowledgeAtom):
            _push(item)
        elif isinstance(item, dict):
            try:
                _push(KnowledgeAtom.model_validate(item))
            except Exception:
                continue
        elif isinstance(item, str):
            _push(KnowledgeAtom.from_tagged_line(item))

    if not out and fallback_lines:
        for line in fallback_lines:
            _push(KnowledgeAtom.from_tagged_line(line))
    return out


def attach_source_chunk_id(
    atoms: Iterable[KnowledgeAtom],
    chunk_id: str,
) -> list[KnowledgeAtom]:
    """Ensure each atom lists ``chunk_id`` in ``source_chunk_ids``."""
    cid = (chunk_id or "").strip()
    if not cid:
        return list(atoms)
    out: list[KnowledgeAtom] = []
    for atom in atoms:
        ids = merge_source_chunk_ids(atom.source_chunk_ids, [cid])
        if ids == list(atom.source_chunk_ids or []):
            out.append(atom)
        else:
            out.append(atom.model_copy(update={"source_chunk_ids": ids}))
    return out


def reattach_source_chunk_ids_from_raw(
    clean: Iterable[KnowledgeAtom],
    raw: Iterable[KnowledgeAtom],
) -> list[KnowledgeAtom]:
    """
    After REDUCE dedup: union source_chunk_ids from overlapping raw atoms
    (exact statement match or containment), so Gemma drops do not erase provenance.
    """
    raw_list = list(raw)
    out: list[KnowledgeAtom] = []
    for c in clean:
        ids = list(c.source_chunk_ids or [])
        ckey = (c.statement or "").strip().lower()
        for r in raw_list:
            rkey = (r.statement or "").strip().lower()
            if not rkey:
                continue
            if ckey == rkey or (ckey and rkey and (ckey in rkey or rkey in ckey)):
                ids = merge_source_chunk_ids(ids, r.source_chunk_ids)
        out.append(
            c.model_copy(update={"source_chunk_ids": merge_source_chunk_ids(ids)})
            if ids != list(c.source_chunk_ids or [])
            else c
        )
    return out


def tagged_takeaways_from_atoms(
    atoms: Iterable[KnowledgeAtom],
    *,
    max_items: int = 24,
) -> list[str]:
    return AggregatedKnowledgeBase.from_atoms(atoms).to_tagged_takeaways(
        max_items=max_items
    )


def format_takeaways_for_tutor(
    takeaways: Iterable[str], *, max_per_bucket: int = 16
) -> str:
    """Разложить tagged takeaways в 5 блоков для контекста Тьютора."""
    kb = AggregatedKnowledgeBase.from_tagged_strings(takeaways)
    return kb.format_tutor_blocks(max_per_bucket=max_per_bucket)
