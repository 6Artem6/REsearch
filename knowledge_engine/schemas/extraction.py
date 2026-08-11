"""Knowledge Triangulation — scope tags for extraction → tutor packing."""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field, field_validator

_SCOPE_TAG_RE = re.compile(
    r"^\s*\[SCOPE:\s*(PRINCIPLE|MECHANIC|INSTANCE)\s*\]\s*(.*)$",
    re.I | re.S,
)


class ScopeType(str, Enum):
    PRINCIPLE = "PRINCIPLE"
    MECHANIC = "MECHANIC"
    INSTANCE = "INSTANCE"


def coerce_scope_type(
    value: object, *, default: ScopeType = ScopeType.PRINCIPLE
) -> ScopeType:
    """Best-effort ScopeType parse (case / typos) for defensive Pydantic."""
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
    # Common aliases / partials
    aliases = {
        "PRINCIPLE": ScopeType.PRINCIPLE,
        "PRINCIPLES": ScopeType.PRINCIPLE,
        "BASIS": ScopeType.PRINCIPLE,
        "FUNDAMENTAL": ScopeType.PRINCIPLE,
        "MECHANIC": ScopeType.MECHANIC,
        "MECHANICS": ScopeType.MECHANIC,
        "MECHANISM": ScopeType.MECHANIC,
        "ALGORITHM": ScopeType.MECHANIC,
        "INSTANCE": ScopeType.INSTANCE,
        "INSTANCES": ScopeType.INSTANCE,
        "EVIDENCE": ScopeType.INSTANCE,
        "EXAMPLE": ScopeType.INSTANCE,
        "CASE": ScopeType.INSTANCE,
        "METRIC": ScopeType.INSTANCE,
    }
    if upper in aliases:
        return aliases[upper]
    for key, scope in aliases.items():
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
            "PRINCIPLE — fundamental principle/pattern; "
            "MECHANIC — generalized algorithm/stage; "
            "INSTANCE — particular case, numbers, libraries, experiment metrics"
        ),
    )
    statement: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Claim without URL; INSTANCE may include numbers and library names",
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
        # без тега — консервативно PRINCIPLE (базис), не INSTANCE
        if len(raw) < 8:
            return None
        return cls(scope=ScopeType.PRINCIPLE, statement=raw[:2000])


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
    """Агрегация атомов для передачи Тьютору (3 раздельных блока)."""

    principles: list[KnowledgeAtom] = Field(default_factory=list)
    mechanics: list[KnowledgeAtom] = Field(default_factory=list)
    evidence_cases: list[KnowledgeAtom] = Field(default_factory=list)

    @classmethod
    def from_atoms(cls, atoms: Iterable[KnowledgeAtom]) -> AggregatedKnowledgeBase:
        principles: list[KnowledgeAtom] = []
        mechanics: list[KnowledgeAtom] = []
        evidence: list[KnowledgeAtom] = []
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
            if atom.scope == ScopeType.PRINCIPLE:
                principles.append(atom)
            elif atom.scope == ScopeType.MECHANIC:
                mechanics.append(atom)
            else:
                evidence.append(atom)
        return cls(
            principles=principles,
            mechanics=mechanics,
            evidence_cases=evidence,
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
        return list(self.principles) + list(self.mechanics) + list(self.evidence_cases)

    def to_tagged_takeaways(self, *, max_items: int = 24) -> list[str]:
        out = [a.format_tagged() for a in self.all_atoms()]
        return out[: max(1, max_items)]

    def format_tutor_blocks(self, *, max_per_bucket: int = 16) -> str:
        """Three explicit blocks for tutor system/user context (English labels)."""

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
                    "FUNDAMENTAL PRINCIPLES (Basis) [SCOPE: PRINCIPLE]",
                    self.principles,
                ),
                _bucket(
                    "GENERALIZED MECHANICS [SCOPE: MECHANIC]",
                    self.mechanics,
                ),
                _bucket(
                    "PRACTICAL CASES AND EMPIRICAL DATA "
                    "(Footnotes and examples only) [SCOPE: INSTANCE]",
                    self.evidence_cases,
                ),
            ]
        )


SCOPE_TAGGING_PROMPT_RULES = (
    "=== KNOWLEDGE TRIANGULATION (mandatory scope tagging) ===\n"
    "Tag every extracted claim with exactly one generalization level:\n"
    "- [SCOPE: PRINCIPLE] — fundamental principle, architectural pattern, concept, "
    "problem, or law (why is it needed? what does it protect against? what is the core idea?).\n"
    "- [SCOPE: MECHANIC] — generalized algorithm, pipeline stage, scheme, or method class "
    "(how it works in theory, without a concrete library or experiment number).\n"
    "- [SCOPE: INSTANCE] — particular case, experiment parameters, concrete numbers, "
    "latencies, library names, limits (how and with which metrics it was done in this study).\n"
    "Do not drop tags when passing results downstream (Map → Reduce). In JSON fill "
    "`knowledge_atoms` / `atoms` as {scope, statement, context_quote, source_chunk_ids}.\n"
    "source_chunk_ids is usually filled by the pipeline from CHUNK_ID; preserve it if present.\n"
    "In textual takeaways, duplicate the `[SCOPE: …]` prefix at the start of each line."
)

KNOWLEDGE_TRIANGULATION_TUTOR_RULES = (
    "### KNOWLEDGE TRIANGULATION (semantic hierarchy)\n\n"
    "When source material carries `[SCOPE: PRINCIPLE|MECHANIC|INSTANCE]` tags "
    "(or separate PRINCIPLE / MECHANIC / EVIDENCE blocks), obey:\n\n"
    "1. Lecture basis (~70% of volume) MUST be built ONLY from "
    "`[SCOPE: PRINCIPLE]` and `[SCOPE: MECHANIC]`.\n"
    "   - Explain fundamentals, architectural problems, and patterns.\n"
    "   - Generalize private names: if sources say «AJV» or «Pydantic», "
    "prefer the category «schema validation tools» in the main narrative.\n\n"
    "2. `[SCOPE: INSTANCE]` particulars (~30%) go ONLY into illustrative blocks:\n"
    "   - FORBIDDEN: present experimental parameters "
    "(e.g. «8.3 ms latency», «32 nesting levels», «library X») as industry-wide standards.\n"
    "   - Format every INSTANCE claim as a case/footnote block in Russian user output:\n"
    "     `> 📊 Практический пример / Показатели исследования [S1]/[Rn]: …`\n\n"
    "3. Conflicting INSTANCE metrics across sources → lift to MECHANIC level "
    "(e.g. «latency ranges from single-digit to tens of milliseconds depending on "
    "validation depth») instead of picking one number as truth.\n"
)


_INLINE_SCOPE_RE = re.compile(
    r"\[SCOPE:\s*(PRINCIPLE|MECHANIC|INSTANCE)\s*\]\s*[^\n\[]+",
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
    """Разложить tagged takeaways в 3 блока для контекста Тьютора."""
    kb = AggregatedKnowledgeBase.from_tagged_strings(takeaways)
    return kb.format_tutor_blocks(max_per_bucket=max_per_bucket)
