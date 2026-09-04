"""Cross-node context drift: weakness ledger for adaptive asterisk-question overlay."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field, field_validator

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.ui.run_log import trace

_STORE_DIR = PACKAGE_ROOT / ".runs"
_lock = threading.RLock()
_STORE_DIR_OVERRIDE: Path | None = None

_MAX_OPEN_TAGS = 12
_MAX_SUMMARIES = 16
_MAX_CLOSED_TAGS = 24
_CLOSED_TAG_MAX_AGE_HOURS = 168  # 7 days
_TAG_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def set_weakness_ledger_store_dir(path: Path | None) -> None:
    """Tests: isolate ledger JSON under tmp_path. ``None`` restores default."""
    global _STORE_DIR_OVERRIDE
    _STORE_DIR_OVERRIDE = path


def parse_curriculum_id_from_anchor(anchor: str) -> str:
    """``node_deep_dive:{curriculum_id}:{node_id}`` → curriculum_id (else empty)."""
    raw = (anchor or "").strip()
    prefix = "node_deep_dive:"
    if not raw.startswith(prefix):
        return ""
    rest = raw[len(prefix) :]
    if ":" not in rest:
        return rest.strip()
    cid, _nid = rest.rsplit(":", 1)
    return cid.strip()


def normalize_weakness_tag(raw: str) -> str:
    t = " ".join((raw or "").strip().lower().split())
    t = t.replace(" ", "_")
    t = _TAG_RE.sub("", t)
    return t[:64]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _store_dir() -> Path:
    return _STORE_DIR_OVERRIDE or _STORE_DIR


def _store_path(curriculum_id: str) -> Path:
    safe = re.sub(r"[^\w\-]+", "_", (curriculum_id or "").strip())[:80] or "unknown"
    return _store_dir() / f"weakness_ledger_{safe}.json"


def _parse_utc(raw: str) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


class ClosedWeaknessTag(BaseModel):
    """Archived tag after overlay close — pruned by age / cap."""

    tag: str = Field(min_length=1, max_length=64)
    closed_at: str = Field(default="", max_length=40)
    node_id: str = Field(default="", max_length=80)


class NodeSessionSummary(BaseModel):
    """Compact per-node snapshot carried into later nodes of the same curriculum."""

    node_id: str = Field(min_length=1, max_length=80)
    curriculum_id: str = Field(default="", max_length=80)
    title: str = Field(default="", max_length=300)
    topic_mastery_score: int = Field(default=0, ge=0, le=100)
    core_clean: bool = Field(
        default=True,
        description="True when core closed without lingering weakness_tags.",
    )
    weakness_tags: list[str] = Field(default_factory=list, max_length=12)
    overlay_types_earned: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="ADVANCED_ASTERISK and/or DEEP_ASTERISK earned on this node.",
    )
    updated_at: str = Field(default="", max_length=40)

    @field_validator("weakness_tags", "overlay_types_earned", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        if not v:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            tag = str(item or "").strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            out.append(tag[:64])
        return out


class SessionWeaknessLedger(BaseModel):
    """Open weakness_tags + node summaries for one curriculum (cross-node)."""

    curriculum_id: str = Field(default="", max_length=80)
    open_weaknesses: list[str] = Field(
        default_factory=list,
        max_length=_MAX_OPEN_TAGS,
        description="Normalized weakness_tags still open for later nodes.",
    )
    node_summaries: list[NodeSessionSummary] = Field(
        default_factory=list,
        max_length=_MAX_SUMMARIES,
    )
    closed_weaknesses: list[ClosedWeaknessTag] = Field(
        default_factory=list,
        max_length=_MAX_CLOSED_TAGS,
        description="Recently closed tags retained for decay/pruning (not prompt-open).",
    )

    @field_validator("open_weaknesses", mode="before")
    @classmethod
    def _coerce_open(cls, v: Any) -> list[str]:
        if not v:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            tag = normalize_weakness_tag(str(item or ""))
            if not tag or tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
        return out[:_MAX_OPEN_TAGS]

    @field_validator("closed_weaknesses", mode="before")
    @classmethod
    def _coerce_closed(cls, v: Any) -> list[Any]:
        if not v:
            return []
        out: list[Any] = []
        for item in v:
            if isinstance(item, str):
                tag = normalize_weakness_tag(item)
                if tag:
                    out.append({"tag": tag, "closed_at": _utc_now()})
            else:
                out.append(item)
        return out[:_MAX_CLOSED_TAGS]

    def record_weaknesses(
        self,
        tags: Sequence[str] | Iterable[str],
        *,
        node_id: str = "",
        title: str = "",
        topic_mastery_score: int = 0,
        note: str = "",
    ) -> list[str]:
        """Append unique weakness_tags. Returns newly added tags."""
        _ = note
        added: list[str] = []
        cur = list(self.open_weaknesses or [])
        seen = set(cur)
        for raw in tags or []:
            tag = normalize_weakness_tag(str(raw or ""))
            if not tag or tag in seen:
                continue
            seen.add(tag)
            cur.append(tag)
            added.append(tag)
        self.open_weaknesses = cur[:_MAX_OPEN_TAGS]
        nid = (node_id or "").strip()
        if nid:
            self.upsert_node_summary(
                NodeSessionSummary(
                    node_id=nid,
                    curriculum_id=self.curriculum_id,
                    title=(title or "").strip()[:300],
                    topic_mastery_score=int(topic_mastery_score or 0),
                    core_clean=not bool(self.open_weaknesses),
                    weakness_tags=list(self.open_weaknesses),
                    updated_at=_utc_now(),
                )
            )
        return added

    def clear_weaknesses(
        self,
        tags: Sequence[str] | Iterable[str] | None = None,
        *,
        node_id: str = "",
        overlay_type: str = "",
    ) -> list[str]:
        """
        Drop matching open tags (all open tags when ``tags`` is empty/None).

        Called after a successful asterisk-question (ADVANCED_ASTERISK / DEEP_ASTERISK).
        Returns tags actually removed.
        """
        if tags is None or (isinstance(tags, (list, tuple)) and len(list(tags)) == 0):
            removed = list(self.open_weaknesses or [])
            self.open_weaknesses = []
        else:
            want = {
                normalize_weakness_tag(str(t or ""))
                for t in tags
                if normalize_weakness_tag(str(t or ""))
            }
            kept: list[str] = []
            removed = []
            for tag in self.open_weaknesses or []:
                if tag in want:
                    removed.append(tag)
                else:
                    kept.append(tag)
            self.open_weaknesses = kept
        nid = (node_id or "").strip()
        archived: list[ClosedWeaknessTag] = list(self.closed_weaknesses or [])
        now = _utc_now()
        existing = {(c.tag, c.node_id) for c in archived}
        for tag in removed:
            key = (tag, nid)
            if key in existing:
                continue
            archived.append(
                ClosedWeaknessTag(tag=tag, closed_at=now, node_id=nid)
            )
            existing.add(key)
        self.closed_weaknesses = archived[-_MAX_CLOSED_TAGS:]
        self.prune_closed_tags()
        if nid:
            prev = next(
                (s for s in self.node_summaries if s.node_id == nid),
                None,
            )
            earned = list(prev.overlay_types_earned) if prev is not None else []
            otype = (overlay_type or "").strip().upper()
            if otype in ("ADVANCED_ASTERISK", "DEEP_ASTERISK") and otype not in earned:
                earned.append(otype)
            self.upsert_node_summary(
                NodeSessionSummary(
                    node_id=nid,
                    curriculum_id=self.curriculum_id,
                    title=(prev.title if prev is not None else "")[:300],
                    topic_mastery_score=(
                        prev.topic_mastery_score if prev is not None else 0
                    ),
                    core_clean=not bool(self.open_weaknesses),
                    weakness_tags=list(self.open_weaknesses),
                    overlay_types_earned=earned,
                    updated_at=_utc_now(),
                )
            )
        return removed

    def upsert_node_summary(self, summary: NodeSessionSummary) -> None:
        rows = [s for s in self.node_summaries if s.node_id != summary.node_id]
        rows.append(summary)
        self.node_summaries = rows[-_MAX_SUMMARIES:]

    def prune_closed_tags(
        self,
        *,
        now: datetime | None = None,
        max_age_hours: int = _CLOSED_TAG_MAX_AGE_HOURS,
        max_keep: int = _MAX_CLOSED_TAGS,
    ) -> dict[str, int]:
        """
        Drop closed tags older than ``max_age_hours`` and cap the archive.

        Also decays ``node_summaries.weakness_tags`` so closed tags do not linger
        on historical snapshots after they have been pruned.
        """
        prior_closed = {c.tag for c in (self.closed_weaknesses or [])}
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(
            hours=max(1, int(max_age_hours))
        )
        kept: list[ClosedWeaknessTag] = []
        dropped = 0
        for rec in self.closed_weaknesses or []:
            ts = _parse_utc(rec.closed_at)
            if ts is not None and ts < cutoff:
                dropped += 1
                continue
            kept.append(rec)
        overflow = max(0, len(kept) - max(1, int(max_keep)))
        if overflow:
            dropped += overflow
            kept = kept[-int(max_keep) :]
        self.closed_weaknesses = kept

        live_closed = {c.tag for c in kept}
        aged_out = prior_closed - live_closed
        decayed = 0
        refreshed: list[NodeSessionSummary] = []
        for summary in self.node_summaries or []:
            tags = list(summary.weakness_tags or [])
            nxt = [t for t in tags if t not in aged_out]
            if len(nxt) != len(tags):
                decayed += len(tags) - len(nxt)
                refreshed.append(
                    summary.model_copy(update={"weakness_tags": nxt})
                )
            else:
                refreshed.append(summary)
        self.node_summaries = refreshed[-_MAX_SUMMARIES:]
        return {"dropped_closed": dropped, "decayed_summary_tags": decayed}

    def open_weakness_tags(self) -> list[str]:
        return list(self.open_weaknesses or [])

    def has_open_weaknesses(self) -> bool:
        return bool(self.open_weaknesses)

    def build_cross_node_prompt_context(self, *, exclude_node_id: str = "") -> str:
        """English tutor/evaluator block. Empty when there is nothing to carry."""
        skip = (exclude_node_id or "").strip()
        tags = list(self.open_weaknesses or [])
        summaries = [
            s for s in (self.node_summaries or []) if s.node_id and s.node_id != skip
        ]
        if not tags and not summaries:
            return ""
        lines = [
            "=== PRIOR WEAKNESSES (cross-node asterisk-question ledger) ===",
        ]
        if tags:
            lines.append("Open weakness_tags: " + "; ".join(tags[:12]))
            lines.append(
                "Prefer ADVANCED_ASTERISK (Bloom L4 asterisk-question) to close "
                "these tags; do not treat them as core WHY/HOW/MECH credit."
            )
        else:
            lines.append("Open weakness_tags: (none — core history is clean).")
            lines.append(
                "Prefer DEEP_ASTERISK (Bloom L5/L6 asterisk-question) after a clean core."
            )
        if summaries:
            lines.append("Prior nodes:")
            for s in summaries[-6:]:
                earned = ", ".join(s.overlay_types_earned) or "none"
                wtags = ", ".join(s.weakness_tags) or "none"
                title = (s.title or s.node_id).strip()
                lines.append(
                    f"- {s.node_id} ({title}): mastery={s.topic_mastery_score}% "
                    f"core_clean={s.core_clean} tags=[{wtags}] "
                    f"overlay=[{earned}]"
                )
        return "\n".join(lines)


class ContextDriftManager:
    """Load/save ``SessionWeaknessLedger`` per curriculum; host-owned, not LLM."""

    def __init__(
        self,
        curriculum_id: str,
        *,
        persist: bool = True,
        store_path: Path | None = None,
        ledger: SessionWeaknessLedger | None = None,
    ) -> None:
        self.curriculum_id = (curriculum_id or "").strip()
        self.persist = bool(persist) and bool(self.curriculum_id)
        self._path = store_path
        if ledger is not None:
            self._ledger = ledger
            if not self._ledger.curriculum_id:
                self._ledger.curriculum_id = self.curriculum_id
        elif self.persist:
            self._ledger = self._load()
        else:
            self._ledger = SessionWeaknessLedger(curriculum_id=self.curriculum_id)

    def _path_resolved(self) -> Path:
        return self._path or _store_path(self.curriculum_id)

    def _load(self) -> SessionWeaknessLedger:
        path = self._path_resolved()
        if not path.is_file():
            return SessionWeaknessLedger(curriculum_id=self.curriculum_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            led = SessionWeaknessLedger.model_validate(raw)
            if not led.curriculum_id:
                led.curriculum_id = self.curriculum_id
            led.prune_closed_tags()
            return led
        except Exception as exc:
            trace(f"CONTEXT_DRIFT load skip | {type(exc).__name__}: {exc}")
            return SessionWeaknessLedger(curriculum_id=self.curriculum_id)

    def _save(self) -> None:
        if not self.persist:
            return
        path = self._path_resolved()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                self._ledger.model_dump_json(indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception as exc:
            trace(f"CONTEXT_DRIFT save skip | {type(exc).__name__}: {exc}")

    @property
    def ledger(self) -> SessionWeaknessLedger:
        return self._ledger

    def record_weaknesses(
        self,
        tags: Sequence[str] | Iterable[str],
        *,
        node_id: str = "",
        title: str = "",
        topic_mastery_score: int = 0,
        note: str = "",
    ) -> list[str]:
        with _lock:
            added = self._ledger.record_weaknesses(
                tags,
                node_id=node_id,
                title=title,
                topic_mastery_score=topic_mastery_score,
                note=note,
            )
            if added:
                self._save()
                trace(
                    "CONTEXT_DRIFT record_weaknesses | "
                    f"curriculum={self.curriculum_id} node={node_id} "
                    f"added={added}"
                )
            return added

    def clear_weaknesses(
        self,
        tags: Sequence[str] | Iterable[str] | None = None,
        *,
        node_id: str = "",
        overlay_type: str = "",
    ) -> list[str]:
        with _lock:
            removed = self._ledger.clear_weaknesses(
                tags,
                node_id=node_id,
                overlay_type=overlay_type,
            )
            if removed or overlay_type:
                self._save()
                trace(
                    "CONTEXT_DRIFT clear_weaknesses | "
                    f"curriculum={self.curriculum_id} node={node_id} "
                    f"removed={removed} overlay={overlay_type or '—'}"
                )
            return removed

    def open_weakness_tags(self) -> list[str]:
        return self._ledger.open_weakness_tags()

    def has_open_weaknesses(self) -> bool:
        return self._ledger.has_open_weaknesses()

    def build_cross_node_prompt_context(self, *, exclude_node_id: str = "") -> str:
        return self._ledger.build_cross_node_prompt_context(
            exclude_node_id=exclude_node_id
        )

    def upsert_node_summary(self, summary: NodeSessionSummary) -> None:
        with _lock:
            if not summary.curriculum_id:
                summary.curriculum_id = self.curriculum_id
            self._ledger.upsert_node_summary(summary)
            self._save()

    def prune_closed_tags(
        self,
        *,
        now: datetime | None = None,
        max_age_hours: int = _CLOSED_TAG_MAX_AGE_HOURS,
        max_keep: int = _MAX_CLOSED_TAGS,
    ) -> dict[str, int]:
        with _lock:
            stats = self._ledger.prune_closed_tags(
                now=now,
                max_age_hours=max_age_hours,
                max_keep=max_keep,
            )
            if stats.get("dropped_closed") or stats.get("decayed_summary_tags"):
                self._save()
            return stats


def mix_prior_weaknesses_into_eval_system(
    system: str,
    *,
    curriculum_id: str = "",
    exclude_node_id: str = "",
    persist: bool = True,
) -> str:
    """Append PRIOR WEAKNESSES block to an evaluator system prompt when the ledger is open."""
    cid = (curriculum_id or "").strip()
    base = (system or "").rstrip()
    if not cid:
        return base
    mgr = ContextDriftManager(cid, persist=persist)
    block = mgr.build_cross_node_prompt_context(exclude_node_id=exclude_node_id)
    if not block:
        return base
    return f"{base}\n\n{block}"


def tags_from_focus_and_critique(
    *,
    focus_hint: str = "",
    unaccounted_edge_cases: Sequence[str] | None = None,
    weak_or_risk_concepts: Sequence[str] | None = None,
    directive: str = "",
) -> list[str]:
    """Host-side tag harvest from evaluator artefacts (not LLM-invented chip labels)."""
    raw: list[str] = []
    hint = (focus_hint or "").strip()
    if hint:
        raw.append(hint[:80])
    for edge in unaccounted_edge_cases or []:
        e = (edge or "").strip()
        if e:
            raw.append(e[:80])
    for concept in weak_or_risk_concepts or []:
        c = (concept or "").strip()
        if c:
            raw.append(c[:80])
    d = (directive or "").strip().upper()
    if d.startswith("PROBE_NEXT_LAYER:"):
        layer = d.split(":", 1)[-1].strip().lower()
        if layer:
            raw.append(f"{layer}_gap")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        tag = normalize_weakness_tag(item)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out[:8]
