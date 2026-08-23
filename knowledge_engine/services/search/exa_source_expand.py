"""Flash Lite expansion of Exa include_domains / category / search type."""

from __future__ import annotations

from knowledge_engine.llm_locale import RUSSIAN_OUTPUT_RULE
from knowledge_engine.schemas.llm_contracts.exa_search import (
    AUTHORITY_KEEP_CLASSES,
    PASS1_INCLUDE_CLASSES,
    BatchDomainAuthorityResponse,
    DomainAuthorityItem,
    DomainAuthorityVerdict,
    ExaSearchContextExpansion,
)
from knowledge_engine.services.search.exa_domains import (
    add_dynamic_exa_domain,
    clean_domain_for_exa,
    is_official_docs_host,
)
from knowledge_engine.src.source_evaluator.evaluator import match_whitelist
from knowledge_engine.ui.run_log import trace

_EXPAND_SYSTEM = (
    "You are a Search Source Architect for Exa neural/keyword web search.\n"
    "Given a lecture topic or engineering query, emit JSON matching "
    "ExaSearchContextExpansion.\n\n"
    "Intent:\n"
    "- language_api: languages, runtimes, specifications, APIs, frameworks, "
    "kernels, CPU/OS internals → canonical spec / official docs / source-tree "
    "hosts for THIS topic.\n"
    "- architecture: system design, production trade-offs → vendor engineering "
    "blogs (not primary_domains).\n"
    "- mixed: official spec/docs/source in primary_domains; blogs via category.\n\n"
    "topic_vector_query: one English high-level gist of the TOPIC family "
    "(same abstraction as a domain general_summary), e.g. "
    '"CPython core internals, official PEP specifications, and standard '
    'library documentation". FORBIDDEN: listing narrow subtopics.\n'
    "primary_domains: 2–8 hostnames WITHOUT scheme or path. Include ONLY hosts "
    "in these target classes (map all three to host-side OFFICIAL_DOCS):\n"
    "- CANONICAL_SPEC: first-party standards, RFC, ISO, IEEE, language PEPs "
    "(e.g. ietf.org, peps.python.org, w3.org, iso.org).\n"
    "- OFFICIAL_DOCS: vendor/language/OS/kernel/database documentation "
    "(e.g. docs.python.org, kernel.org, cppreference.com, man7.org).\n"
    "- SOURCE_TREE: official implementation hosts "
    "(e.g. github.com for CPython, kernel.org). Emit hostname only.\n"
    "FORBIDDEN in primary_domains — do not emit these classes:\n"
    "- AGGREGATOR_BLOG: article aggregators and tutorials "
    "(medium.com, habr.com, dev.to, geeksforgeeks.org).\n"
    "- QNA_FORUM: forums and Q&A (stackoverflow.com, reddit.com).\n"
    "- ACADEMY_SEO: generic SEO courses/cheatsheets "
    "(w3schools.com, javatpoint.com, baeldung.com).\n"
    "allowed_categories: Exa categories from "
    "company | research paper | news | github | pdf. "
    "Use research paper and/or pdf for specs; github for source trees; "
    "company for vendor blogs; empty list if the query should not constrain "
    "category.\n"
    "search_type: keyword for exact spec/API names; auto for mixed; "
    "neural only for purely narrative architecture queries.\n"
    "use_broader_search: true unless the topic is a named specification.\n"
    "include_official_docs: true whenever language_api or mixed.\n"
    f"{RUSSIAN_OUTPUT_RULE}\n"
    "JSON field values stay in English (hostnames, enums, topic_vector_query)."
)

_BATCH_AUTHORITY_SYSTEM = (
    "You classify a BATCH of hostnames for an engineering knowledge base.\n"
    "JSON matching BatchDomainAuthorityResponse: one item per listed domain.\n"
    "classification MUST be exactly one of the host enums below. "
    "The host derives KEEP/REJECT from classification.\n"
    "- OFFICIAL_DOCS → KEEP, Pass 1 include_domains: canonical specs, "
    "standards (RFC/ISO/IEEE/PEP), official language/OS/framework docs, "
    "and official source trees.\n"
    "- VENDOR_BLOG → KEEP for ordinary search, NEVER Pass 1 include_domains: "
    "engineering blogs of the product vendor (e.g. engineering.fb.com).\n"
    "- ACADEMIC_OR_PAPER → KEEP for ordinary search, NEVER Pass 1: "
    "arxiv, university labs, peer-reviewed publication hosts.\n"
    "- COMMUNITY_BLOG → REJECT: personal blogs, Habr, Medium, commercial "
    "tutorials (geeksforgeeks, baeldung, w3schools).\n"
    "- SPAM_AGGREGATOR → REJECT: SEO farms, content mills, code-generator "
    "scrapers, link aggregators without original engineering depth.\n"
    "general_summary: ONE canonical high-level gist of what the SITE is "
    "(resource family), in English. Examples: "
    '"CPython core internals, official PEP specifications, and standard '
    'library documentation"; '
    '"Linux kernel architecture, syscall specs, and subsystem documentation". '
    "FORBIDDEN: enumerating narrow lecture subtopics or article titles.\n"
    "reason: one short English sentence per item.\n"
)

_PASS2_DEFAULT_CATEGORIES: tuple[str, ...] = ("github", "pdf", "research paper")


def _unique_hosts(raw: list[str], *, cap: int = 16) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        host = clean_domain_for_exa(item)
        if not host or "." not in host or host in seen:
            continue
        seen.add(host)
        out.append(host)
        if len(out) >= cap:
            break
    return out


def fallback_exa_search_context(query: str) -> ExaSearchContextExpansion:
    """Deterministic fallback: skip Pass 1 domains, keep Pass 2 categories."""
    _ = query
    return ExaSearchContextExpansion(
        intent="mixed",
        primary_domains=[],
        allowed_categories=list(_PASS2_DEFAULT_CATEGORIES),
        search_type="auto",
        use_broader_search=True,
        include_official_docs=True,
        topic_vector_query="",
    )


def exa_pass2_categories(ctx: ExaSearchContextExpansion) -> list[str]:
    """Native Exa categories for the unconstrained Pass 2 search."""
    cats = [c for c in (ctx.allowed_categories or []) if c]
    if cats:
        return cats
    return list(_PASS2_DEFAULT_CATEGORIES)


def _lookup_registry_official_docs(topic_text: str) -> list[str]:
    q = (topic_text or "").strip()
    if not q:
        return []
    try:
        from knowledge_engine.services.search.domain_registry import (
            get_domain_registry,
        )

        return get_domain_registry().search_official_docs(q)
    except Exception as exc:
        trace(f"DOMAIN_REGISTRY lookup ⊘ | {exc}")
        return []


def _merge_registry_into_expansion(
    expanded: ExaSearchContextExpansion,
    query: str,
) -> ExaSearchContextExpansion:
    topic = (expanded.topic_vector_query or "").strip() or (query or "").strip()
    found = _lookup_registry_official_docs(topic)
    if not found:
        return expanded
    merged = _unique_hosts(list(expanded.primary_domains) + found, cap=16)
    trace(
        f"DOMAIN_REGISTRY merge | added={len(found)} "
        f"primary={len(merged)} topic={topic[:80]}"
    )
    return expanded.model_copy(update={"primary_domains": merged})


def expand_search_context_with_flash_lite(query: str) -> ExaSearchContextExpansion:
    """Lite topic + domain hypotheses, then LanceDB official-docs lookup."""
    q = (query or "").strip()
    if len(q) < 4:
        return _merge_registry_into_expansion(fallback_exa_search_context(q), q)

    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    user = (
        f"Query / lecture topic:\n{q[:1200]}\n\n"
        "Return ExaSearchContextExpansion JSON."
    )
    try:
        out = run_gemini_lite_structured(
            _EXPAND_SYSTEM,
            user,
            f"exa_expand:{q[:80]}",
            ExaSearchContextExpansion,
            "exa / search_context_expand",
        )
    except Exception as exc:
        trace(f"EXA expand ⊘ lite | {exc}")
        return _merge_registry_into_expansion(fallback_exa_search_context(q), q)

    if not isinstance(out, ExaSearchContextExpansion):
        out = ExaSearchContextExpansion.model_validate(out)

    domains = _unique_hosts(list(out.primary_domains or []))

    search_type = out.search_type or "auto"
    if out.intent == "language_api" and search_type == "neural":
        search_type = "keyword"

    expanded = ExaSearchContextExpansion(
        intent=out.intent or "mixed",
        primary_domains=domains[:16],
        allowed_categories=list(out.allowed_categories or [])[:4],
        search_type=search_type,
        use_broader_search=bool(out.use_broader_search),
        include_official_docs=bool(out.include_official_docs)
        or out.intent in ("language_api", "mixed"),
        topic_vector_query=(out.topic_vector_query or "").strip()[:400],
    )
    expanded = _merge_registry_into_expansion(expanded, q)
    trace(
        f"EXA expand ✓ | intent={expanded.intent} type={expanded.search_type} "
        f"domains={len(expanded.primary_domains)} "
        f"cats={expanded.allowed_categories}"
    )
    return expanded


def _item_to_verdict(item: DomainAuthorityItem) -> DomainAuthorityVerdict:
    return DomainAuthorityVerdict(
        domain=clean_domain_for_exa(item.domain),
        classification=item.classification,
        status="KEEP",
        reason=(item.reason or "").strip()[:400],
    )


def _commit_authority_item(item: DomainAuthorityItem) -> DomainAuthorityVerdict:
    verdict = _item_to_verdict(item)
    host = verdict.domain
    if verdict.status == "KEEP":
        add_dynamic_exa_domain(host, verdict.classification)
        try:
            from knowledge_engine.src.source_evaluator.curriculum_source_pool import (
                register_curriculum_source,
            )

            register_curriculum_source(
                f"https://{host}/",
                f"exa_domain:{host}",
                category=verdict.classification.lower(),
                trust_score=0.88 if verdict.classification == "OFFICIAL_DOCS" else 0.8,
                status="lite_approved",
                reason=verdict.reason,
            )
        except Exception:
            pass
        trace(f"EXA domain KEEP | {host} | {verdict.classification}")
    else:
        trace(f"EXA domain REJECT | {host} | {verdict.classification}")
    return verdict


def classify_exa_domains_batch_with_flash_lite(
    hosts: list[str],
) -> list[DomainAuthorityItem]:
    """One Flash Lite call for the unknown host pack. Official hosts skipped."""
    unique = _unique_hosts(list(hosts), cap=16)
    if not unique:
        return []
    known: list[DomainAuthorityItem] = []
    unknown: list[str] = []
    for host in unique:
        if is_official_docs_host(host):
            known.append(
                DomainAuthorityItem(
                    domain=host,
                    classification="OFFICIAL_DOCS",
                    general_summary="",
                    reason="Static foundational_docs or previously classified.",
                )
            )
        else:
            unknown.append(host)
    if not unknown:
        return known

    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    listed = "\n".join(f"- {h}" for h in unknown)
    user = (
        f"Hostnames to classify ({len(unknown)}):\n{listed}\n\n"
        "Return BatchDomainAuthorityResponse JSON with one item per hostname."
    )
    try:
        out = run_gemini_lite_structured(
            _BATCH_AUTHORITY_SYSTEM,
            user,
            f"exa_domain_batch:{','.join(unknown)[:80]}",
            BatchDomainAuthorityResponse,
            "exa / domain_authority_batch",
        )
    except Exception as exc:
        trace(f"EXA domain batch classify ⊘ | n={len(unknown)} | {exc}")
        return known
    if not isinstance(out, BatchDomainAuthorityResponse):
        out = BatchDomainAuthorityResponse.model_validate(out)

    by_host: dict[str, DomainAuthorityItem] = {}
    for raw in out.items or []:
        host = clean_domain_for_exa(raw.domain)
        if host:
            by_host[host] = DomainAuthorityItem(
                domain=host,
                classification=raw.classification,
                general_summary=(raw.general_summary or "").strip()[:400],
                reason=(raw.reason or "").strip()[:400],
            )
    ordered: list[DomainAuthorityItem] = list(known)
    keep_for_registry: list[DomainAuthorityItem] = []
    for host in unknown:
        item = by_host.get(host)
        if item is None:
            trace(f"EXA domain batch ⊘ | missing item | {host}")
            continue
        _commit_authority_item(item)
        ordered.append(item)
        if item.classification in AUTHORITY_KEEP_CLASSES:
            keep_for_registry.append(item)
    if keep_for_registry:
        try:
            from knowledge_engine.services.search.domain_registry import (
                get_domain_registry,
            )

            get_domain_registry().upsert_keep_items(keep_for_registry)
        except Exception as exc:
            trace(f"DOMAIN_REGISTRY upsert skip | {exc}")
    return ordered


def classify_exa_domain_with_flash_lite(domain: str) -> DomainAuthorityVerdict | None:
    """SSOT wrapper: batch of one, then DomainAuthorityVerdict."""
    host = clean_domain_for_exa(domain)
    if not host:
        return None
    items = classify_exa_domains_batch_with_flash_lite([host])
    for item in items:
        if clean_domain_for_exa(item.domain) == host:
            return _item_to_verdict(item)
    return None


def filter_pass1_official_hosts(hosts: list[str] | tuple[str, ...]) -> list[str]:
    """Keep HTTP-live hosts that are OFFICIAL_DOCS (whitelist or batch classifier)."""
    unique = _unique_hosts(list(hosts), cap=16)
    if not unique:
        return []
    official: list[str] = []
    unknown: list[str] = []
    for host in unique:
        if is_official_docs_host(host):
            trace(f"EXA pass 1 authority ✓ | {host} | OFFICIAL_DOCS")
            official.append(host)
        else:
            unknown.append(host)
    if unknown:
        items = classify_exa_domains_batch_with_flash_lite(unknown)
        by_host = {clean_domain_for_exa(it.domain): it for it in items}
        for host in unknown:
            item = by_host.get(host)
            cls = item.classification if item else "unclassified"
            if item and item.classification in PASS1_INCLUDE_CLASSES:
                trace(f"EXA pass 1 authority ✓ | {host} | {cls}")
                official.append(host)
                continue
            trace(f"EXA pass 1 authority ⊘ | {host} | {cls}")
    return official


def absorb_new_exa_hosts(urls: list[str], *, max_new: int = 3) -> None:
    """Classify a few unknown hosts from an Exa result page (one batch)."""
    seen: set[str] = set()
    unknown: list[str] = []
    for url in urls:
        if len(unknown) >= max_new:
            break
        host = clean_domain_for_exa(url)
        if not host or host in seen:
            continue
        seen.add(host)
        matched, _ = match_whitelist(url if "://" in url else f"https://{host}/")
        if matched or is_official_docs_host(host):
            continue
        unknown.append(host)
    if unknown:
        classify_exa_domains_batch_with_flash_lite(unknown)
