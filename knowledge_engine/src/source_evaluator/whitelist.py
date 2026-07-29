"""Базовый реестр авторитетных источников (Whitelist Matrix)."""

from __future__ import annotations

APPROVED_SOURCES_WHITELIST = {
    # 1. Индивидуальные ML/AI практики и исследователи
    "practitioners": [
        "eugeneyan.com",
        "karpathy.ai",
        "lilianweng.github.io",
        "hamel.dev",
        "chiphuyen.com",
        "dataintensive.net",
        "martinfowler.com",
    ],
    # 2. R&D лаборатории и первопроходцы AI/ML
    "ai_pioneers_labs": [
        "research.yandex.com",
        "habr.com/ru/companies/yandex",
        "openai.com/index/research",
        "openai.com/research",
        "anthropic.com/research",
        "openreview.net",
        "deepmind.google",
        "deepmind.google/research",
    ],
    # 3. Инфраструктура, highload и системный дизайн
    "engineering_blogs": [
        "blog.cloudflare.com",
        "cloudflare.com",
        "uber.com/blog/engineering",
        "netflixtechblog.com",
        "bytebytego.com",
        "blog.bytebytego.com",
        "langchain.com/blog",
        "blog.llamaindex.ai",
        "eng.uber.com",
        "stripe.com/blog",
    ],
    # 4. Эталонные спецификации и стандартные платформы
    "foundational_docs": [
        "developer.mozilla.org",
        "learn.microsoft.com",
        "docs.aws.amazon.com",
        "aws.amazon.com/architecture",
    ],
}
