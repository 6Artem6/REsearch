"""GitHub Git Trees API ingest — без полного zip, с fallback на archive.zip."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from knowledge_engine.services.article_ingestion.ast_code_chunker import (
    EXTENSION_TO_LANGUAGE,
)

logger = logging.getLogger(__name__)

GITHUB_TREES_API = "https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
GITHUB_ARCHIVE_ZIP = "https://github.com/{owner}/{repo}/archive/{ref}.zip"

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        "vendor",
        "venv",
        ".venv",
        "dist",
        "build",
        "__pycache__",
        ".idea",
        ".vscode",
    }
)
_EXTRA_TEXT_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".rst"})
_REPO_PAGE_SEGMENTS = frozenset(
    {
        "blob",
        "issues",
        "pull",
        "pulls",
        "releases",
        "actions",
        "wiki",
        "compare",
        "settings",
        "security",
        "projects",
        "discussions",
    }
)
_API_FALLBACK_STATUSES = frozenset({401, 403, 404})
_USER_AGENT = "REsearch-KnowledgeEngine/1.0 (GitHub Trees ingest)"
_DEFAULT_REF = "main"
_HTTP_TIMEOUT = 20.0


class GitHubTreesApiError(Exception):
    """Ошибка Trees API: вызывающий должен уйти в zip / HTML fallback."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _allowed_suffixes() -> frozenset[str]:
    return frozenset(EXTENSION_TO_LANGUAGE.keys()) | _EXTRA_TEXT_SUFFIXES


def is_github_tree_ingest_url(url: str) -> bool:
    """True для корня репозитория или ``/tree/{ref}``, не для blob/issues."""
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "github.com":
        return False
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        return False
    if len(parts) == 2:
        return True
    return parts[2].lower() in ("tree", "commit")


def format_repo_corpus(files: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in files:
        path = str(item.get("path") or "").strip()
        body = str(item.get("content") or "").strip()
        if not path or not body:
            continue
        parts.append(f"### {path}\n{body}")
    return "\n\n".join(parts)


def parse_github_url(url: str) -> tuple[str, str, str]:
    """``(owner, repo, ref)``; ref по умолчанию ``main``."""
    parsed = urlparse((url or "").strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "github.com":
        raise ValueError(f"not a github.com URL: {url!r}")
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"github URL missing owner/repo: {url!r}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    ref = _DEFAULT_REF
    if len(parts) >= 4 and parts[2].lower() in ("tree", "commit", "blob"):
        ref = parts[3] or _DEFAULT_REF
    return owner, repo, ref


def parse_github_blob_url(url: str) -> tuple[str, str, str, str] | None:
    """``(owner, repo, ref, blob_path)`` for ``/blob/`` URLs; else ``None``."""
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "github.com":
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 5 or parts[2].lower() != "blob":
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    ref = parts[3]
    blob_path = "/".join(parts[4:])
    if not owner or not repo or not ref or not blob_path:
        return None
    return owner, repo, ref, blob_path


def path_is_skipped(path: str) -> bool:
    parts = Path((path or "").replace("\\", "/")).parts
    return any(part in SKIP_DIR_NAMES for part in parts)


def suffix_is_allowed(path: str) -> bool:
    suffix = Path(path or "").suffix.lower()
    return bool(suffix) and suffix in _allowed_suffixes()


class GitHubTreeLoader:
    """Загрузка текстовых файлов репозитория через Git Trees API."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        token: str | None = None,
        max_file_size: int | None = None,
        timeout_sec: float = _HTTP_TIMEOUT,
    ) -> None:
        from knowledge_engine import config as ke_config

        self._owns_client = client is None
        self._client = client
        self._token = (
            token if token is not None else ke_config.GITHUB_TOKEN
        )
        self._max_file_size = (
            int(max_file_size)
            if max_file_size is not None
            else int(ke_config.MAX_GITHUB_FILE_SIZE_BYTES)
        )
        self._timeout = timeout_sec
        self.used_zip_fallback = False

    def _headers(self, *, json_api: bool) -> dict[str, str]:
        headers = {"User-Agent": _USER_AGENT}
        if json_api:
            headers["Accept"] = "application/vnd.github+json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _client_or_ephemeral(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self._timeout, follow_redirects=True)

    def _get(self, url: str, *, json_api: bool) -> httpx.Response:
        client = self._client_or_ephemeral()
        close = self._client is None
        try:
            try:
                return client.get(url, headers=self._headers(json_api=json_api))
            except httpx.TimeoutException as exc:
                raise GitHubTreesApiError(
                    f"timeout fetching {url[:120]}", status_code=None
                ) from exc
            except httpx.HTTPError as exc:
                raise GitHubTreesApiError(
                    f"HTTP error fetching {url[:120]}: {exc}", status_code=None
                ) from exc
        finally:
            if close:
                client.close()

    @staticmethod
    def parse_github_url(url: str) -> tuple[str, str, str]:
        return parse_github_url(url)

    def fetch_tree_structure(
        self, owner: str, repo: str, ref: str = _DEFAULT_REF
    ) -> list[dict[str, Any]]:
        api_url = GITHUB_TREES_API.format(
            owner=quote(owner, safe=""),
            repo=quote(repo, safe=""),
            ref=quote(ref or _DEFAULT_REF, safe=""),
        )
        api_url = f"{api_url}?recursive=1"
        resp = self._get(api_url, json_api=True)
        if resp.status_code in _API_FALLBACK_STATUSES:
            raise GitHubTreesApiError(
                f"GitHub Trees API HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise GitHubTreesApiError(
                f"GitHub Trees API HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        try:
            payload = resp.json()
        except Exception as exc:
            raise GitHubTreesApiError("GitHub Trees API returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise GitHubTreesApiError("GitHub Trees API payload is not an object")
        if payload.get("truncated"):
            raise GitHubTreesApiError("GitHub Trees API tree is truncated")
        tree = payload.get("tree")
        if not isinstance(tree, list):
            raise GitHubTreesApiError("GitHub Trees API missing tree[]")
        return [item for item in tree if isinstance(item, dict)]

    def fetch_tree(
        self, owner: str, repo: str, ref: str = _DEFAULT_REF
    ) -> list[dict[str, Any]]:
        """Alias for ``fetch_tree_structure`` (one recursive Trees API call)."""
        return self.fetch_tree_structure(owner, repo, ref)

    def filter_tree_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for item in items:
            if str(item.get("type") or "") != "blob":
                continue
            path = str(item.get("path") or "")
            if not path or path_is_skipped(path) or not suffix_is_allowed(path):
                continue
            size_raw = item.get("size")
            try:
                size = int(size_raw) if size_raw is not None else 0
            except (TypeError, ValueError):
                size = 0
            if size > self._max_file_size:
                continue
            kept.append(item)
        return kept

    def fetch_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> str | None:
        raw_url = GITHUB_RAW_URL.format(
            owner=quote(owner, safe=""),
            repo=quote(repo, safe=""),
            ref=quote(ref or _DEFAULT_REF, safe=""),
            path=quote(path.lstrip("/"), safe="/"),
        )
        resp = self._get(raw_url, json_api=False)
        if resp.status_code in _API_FALLBACK_STATUSES:
            raise GitHubTreesApiError(
                f"GitHub raw content HTTP {resp.status_code} for {path}",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            logger.warning("skip GitHub file %s (HTTP %s)", path, resp.status_code)
            return None
        text = resp.text
        if "\x00" in text:
            return None
        return text

    def _load_via_trees(
        self, owner: str, repo: str, ref: str
    ) -> list[dict[str, Any]]:
        items = self.filter_tree_items(self.fetch_tree_structure(owner, repo, ref))
        out: list[dict[str, Any]] = []
        for item in items:
            path = str(item.get("path") or "")
            body = self.fetch_file_content(owner, repo, path, ref)
            if not body or not body.strip():
                continue
            out.append(
                {
                    "path": path,
                    "content": body,
                    "size": int(item.get("size") or len(body.encode("utf-8"))),
                    "sha": str(item.get("sha") or ""),
                    "url": f"https://github.com/{owner}/{repo}/blob/{ref}/{path}",
                }
            )
        return out

    def _load_via_zip(
        self, owner: str, repo: str, ref: str
    ) -> list[dict[str, Any]]:
        zip_url = GITHUB_ARCHIVE_ZIP.format(owner=owner, repo=repo, ref=ref)
        resp = self._get(zip_url, json_api=False)
        if resp.status_code >= 400:
            raise GitHubTreesApiError(
                f"GitHub zip archive HTTP {resp.status_code}",
                status_code=resp.status_code,
            )
        try:
            archive = zipfile.ZipFile(io.BytesIO(resp.content))
        except zipfile.BadZipFile as exc:
            raise GitHubTreesApiError("GitHub archive is not a zip") from exc
        out: list[dict[str, Any]] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            inner = info.filename.replace("\\", "/")
            # github zip: "{repo}-{ref}/path"
            rel = inner.split("/", 1)[1] if "/" in inner else inner
            if path_is_skipped(rel) or not suffix_is_allowed(rel):
                continue
            if info.file_size > self._max_file_size:
                continue
            try:
                raw = archive.read(info)
            except Exception:
                continue
            if b"\x00" in raw[:1024]:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            if not text.strip():
                continue
            out.append(
                {
                    "path": rel,
                    "content": text,
                    "size": int(info.file_size),
                    "sha": "",
                    "url": f"https://github.com/{owner}/{repo}/blob/{ref}/{rel}",
                }
            )
        return out

    def load_repository_files(self, url: str) -> list[dict[str, Any]]:
        owner, repo, ref = self.parse_github_url(url)
        self.used_zip_fallback = False
        try:
            return self._load_via_trees(owner, repo, ref)
        except Exception as exc:
            logger.warning(
                "GitHub Trees API failed (%s: %s); fallback to zip archive",
                type(exc).__name__,
                exc,
            )
            self.used_zip_fallback = True
            return self._load_via_zip(owner, repo, ref)


def maybe_fetch_github_repo_corpus(url: str) -> tuple[str, str] | None:
    """Opt-in: корпус файлов репозитория или ``None`` (обычный HTML/raw fetch)."""
    from knowledge_engine import config as ke_config

    if not ke_config.USE_GITHUB_TREES_API:
        return None
    if not is_github_tree_ingest_url(url):
        return None
    try:
        loader = GitHubTreeLoader()
        files = loader.load_repository_files(url)
        text = format_repo_corpus(files)
        if not (text or "").strip():
            return None
        method = "github_zip" if loader.used_zip_fallback else "github_trees"
        return text, method
    except Exception as exc:
        logger.warning(
            "GitHub repo ingest failed (%s: %s); fallback to standard fetch",
            type(exc).__name__,
            exc,
        )
        return None
