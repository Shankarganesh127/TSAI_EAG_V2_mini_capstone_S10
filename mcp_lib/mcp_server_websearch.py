import asyncio
import hashlib
import re
import sys
import traceback
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import Context, FastMCP

try:
    from .models import SearchInput, UrlInput, PythonCodeOutput
    from .server_utils import load_mcp_config, run_mcp_server
except ImportError:
    from models import SearchInput, UrlInput, PythonCodeOutput
    from server_utils import load_mcp_config, run_mcp_server

mcp = FastMCP("websearch")

_cfg_path = Path(__file__).with_name("default_mcp_config.yaml")
_project_config = load_mcp_config(_cfg_path)
_cfg = _project_config.get("websearch", {})

SEARCH_RPM       = int(_cfg.get("search_requests_per_minute", 30))
FETCH_RPM        = int(_cfg.get("fetch_requests_per_minute",  20))
REQUEST_TIMEOUT  = float(_cfg.get("request_timeout",          30.0))
MAX_CONTENT_LEN  = int(_cfg.get("max_content_length",         8000))
DDG_URL          = str(_cfg.get("ddg_url", "https://html.duckduckgo.com/html"))
WEB_DOCUMENT_DIR = (
    Path(__file__).parent
    / _project_config.get("documents", {}).get("documents_dir", "Documents")
    / "web"
)


def _store_web_document(kind: str, key: str, content: str) -> Path:
    """Persist web-derived content in the local document corpus."""
    WEB_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    path = WEB_DOCUMENT_DIR / f"{kind}_{digest}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _refresh_document_index() -> dict:
    try:
        from .mcp_server_documents import process_documents
    except ImportError:
        from mcp_server_documents import process_documents
    return process_documents()


async def _persist_and_index_web_content(
    kind: str,
    key: str,
    content: str,
    ctx: Context,
) -> None:
    """Store web content and refresh RAG without failing the web request."""
    _store_web_document(kind, key, content)
    try:
        await asyncio.to_thread(_refresh_document_index)
    except Exception as exc:
        await ctx.warning(f"Web content saved, but vector indexing failed: {exc}")


# --- Rate limiter ---

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.rpm = requests_per_minute
        self.requests: list = []

    async def acquire(self):
        now = datetime.now()
        self.requests = [r for r in self.requests if now - r < timedelta(minutes=1)]
        if len(self.requests) >= self.rpm:
            wait = 60 - (now - self.requests[0]).total_seconds()
            if wait > 0:
                await asyncio.sleep(wait)
        self.requests.append(now)


# --- Search result model ---

@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


# --- DuckDuckGo searcher ---

class DuckDuckGoSearcher:
    BASE_URL = DDG_URL
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter(SEARCH_RPM)

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        if not results:
            return "No results found."
        lines = [f"Found {len(results)} results:\n"]
        for r in results:
            lines += [f"{r.position}. {r.title}", f"   URL: {r.link}", f"   {r.snippet}", ""]
        return "\n".join(lines)

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> list[SearchResult]:
        await self.rate_limiter.acquire()
        await ctx.info(f"Searching DuckDuckGo for: {query}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL,
                    data={"q": query, "b": "", "kl": ""},
                    headers=self.HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for item in soup.select(".result"):
                title_elem = item.select_one(".result__title a")
                if not title_elem:
                    continue
                link = title_elem.get("href", "")
                if "y.js" in link:
                    continue
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])
                snippet_elem = item.select_one(".result__snippet")
                results.append(SearchResult(
                    title=title_elem.get_text(strip=True),
                    link=link,
                    snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                    position=len(results) + 1,
                ))
                if len(results) >= max_results:
                    break

            await ctx.info(f"Found {len(results)} results")
            return results
        except Exception as exc:
            await ctx.error(f"Search error: {exc}")
            traceback.print_exc(file=sys.stderr)
            return []


# --- Web content fetcher ---

class WebContentFetcher:
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def __init__(self):
        self.rate_limiter = RateLimiter(FETCH_RPM)

    async def fetch(self, url: str, ctx: Context) -> str:
        await self.rate_limiter.acquire()
        await ctx.info(f"Fetching: {url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self.HEADERS,
                    follow_redirects=True,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text()).strip()
            if len(text) > MAX_CONTENT_LEN:
                text = text[:MAX_CONTENT_LEN] + "... [truncated]"
            return text
        except Exception as exc:
            await ctx.error(f"Fetch error: {exc}")
            return f"Error: {exc}"


# --- Tool instances ---

searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()


# --- Tools ---

@mcp.tool()
async def duckduckgo_search_results(input: SearchInput, ctx: Context) -> PythonCodeOutput:
    """Search DuckDuckGo and return formatted results."""
    results = await searcher.search(input.query, ctx, input.max_results)
    formatted = searcher.format_results(results)
    if results:
        await _persist_and_index_web_content(
            "search",
            input.query,
            f"# Web search: {input.query}\n\n{formatted}",
            ctx,
        )
    return PythonCodeOutput(result=formatted)


@mcp.tool()
async def download_raw_html_from_url(input: UrlInput, ctx: Context) -> PythonCodeOutput:
    """Fetch and return clean text content from a webpage URL."""
    content = await fetcher.fetch(input.url, ctx)
    if content and not content.startswith("Error:"):
        await _persist_and_index_web_content(
            "page",
            input.url,
            f"# Web page\n\nSource: {input.url}\n\n{content}",
            ctx,
        )
    return PythonCodeOutput(result=content)


if __name__ == "__main__":
    run_mcp_server(mcp)
