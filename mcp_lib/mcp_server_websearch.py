import sys
import re
import asyncio
import traceback
import urllib.parse
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
import yaml
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP, Context

from models import SearchInput, UrlInput, PythonCodeOutput

mcp = FastMCP("websearch")

# Load parameters from default_mcp_config.yaml
_cfg_path = Path(__file__).with_name("default_mcp_config.yaml")
with _cfg_path.open("r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f).get("websearch", {})

SEARCH_RPM       = int(_cfg.get("search_requests_per_minute", 30))
FETCH_RPM        = int(_cfg.get("fetch_requests_per_minute",  20))
REQUEST_TIMEOUT  = float(_cfg.get("request_timeout",          30.0))
MAX_CONTENT_LEN  = int(_cfg.get("max_content_length",         8000))
DDG_URL          = str(_cfg.get("ddg_url", "https://html.duckduckgo.com/html"))


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

    def _format(self, results: List[SearchResult]) -> str:
        if not results:
            return "No results found."
        lines = [f"Found {len(results)} results:\n"]
        for r in results:
            lines += [f"{r.position}. {r.title}", f"   URL: {r.link}", f"   {r.snippet}", ""]
        return "\n".join(lines)

    async def search(self, query: str, ctx: Context, max_results: int = 10) -> List[SearchResult]:
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
        except Exception as e:
            await ctx.error(f"Search error: {e}")
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
                response = await client.get(url, headers=self.HEADERS, follow_redirects=True, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = re.sub(r"\s+", " ", soup.get_text()).strip()
            if len(text) > MAX_CONTENT_LEN:
                text = text[:MAX_CONTENT_LEN] + "... [truncated]"
            return text
        except Exception as e:
            await ctx.error(f"Fetch error: {e}")
            return f"Error: {e}"


# --- Tool instances ---

searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()


# --- Tools ---

@mcp.tool()
async def duckduckgo_search_results(input: SearchInput, ctx: Context) -> PythonCodeOutput:
    """Search DuckDuckGo and return formatted results."""
    results = await searcher.search(input.query, ctx, input.max_results)
    return PythonCodeOutput(result=searcher._format(results))


@mcp.tool()
async def download_raw_html_from_url(input: UrlInput, ctx: Context) -> PythonCodeOutput:
    """Fetch and return clean text content from a webpage URL."""
    return PythonCodeOutput(result=await fetcher.fetch(input.url, ctx))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run()
    else:
        mcp.run(transport="stdio")
