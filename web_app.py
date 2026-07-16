import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, Callable

import uvicorn
import bleach
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

from main import RuntimeServices, create_runtime, run_agent

WEB_DIR = Path(__file__).with_name("web")
STATIC_DIR = WEB_DIR / "static"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)


class QueryResponse(BaseModel):
    answer: str
    answer_html: str


MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table")
ALLOWED_TAGS = {
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4",
    "h5", "h6", "hr", "li", "ol", "p", "pre", "strong", "table", "tbody",
    "td", "th", "thead", "tr", "ul",
}


def render_markdown(text: str) -> str:
    """Render agent Markdown into safe HTML for the chat window."""
    rendered = MARKDOWN.render(text)
    return bleach.clean(
        rendered,
        tags=ALLOWED_TAGS,
        attributes={"a": ["href", "title", "target", "rel"]},
        protocols={"http", "https", "mailto"},
        strip=True,
    )


def enable_background_logging() -> None:
    """Keep rotating file logs while removing console output."""
    root_logger = logging.getLogger()
    root_logger.handlers = [
        handler
        for handler in root_logger.handlers
        if not isinstance(handler, logging.StreamHandler)
        or isinstance(handler, logging.FileHandler)
    ]


def create_lifespan(
    runtime_factory: Callable[..., Awaitable[RuntimeServices]],
):
    """Create a lifespan handler with an injectable runtime factory."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = await runtime_factory(background_logs=True)
        enable_background_logging()
        app.state.services = services
        app.state.query_lock = asyncio.Lock()
        yield

    return lifespan


def create_app(
    runtime_factory: Callable[..., Awaitable[RuntimeServices]] = create_runtime,
) -> FastAPI:
    """Create the local query web application."""
    app = FastAPI(
        title="Local RAG Agent",
        lifespan=create_lifespan(runtime_factory),
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.post("/api/query", response_model=QueryResponse)
    async def submit_query(payload: QueryRequest, request: Request) -> QueryResponse:
        services: RuntimeServices = request.app.state.services
        async with request.app.state.query_lock:
            answer = await run_agent(payload.query.strip(), services)
        if answer is None:
            raise HTTPException(
                status_code=500,
                detail="The agent could not complete this request. Check the background log.",
            )
        return QueryResponse(answer=answer, answer_html=render_markdown(answer))

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
