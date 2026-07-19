from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import web_app


async def fake_runtime_factory(**kwargs):
    return SimpleNamespace()


def test_health_and_home_page():
    app = web_app.create_app(fake_runtime_factory)
    with TestClient(app) as client:
        health = client.get("/api/health")
        home = client.get("/")

    assert health.json() == {"status": "ok"}
    assert home.status_code == 200
    assert "Momentum" in home.text


def test_query_endpoint_returns_agent_answer(monkeypatch):
    run_agent = AsyncMock(return_value="Stored answer")
    monkeypatch.setattr(web_app, "run_agent", run_agent)
    app = web_app.create_app(fake_runtime_factory)

    with TestClient(app) as client:
        response = client.post("/api/query", json={"query": "What is stored?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Stored answer",
        "answer_html": "<p>Stored answer</p>\n",
    }
    run_agent.assert_awaited_once()


def test_query_endpoint_attaches_browser_timezone(monkeypatch):
    run_agent = AsyncMock(return_value="It is 21:00 BST")
    monkeypatch.setattr(web_app, "run_agent", run_agent)
    app = web_app.create_app(fake_runtime_factory)

    with TestClient(app) as client:
        response = client.post(
            "/api/query",
            json={
                "query": "What is the current time in my location?",
                "timezone": "Europe/London",
            },
        )

    assert response.status_code == 200
    submitted_query = run_agent.await_args.args[0]
    assert "What is the current time in my location?" in submitted_query
    assert "User location timezone (IANA): Europe/London" in submitted_query

def test_markdown_is_rendered_and_unsafe_html_is_removed():
    rendered = web_app.render_markdown(
        "## Plan\n\n- Perceive\n- Act\n\n<script>alert('x')</script>"
    )

    assert "<h2>Plan</h2>" in rendered
    assert "<li>Perceive</li>" in rendered
    assert "<script>" not in rendered
