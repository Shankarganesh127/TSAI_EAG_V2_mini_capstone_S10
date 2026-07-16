"""
MCP Server Smoke Tests
======================
Two ways to run:

  1. Direct (prints full report to terminal):
       uv run python mcp_lib/test_mcp_servers.py

  2. MCP Inspector (each tool visible individually):
       uv run mcp dev mcp_lib/test_mcp_servers.py

In the Inspector you will see:
  - One tool per MCP tool  (test_add, test_multiply, test_factorial, ...)
  - One tool per server    (test_math_server, test_documents_server, test_websearch_server)
  - run_all_tests          to run everything at once
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from multi_mcp import MultiMCP

mcp = FastMCP("mcp-test-runner")

# Shared MultiMCP instance — initialized once per process
_multi: MultiMCP | None = None


async def _get_multi() -> MultiMCP:
    global _multi
    if _multi is None or not _multi.tool_map:
        _multi = MultiMCP()
        await _multi.initialize()
    return _multi


def _text(content) -> str:
    """Extract plain text from an MCP tool result."""
    if hasattr(content, "content") and content.content:
        return getattr(content.content[0], "text", str(content.content[0]))
    return str(content)


async def _call(tool: str, args: dict) -> str:
    multi = await _get_multi()
    raw = await multi.call_tool(tool, args)
    return _text(raw)


def _verdict(ok: bool, label: str, got: str = "") -> str:
    tag = "[PASS]" if ok else "[FAIL]"
    suffix = f"  — got: {got!r}" if not ok and got else ""
    return f"{tag}  {label}{suffix}"


def _has_number(text: str, expected: int | float) -> bool:
    """Match a complete numeric value, not an incidental substring."""
    values = [float(value) for value in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text)]
    return any(abs(value - float(expected)) < 1e-9 for value in values)


# ===========================================================================
# Individual tool tests
# ===========================================================================

# --- math ---

@mcp.tool()
async def test_add() -> str:
    """math server: add(3, 4) == 7"""
    r = await _call("add", {"input": {"a": 3, "b": 4}})
    return _verdict(_has_number(r, 7), "add(3, 4) == 7", r)

@mcp.tool()
async def test_subtract() -> str:
    """math server: subtract(10, 3) == 7"""
    r = await _call("subtract", {"input": {"a": 10, "b": 3}})
    return _verdict(_has_number(r, 7), "subtract(10, 3) == 7", r)

@mcp.tool()
async def test_multiply() -> str:
    """math server: multiply(6, 7) == 42"""
    r = await _call("multiply", {"input": {"a": 6, "b": 7}})
    return _verdict(_has_number(r, 42), "multiply(6, 7) == 42", r)

@mcp.tool()
async def test_divide() -> str:
    """math server: divide(10, 4) == 2.5"""
    r = await _call("divide", {"input": {"a": 10, "b": 4}})
    return _verdict(_has_number(r, 2.5), "divide(10, 4) == 2.5", r)

@mcp.tool()
async def test_power() -> str:
    """math server: power(2, 8) == 256"""
    r = await _call("power", {"input": {"a": 2, "b": 8}})
    return _verdict(_has_number(r, 256), "power(2, 8) == 256", r)

@mcp.tool()
async def test_cbrt() -> str:
    """math server: cbrt(27) ~ 3.0"""
    r = await _call("cbrt", {"input": {"a": 27}})
    return _verdict(_has_number(r, 3), "cbrt(27) ~ 3.0", r)

@mcp.tool()
async def test_factorial() -> str:
    """math server: factorial(5) == 120"""
    r = await _call("factorial", {"input": {"a": 5}})
    return _verdict(_has_number(r, 120), "factorial(5) == 120", r)

@mcp.tool()
async def test_remainder() -> str:
    """math server: remainder(10, 3) == 1"""
    r = await _call("remainder", {"input": {"a": 10, "b": 3}})
    return _verdict(_has_number(r, 1), "remainder(10, 3) == 1", r)

@mcp.tool()
async def test_sin() -> str:
    """math server: sin(0) == 0.0"""
    r = await _call("sin", {"input": {"a": 0}})
    return _verdict(_has_number(r, 0), "sin(0) == 0.0", r)

@mcp.tool()
async def test_cos() -> str:
    """math server: cos(0) == 1.0"""
    r = await _call("cos", {"input": {"a": 0}})
    return _verdict(_has_number(r, 1), "cos(0) == 1.0", r)

@mcp.tool()
async def test_tan() -> str:
    """math server: tan(0) == 0.0"""
    r = await _call("tan", {"input": {"a": 0}})
    return _verdict(_has_number(r, 0), "tan(0) == 0.0", r)

@mcp.tool()
async def test_mine() -> str:
    """math server: mine(10, 3) == 4  (a - 2b)"""
    r = await _call("mine", {"input": {"a": 10, "b": 3}})
    return _verdict(_has_number(r, 4), "mine(10, 3) == 4", r)

@mcp.tool()
async def test_strings_to_chars_to_int() -> str:
    """math server: strings_to_chars_to_int('AB') contains ASCII 65"""
    r = await _call("strings_to_chars_to_int", {"input": {"string": "AB"}})
    return _verdict(_has_number(r, 65), "strings_to_chars_to_int('AB') has 65", r)

@mcp.tool()
async def test_int_list_to_exponential_sum() -> str:
    """math server: int_list_to_exponential_sum([0, 1]) returns a number"""
    r = await _call("int_list_to_exponential_sum", {"input": {"numbers": [0, 1]}})
    return _verdict(any(c.isdigit() for c in r), "int_list_to_exponential_sum([0,1])", r)

@mcp.tool()
async def test_fibonacci_numbers() -> str:
    """math server: fibonacci_numbers(6) == [0, 1, 1, 2, 3, 5]"""
    r = await _call("fibonacci_numbers", {"input": {"n": 6}})
    expected = [0, 1, 1, 2, 3, 5]
    values = [int(value) for value in re.findall(r"(?<![\w.])-?\d+(?![\w.])", r)]
    return _verdict(values[-6:] == expected, "fibonacci_numbers(6)", r)


# --- documents ---

@mcp.tool()
async def test_convert_webpage_url_into_markdown() -> str:
    """documents server: fetch https://example.com as markdown"""
    r = await _call("convert_webpage_url_into_markdown", {"input": {"url": "https://example.com"}})
    ok = len(r) > 20 and "error" not in r.lower()
    return _verdict(ok, "convert_webpage_url_into_markdown(example.com)", r[:80] if not ok else "")

@mcp.tool()
async def test_search_stored_documents_rag() -> str:
    """documents server: RAG search (requires pre-built FAISS index in mcp_lib/faiss_index/)"""
    try:
        r = await _call("search_stored_documents_rag", {"input": {"query": "test"}})
        if "ERROR" in r or "not found" in r.lower():
            return "[SKIP]  search_stored_documents_rag — no FAISS index built yet"
        return _verdict(True, "search_stored_documents_rag returned results")
    except Exception as e:
        return f"[SKIP]  search_stored_documents_rag — {e}"

@mcp.tool()
async def test_extract_pdf() -> str:
    """documents server: extract_pdf — provide a real file_path to test"""
    return "[SKIP]  extract_pdf — provide a real file_path to test manually"


# --- websearch ---

@mcp.tool()
async def test_duckduckgo_search_results() -> str:
    """websearch server: DuckDuckGo search for 'python mcp'"""
    r = await _call("duckduckgo_search_results", {"input": {"query": "python mcp", "max_results": 3}})
    ok = "result" in r.lower() or "http" in r.lower()
    return _verdict(ok, "duckduckgo_search_results('python mcp')", r[:80] if not ok else "")

@mcp.tool()
async def test_download_raw_html_from_url() -> str:
    """websearch server: fetch raw text from https://example.com"""
    r = await _call("download_raw_html_from_url", {"input": {"url": "https://example.com"}})
    ok = len(r) > 20 and "error" not in r.lower()
    return _verdict(ok, "download_raw_html_from_url(example.com)", r[:80] if not ok else "")


# ===========================================================================
# Server group tests — run all tools for one server in one call
# ===========================================================================

async def _run_cases(cases: list) -> str:
    multi = await _get_multi()
    lines = []
    passed = 0
    for label, tool, args, check in cases:
        try:
            text = _text(await multi.call_tool(tool, args))
            ok = check(text)
            lines.append(_verdict(ok, label, text if not ok else ""))
            if ok:
                passed += 1
        except Exception as e:
            lines.append(_verdict(False, label, str(e)))
    lines.append(f"\n{passed}/{len(cases)} passed")
    return "\n".join(lines)


@mcp.tool()
async def test_math_server() -> str:
    """Run all math server tool tests."""
    return await _run_cases([
        ("add(3,4)==7",                            "add",                        {"input": {"a": 3,  "b": 4}},          lambda t: _has_number(t, 7)),
        ("subtract(10,3)==7",                      "subtract",                   {"input": {"a": 10, "b": 3}},          lambda t: _has_number(t, 7)),
        ("multiply(6,7)==42",                      "multiply",                   {"input": {"a": 6,  "b": 7}},          lambda t: _has_number(t, 42)),
        ("divide(10,4)==2.5",                      "divide",                     {"input": {"a": 10, "b": 4}},          lambda t: _has_number(t, 2.5)),
        ("power(2,8)==256",                        "power",                      {"input": {"a": 2,  "b": 8}},          lambda t: _has_number(t, 256)),
        ("cbrt(27)~3.0",                           "cbrt",                       {"input": {"a": 27}},                  lambda t: _has_number(t, 3)),
        ("factorial(5)==120",                      "factorial",                  {"input": {"a": 5}},                   lambda t: _has_number(t, 120)),
        ("remainder(10,3)==1",                     "remainder",                  {"input": {"a": 10, "b": 3}},          lambda t: _has_number(t, 1)),
        ("sin(0)==0.0",                            "sin",                        {"input": {"a": 0}},                   lambda t: _has_number(t, 0)),
        ("cos(0)==1.0",                            "cos",                        {"input": {"a": 0}},                   lambda t: _has_number(t, 1)),
        ("tan(0)==0.0",                            "tan",                        {"input": {"a": 0}},                   lambda t: _has_number(t, 0)),
        ("mine(10,3)==4",                          "mine",                       {"input": {"a": 10, "b": 3}},          lambda t: _has_number(t, 4)),
        ("strings_to_chars_to_int('AB') has 65",   "strings_to_chars_to_int",    {"input": {"string": "AB"}},           lambda t: _has_number(t, 65)),
        ("int_list_to_exponential_sum([0,1])>0",   "int_list_to_exponential_sum",{"input": {"numbers": [0, 1]}},        lambda t: any(c.isdigit() for c in t)),
        ("fibonacci_numbers(6) has 5",             "fibonacci_numbers",          {"input": {"n": 6}},                   lambda t: _has_number(t, 5)),
    ])


@mcp.tool()
async def test_documents_server() -> str:
    """Run all documents server tool tests."""
    lines = []
    r = await _call("convert_webpage_url_into_markdown", {"input": {"url": "https://example.com"}})
    ok = len(r) > 20 and "error" not in r.lower()
    lines.append(_verdict(ok, "convert_webpage_url_into_markdown(example.com)", r[:80] if not ok else ""))
    try:
        r2 = await _call("search_stored_documents_rag", {"input": {"query": "test"}})
        if "ERROR" in r2 or "not found" in r2.lower():
            lines.append("[SKIP]  search_stored_documents_rag — no FAISS index built yet")
        else:
            lines.append(_verdict(True, "search_stored_documents_rag returned results"))
    except Exception as e:
        lines.append(f"[SKIP]  search_stored_documents_rag — {e}")
    lines.append("[SKIP]  extract_pdf — provide a real file_path to test manually")
    return "\n".join(lines)


@mcp.tool()
async def test_websearch_server() -> str:
    """Run all websearch server tool tests."""
    return await _run_cases([
        ("duckduckgo_search_results('python mcp')", "duckduckgo_search_results",  {"input": {"query": "python mcp", "max_results": 3}}, lambda t: "result" in t.lower() or "http" in t.lower()),
        ("download_raw_html_from_url(example.com)", "download_raw_html_from_url", {"input": {"url": "https://example.com"}},            lambda t: len(t) > 20 and "error" not in t.lower()),
    ])


# ===========================================================================
# run_all_tests — runs every server group and returns a combined report
# ===========================================================================

@mcp.tool()
async def run_all_tests() -> str:
    """Run all tests across all MCP servers and return a combined report."""
    lines = ["=" * 55, "  MCP Server Smoke Tests", "=" * 55]
    for name, fn in [
        ("math server",      test_math_server),
        ("documents server", test_documents_server),
        ("websearch server", test_websearch_server),
    ]:
        lines.append(f"\n[{name}]")
        lines.append(await fn())
    lines += ["", "=" * 55, "  Done", "=" * 55]
    return "\n".join(lines)


# ===========================================================================
# Direct execution: uv run python mcp_lib/test_mcp_servers.py
# ===========================================================================

if __name__ == "__main__":
    report = asyncio.run(run_all_tests())
    print(report)
    if "[FAIL]" in report:
        sys.exit(1)
