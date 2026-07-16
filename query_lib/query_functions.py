import asyncio
import json

try:
    from query_lib.query_context import QueryDissection, Complexity, AgentRouting, EnrichedQuery
except ModuleNotFoundError:
    from query_context import QueryDissection, Complexity, AgentRouting, EnrichedQuery


_ENRICHMENT_PROMPT = """\
You are a query pre-processor for an AI agent. Given a raw user input, your job is to:
1. Fix grammar, spelling, and punctuation.
2. Identify the primary intent and key entities.
3. Break the goal into clear sub-goals.
4. List any implicit assumptions.
5. Write an elaborated, detailed query that will help the agent understand exactly what the user needs.

Raw input: {raw_query}

Respond ONLY with a JSON object in this exact format:
{{
  "corrected_query": "<grammar-corrected version of the raw input>",
  "intent": "<primary intent: search | compute | summarize | generate | compare | other>",
  "entities": ["<entity1>", "<entity2>"],
  "sub_goals": ["<sub-goal 1>", "<sub-goal 2>"],
  "user_needs": "<one sentence describing what the user truly wants>",
  "assumptions": ["<assumption 1>", "<assumption 2>"],
  "elaborated_query": "<full, detailed, unambiguous query for the agent>"
}}"""


def _fallback_enrichment(raw_query: str) -> EnrichedQuery:
    cleaned = raw_query.strip()
    return EnrichedQuery(
        raw_query=raw_query,
        corrected_query=cleaned,
        intent="unknown",
        entities=[],
        sub_goals=[],
        user_needs=cleaned,
        assumptions=[],
        elaborated_query=cleaned,
    )


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(text)


async def enrich_query(raw_query: str, llm_client=None) -> EnrichedQuery:
    """
    Use the LLM to correct, elaborate, and fully specify a raw user query.
    Falls back to a minimal enrichment when no LLM is available.
    """
    if not llm_client:
        return _fallback_enrichment(raw_query)

    prompt = _ENRICHMENT_PROMPT.format(raw_query=raw_query)
    try:
        raw = await asyncio.to_thread(llm_client.chat, prompt)
        data = _extract_json_object(raw)
        return EnrichedQuery(raw_query=raw_query, **data)
    except Exception:
        return _fallback_enrichment(raw_query)


def dissect_user_query(
    raw_query: str,
    conversation_history: list[str] | None = None,
    memory_results: list[str] | None = None,
    known_info: dict | None = None,
) -> QueryDissection:

    conversation_history = conversation_history or []
    memory_results = memory_results or []
    known_info = known_info or {}

    return QueryDissection(
        raw_query=raw_query,
        normalized_query=raw_query.strip(),
        intent="unknown",
        main_goal="extract_user_goal",
        conversation_history=conversation_history,
        relevant_memory=memory_results,
        known_information=known_info,
        complexity=Complexity(
            level="medium",
            requires_tools=False,
            requires_memory=True,
            requires_validation=True
        ),
        agent_routing=AgentRouting(
            perception_agent=True,
            memory_agent=True,
            planning_agent=True,
            action_agent=False,
            validation_agent=True,
            reflection_agent=False
        )
    )
