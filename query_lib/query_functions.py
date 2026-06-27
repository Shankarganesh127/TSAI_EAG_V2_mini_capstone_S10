try:
    from query_lib.query_context import QueryDissection, Complexity, AgentRouting
except ModuleNotFoundError:
    from query_context import QueryDissection, Complexity, AgentRouting


def dissect_user_query(
    raw_query: str,
    conversation_history: list[str] | None = None,
    memory_results: list[str] | None = None,
    known_info: dict | None = None,
) -> QueryDissection:

    conversation_history = conversation_history or []
    memory_results = memory_results or []
    known_info = known_info or {}

    dissection = QueryDissection(
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

    return dissection