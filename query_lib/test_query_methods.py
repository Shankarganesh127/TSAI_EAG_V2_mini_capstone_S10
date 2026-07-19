from query_lib.query_functions import dissect_user_query


def test_dissect_user_query_preserves_context():
    result = dissect_user_query(
        raw_query="Find current F1 standings and save to Google Sheet",
        conversation_history=["User wants an MCP-based architecture"],
        memory_results=["User prefers FAISS"],
        known_info={"preferred_memory_store": "FAISS"},
    )

    assert result.normalized_query == "Find current F1 standings and save to Google Sheet"
    assert result.conversation_history == ["User wants an MCP-based architecture"]
    assert result.relevant_memory == ["User prefers FAISS"]
    assert result.known_information["preferred_memory_store"] == "FAISS"


def test_current_time_queries_are_classified_as_volatile():
    from main import is_current_time_query

    assert is_current_time_query("What is the current time in my location?")
    assert is_current_time_query("Show the local time in Tokyo")
    assert not is_current_time_query("Explain how time zones work")