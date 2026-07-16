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
