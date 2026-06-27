try:
    from query_lib.query_functions import dissect_user_query
except ModuleNotFoundError:
    from query_functions import dissect_user_query

query = "Find current F1 standings and save to Google Sheet"

result = dissect_user_query(
    raw_query=query,
    conversation_history=[
        "User wants MCP-based AI agent architecture",
        "User wants Gmail, Telegram, and Google Drive MCP servers"
    ],
    memory_results=[
        "User prefers FAISS for conversation memory search"
    ],
    known_info={
        "preferred_memory_store": "FAISS",
        "preferred_architecture": [
            "Perception",
            "Decision",
            "Action",
            "Validation"
        ]
    }
)

print(result.model_dump_json(indent=2))