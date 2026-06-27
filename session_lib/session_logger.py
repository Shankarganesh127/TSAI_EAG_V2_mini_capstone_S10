"""
session_logger.py
-----------------
Single generic session logger for the agent loop.

Log file location:
    session_lib/session_logs/YYYY/MM/DD/<session_id>.json

File format:
    {
        "session_id": "...",
        "query":      "...",
        "started_at": "ISO-timestamp",
        "stages": [
            {
                "stage":     "<stage_name>",
                "timestamp": "ISO-timestamp",
                <...output fields from that stage, flattened directly here...>
            },
            ...
        ]
    }

Usage:
    from session_lib.session_logger import SessionLogger

    logger = SessionLogger(session_id, query)

    # Call log() with a stage name + the raw output dict of that stage.
    # The output dict is merged directly into the log entry — no nesting.
    logger.log("memory_search",      memory_results)
    logger.log("perception_initial", perception_result)
    logger.log("decision",           decision_output)
    logger.log("step_execution",     executor_response)
    logger.log("step_perception",    perception_result)
    logger.log("replan",             decision_output)
    logger.log("conclusion",         step_obj.to_dict())
    logger.log("session_end",        session.state)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from session_lib.config import get_default_session_config
except ModuleNotFoundError:
    from config import get_default_session_config  # direct script execution

_CFG = get_default_session_config()


def _get_log_path(session_id: str, base_dir: str) -> Path:
    now = datetime.now()
    day_dir = Path(base_dir) / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{session_id}.json"


def _load(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            if _CFG.overwrite_corrupt:
                print(f"⚠️  Corrupt session file at {path}. Starting fresh.")
    return {}


def _save(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=_CFG.json_indent, default=str)


class SessionLogger:
    """
    One call logs one stage. Pass the stage output dict directly — it is
    flattened into the log entry alongside 'stage' and 'timestamp'.

        logger = SessionLogger(session_id, query)
        logger.log("perception_initial", perception_result)
        logger.log("step_execution",     executor_response)
    """

    def __init__(self, session_id: str, query: str, base_dir: str | None = None):
        self.session_id = session_id
        self._path = _get_log_path(session_id, base_dir or _CFG.base_dir)

        existing = _load(self._path)
        if existing:
            self._data = existing
        else:
            self._data: dict[str, Any] = {
                "session_id": session_id,
                "query": query,
                "started_at": datetime.now().isoformat(),
                "stages": [],
            }
            _save(self._path, self._data)
        if _CFG.verbose:
            print(f"📂 Session log → {self._path}")

    def log(self, stage: str, output: dict | Any = None) -> None:
        """
        Append one stage entry and persist immediately.

        `output` is the raw output dict of that stage. Its keys are merged
        directly into the log entry (no nested 'data' wrapper).
        """
        entry = {"stage": stage, "timestamp": datetime.now().isoformat()}
        if isinstance(output, dict):
            entry.update(output)
        elif output is not None:
            entry["output"] = output
        self._data["stages"].append(entry)
        _save(self._path, self._data)
        if _CFG.verbose:
            print(f"  ✅ [{stage}]")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uuid

    logger = SessionLogger(session_id=str(uuid.uuid4()), query="What is the capital of France?")

    logger.log("memory_search", {"result_count": 0, "results": []})
    logger.log("perception_initial", {
        "entities": ["France", "capital"],
        "result_requirement": "Name of France's capital city",
        "original_goal_achieved": False,
        "solution_summary": "",
        "confidence": "0.5",
    })
    logger.log("decision", {
        "plan_text": ["Step 0: Search for capital of France"],
        "step_index": 0, "type": "CODE", "description": "web search",
    })
    logger.log("step_execution", {
        "step_index": 0, "status": "completed",
        "result": "Paris is the capital of France.",
    })
    logger.log("step_perception", {
        "step_index": 0,
        "original_goal_achieved": True,
        "solution_summary": "Paris",
        "confidence": "0.99",
        "reasoning": "Direct answer found",
        "local_goal_achieved": True,
        "local_reasoning": "step complete",
        "last_tooluse_summary": "web_search returned Paris",
        "result_requirement": "capital name",
        "entities": ["Paris", "France"],
    })
    logger.log("session_end", {
        "original_goal_achieved": True,
        "final_answer": "Paris",
        "confidence": 0.99,
        "solution_summary": "Paris is the capital of France.",
    })
