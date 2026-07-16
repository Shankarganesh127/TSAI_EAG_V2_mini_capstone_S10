import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

_log = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class SubAgentError(RuntimeError):
    """Raised when a configured LLM cannot produce a valid stage result."""


class BaseSubAgent(ABC, Generic[InputT, OutputT]):
    """
    Base for all internal stage sub-agents.
    Each subclass defines PROMPT_TEMPLATE, output_model, and _default_output.
    The LLM is called only when llm_client is set; otherwise the default
    is returned so the pipeline works without any LLM configured.
    """

    PROMPT_TEMPLATE: str = ""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    @property
    @abstractmethod
    def output_model(self) -> Type[OutputT]: ...

    async def run(self, input_data: InputT) -> OutputT:
        if self.llm_client is None:
            return self._default_output(input_data)
        prompt = self._build_prompt(input_data)
        return await self._call(prompt, input_data)

    async def _call(self, prompt: str, input_data: InputT) -> OutputT:
        """Call the configured LLM and return a validated result."""
        stage = self.__class__.__name__
        _log.debug("[%s] PROMPT:\n%s", stage, prompt)
        try:
            raw = await asyncio.to_thread(self.llm_client.chat, prompt)
            _log.debug("[%s] RESPONSE:\n%s", stage, raw)
            return self._parse_output(raw, input_data)
        except Exception as exc:
            _log.error("[%s] LLM call failed: %s", stage, exc)
            raise SubAgentError(f"{stage} failed: {exc}") from exc

    def _build_prompt(self, input_data: InputT) -> str:
        return self.PROMPT_TEMPLATE.format(**input_data.model_dump())

    def _parse_output(self, raw: str, input_data: InputT) -> OutputT:
        """Extract JSON from LLM response and validate into the output model."""
        try:
            text = raw.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return self.output_model.model_validate_json(text)
        except Exception as exc:
            _log.warning("%s parse failed (raw=%r): %s", self.__class__.__name__, raw[:200], exc)
            raise SubAgentError(
                f"{self.__class__.__name__} returned invalid structured output"
            ) from exc

    @abstractmethod
    def _default_output(self, input_data: InputT) -> OutputT: ...
