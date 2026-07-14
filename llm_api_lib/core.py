from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion, embedding
import yaml


@dataclass
class LLMConfig:
    provider: str
    model: str
    embedding_model: str
    api_key: str | None
    base_url: str | None
    temperature: float


def _load_default_llm_yaml() -> dict:
    """Load llm defaults from llm_api_lib/default_config.yaml."""
    config_path = Path(__file__).with_name("default_config.yaml")
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file) or {}
    return yaml_config.get("llm", {})


_DEFAULT_LLM = _load_default_llm_yaml()
_DEFAULT_PROVIDER = str(_DEFAULT_LLM.get("provider", "openai")).strip().lower()
_DEFAULT_TEMPERATURE = float(_DEFAULT_LLM.get("temperature", 0.2))
_DEFAULT_LOCAL_MODEL = str(_DEFAULT_LLM.get("model", "openai/llama-3.1-8b-instruct"))
_DEFAULT_LOCAL_BASE_URL = str(_DEFAULT_LLM.get("base_url", "http://localhost:8000/v1"))


def _resolve_provider_config(provider: str, temperature: float) -> LLMConfig:
    provider = provider.strip().lower()

    if provider == "openai":
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=temperature,
        )

    if provider in {"claude", "anthropic"}:
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20241022"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=None,
            temperature=temperature,
        )

    if provider in {"gemini", "google"}:
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            base_url=None,
            temperature=temperature,
        )

    if provider == "local":
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", _DEFAULT_LOCAL_MODEL),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                os.getenv("LLM_MODEL", _DEFAULT_LOCAL_MODEL),
            ),
            api_key=os.getenv("LOCAL_LLM_API_KEY", "dummy"),
            base_url=os.getenv("LOCAL_LLM_BASE_URL", _DEFAULT_LOCAL_BASE_URL),
            temperature=temperature,
        )

    raise ValueError("Unsupported LLM_PROVIDER. Use: openai, gemini, claude, local")


def load_config() -> LLMConfig:
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
    return _resolve_provider_config(provider, temperature)


# ── Provider-specific chat functions ─────────────────────────────────────────

def _chat_openai(prompt: str, config: LLMConfig) -> str:
    if not config.api_key:
        raise ValueError("Missing OPENAI_API_KEY.")
    kwargs: dict = {
        "model": config.model,
        "temperature": config.temperature,
        "api_key": config.api_key,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    response = completion(messages=[{"role": "user", "content": prompt}], **kwargs)
    return response["choices"][0]["message"]["content"]


def _chat_claude(prompt: str, config: LLMConfig) -> str:
    if not config.api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY.")
    response = completion(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        api_key=config.api_key,
        temperature=config.temperature,
    )
    return response["choices"][0]["message"]["content"]


def _chat_gemini(prompt: str, config: LLMConfig) -> str:
    if not config.api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    from google import genai
    from google.genai import types

    # Strip litellm-style prefix (e.g. "gemini/gemini-2.5-flash" -> "gemini-2.5-flash")
    model = config.model.removeprefix("gemini/")
    client = genai.Client(api_key=config.api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=config.temperature),
    )
    return response.text


def _chat_local(prompt: str, config: LLMConfig) -> str:
    kwargs: dict = {
        "model": config.model,
        "temperature": config.temperature,
        "api_key": config.api_key or "dummy",
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    response = completion(messages=[{"role": "user", "content": prompt}], **kwargs)
    return response["choices"][0]["message"]["content"]


# ── Provider-specific embed functions ────────────────────────────────────────

def _embed_openai(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    if not config.api_key:
        raise ValueError("Missing OPENAI_API_KEY.")
    kwargs: dict = {"model": config.embedding_model, "api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    response = embedding(input=text, **kwargs)
    return [item["embedding"] for item in response.get("data", [])]


def _embed_gemini(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    if not config.api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    from google import genai

    # Strip litellm-style prefix if present (e.g. "gemini/text-embedding-004" -> "models/text-embedding-004")
    embed_model = config.embedding_model.removeprefix("gemini/")
    if not embed_model.startswith("models/"):
        embed_model = f"models/{embed_model}"
    client = genai.Client(api_key=config.api_key)
    texts = [text] if isinstance(text, str) else text
    result = []
    for t in texts:
        response = client.models.embed_content(model=embed_model, contents=t)
        result.append(response.embeddings[0].values)
    return result


def _embed_claude(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    # Claude has no native embedding API; fall back to litellm with OpenAI embeddings
    kwargs: dict = {"model": config.embedding_model}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    response = embedding(input=text, **kwargs)
    return [item["embedding"] for item in response.get("data", [])]


def _embed_local(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    kwargs: dict = {"model": config.embedding_model, "api_key": config.api_key or "dummy"}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    response = embedding(input=text, **kwargs)
    return [item["embedding"] for item in response.get("data", [])]


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        load_env: bool = True,
    ) -> None:
        if config is not None:
            self.config = config
            return

        if load_env:
            load_dotenv()
            provider_name = (provider or os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER)).strip().lower()
            temp = (
                temperature
                if temperature is not None
                else float(os.getenv("LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
            )
        else:
            provider_name = (provider or _DEFAULT_PROVIDER).strip().lower()
            temp = temperature if temperature is not None else _DEFAULT_TEMPERATURE

        cfg = _resolve_provider_config(provider_name, temp)
        self.config = replace(
            cfg,
            model=model if model is not None else cfg.model,
            embedding_model=(
                embedding_model if embedding_model is not None else cfg.embedding_model
            ),
            api_key=api_key if api_key is not None else cfg.api_key,
            base_url=base_url if base_url is not None else cfg.base_url,
            temperature=temp,
        )

    def chat(self, prompt: str) -> str:
        provider = self.config.provider
        if provider == "openai":
            return _chat_openai(prompt, self.config)
        if provider in {"claude", "anthropic"}:
            return _chat_claude(prompt, self.config)
        if provider in {"gemini", "google"}:
            return _chat_gemini(prompt, self.config)
        if provider == "local":
            return _chat_local(prompt, self.config)
        raise ValueError(f"Unsupported provider: '{provider}'. Use: openai, gemini, claude, local")

    def embed(self, text: str | list[str]) -> list[list[float]]:
        provider = self.config.provider
        if provider == "openai":
            return _embed_openai(text, self.config)
        if provider in {"gemini", "google"}:
            return _embed_gemini(text, self.config)
        if provider in {"claude", "anthropic"}:
            return _embed_claude(text, self.config)
        if provider == "local":
            return _embed_local(text, self.config)
        raise ValueError(f"Unsupported provider: '{provider}'. Use: openai, gemini, claude, local")
