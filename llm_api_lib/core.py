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
    embedding_provider: str
    embedding_model: str
    api_key: str | None
    embedding_api_key: str | None
    base_url: str | None
    embedding_base_url: str | None
    temperature: float


def _embedding_settings(chat_provider: str) -> dict:
    default_provider = (
        "openai" if chat_provider in {"claude", "anthropic"} else chat_provider
    )
    provider = os.getenv("EMBEDDING_PROVIDER", default_provider).strip().lower()
    defaults = {
        "openai": "openai/text-embedding-3-small",
        "gemini": "gemini-embedding-2",
        "google": "gemini-embedding-2",
        "local": _DEFAULT_LOCAL_MODEL,
    }
    model = os.getenv("EMBEDDING_MODEL", defaults.get(provider, ""))
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif provider in {"gemini", "google"}:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = None
    elif provider == "local":
        api_key = os.getenv("LOCAL_LLM_API_KEY", "dummy")
        base_url = os.getenv("LOCAL_LLM_BASE_URL", _DEFAULT_LOCAL_BASE_URL)
    else:
        api_key = None
        base_url = None
    return {
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_api_key": api_key,
        "embedding_base_url": base_url,
    }


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
    embedding = _embedding_settings(provider)

    if provider == "openai":
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=temperature,
            **embedding,
        )

    if provider in {"claude", "anthropic"}:
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=None,
            temperature=temperature,
            **embedding,
        )

    if provider in {"gemini", "google"}:
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            base_url=None,
            temperature=temperature,
            **embedding,
        )

    if provider == "local":
        return LLMConfig(
            provider=provider,
            model=os.getenv("LLM_MODEL", _DEFAULT_LOCAL_MODEL),
            api_key=os.getenv("LOCAL_LLM_API_KEY", "dummy"),
            base_url=os.getenv("LOCAL_LLM_BASE_URL", _DEFAULT_LOCAL_BASE_URL),
            temperature=temperature,
            **embedding,
        )

    raise ValueError("Unsupported LLM_PROVIDER. Use: openai, gemini, claude, local")


def load_config() -> LLMConfig:
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
    return _resolve_provider_config(provider, temperature)


def _chat_content(response) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    return response["choices"][0]["message"]["content"]


def _embedding_values(response) -> list[list[float]]:
    """Extract vectors from an OpenAI-compatible embedding response."""
    return [item["embedding"] for item in response.get("data", [])]


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
    return _chat_content(response)


def _chat_claude(prompt: str, config: LLMConfig) -> str:
    if not config.api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY.")
    response = completion(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        api_key=config.api_key,
        temperature=config.temperature,
    )
    return _chat_content(response)


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
    return _chat_content(response)


# ── Provider-specific embed functions ────────────────────────────────────────

def _embed_openai(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    if not config.embedding_api_key:
        raise ValueError("Missing OPENAI_API_KEY.")
    kwargs: dict = {
        "model": config.embedding_model,
        "api_key": config.embedding_api_key,
    }
    if config.embedding_base_url:
        kwargs["base_url"] = config.embedding_base_url
    response = embedding(input=text, **kwargs)
    return _embedding_values(response)


def _embed_gemini(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    if not config.embedding_api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    from google import genai

    embed_model = _gemini_embedding_model_name(config.embedding_model)
    client = genai.Client(api_key=config.embedding_api_key)
    texts = [text] if isinstance(text, str) else text
    result = []
    for t in texts:
        response = client.models.embed_content(model=embed_model, contents=t)
        result.append(response.embeddings[0].values)
    return result


def _gemini_embedding_model_name(model: str) -> str:
    """Validate and normalize a Gemini embedding model identifier."""
    incompatible_prefixes = ("openai/", "anthropic/", "claude/", "local/")
    if model.lower().startswith(incompatible_prefixes):
        raise ValueError(
            f"Embedding model '{model}' is incompatible with the Gemini provider. "
            "Set EMBEDDING_MODEL=gemini-embedding-2."
        )
    return model.removeprefix("models/").removeprefix("gemini/")


def _embed_claude(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    # Claude has no native embedding API; fall back to litellm with OpenAI embeddings
    kwargs: dict = {"model": config.embedding_model}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    response = embedding(input=text, **kwargs)
    return _embedding_values(response)


def _embed_local(text: str | list[str], config: LLMConfig) -> list[list[float]]:
    kwargs: dict = {
        "model": config.embedding_model,
        "api_key": config.embedding_api_key or "dummy",
    }
    if config.embedding_base_url:
        kwargs["base_url"] = config.embedding_base_url
    response = embedding(input=text, **kwargs)
    return _embedding_values(response)


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
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
            embedding_provider=(
                embedding_provider
                if embedding_provider is not None
                else cfg.embedding_provider
            ),
            embedding_api_key=(
                embedding_api_key
                if embedding_api_key is not None
                else cfg.embedding_api_key
            ),
            embedding_base_url=(
                embedding_base_url
                if embedding_base_url is not None
                else cfg.embedding_base_url
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
        provider = self.config.embedding_provider
        if provider == "openai":
            return _embed_openai(text, self.config)
        if provider in {"gemini", "google"}:
            return _embed_gemini(text, self.config)
        if provider in {"claude", "anthropic"}:
            return _embed_claude(text, self.config)
        if provider == "local":
            return _embed_local(text, self.config)
        raise ValueError(f"Unsupported provider: '{provider}'. Use: openai, gemini, claude, local")
