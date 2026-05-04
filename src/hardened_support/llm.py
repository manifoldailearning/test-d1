"""Shared OpenAI chat model wiring (LangChain). Requires ``OPENAI_API_KEY``."""
from __future__ import annotations

from langchain_openai import ChatOpenAI


def chat_openai(*, model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)
