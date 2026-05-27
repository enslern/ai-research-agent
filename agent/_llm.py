"""
Shared lazy LLM client factory.

The OpenAI client raises at instantiation time if no API key is present.
By deferring construction to the first call, modules can be imported freely
in tests without needing a real GROQ_API_KEY in the environment.
"""

from __future__ import annotations
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY", "dummy-key-for-tests")
        _client = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client
