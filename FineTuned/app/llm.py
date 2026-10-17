"""Streaming client for llama.cpp's llama-server."""

import json
from collections.abc import Iterator

import requests

from app import config
from app.retrieval import Chunk

SYSTEM_PROMPT = (
    "You are Ask SolarSPELL, an offline assistant for teachers and students. "
    "Answer clearly and concisely using the library excerpts provided. "
    "If the excerpts do not contain the answer, say so instead of guessing."
)

# The fine-tuned model was trained on Alpaca-style prompts; stop if it starts
# writing a new turn on its own.
ALPACA_STOP = ["### Instruction:", "### Input:", "### Response:"]


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{n}] {c.source} (p. {c.page}): {c.text[:config.CONTEXT_CHARS]}"
        for n, c in enumerate(chunks, start=1)
    )


def _request(question: str, context: str) -> tuple[str, dict]:
    if config.PROMPT_FORMAT == "alpaca":
        prompt = f"### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:\n"
        return f"{config.LLM_URL}/completion", {
            "prompt": prompt,
            "n_predict": config.MAX_TOKENS,
            "temperature": config.TEMPERATURE,
            "stop": ALPACA_STOP,
            "stream": True,
        }

    return f"{config.LLM_URL}/v1/chat/completions", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Library excerpts:\n{context}\n\nQuestion: {question}"},
        ],
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "stream": True,
    }


def _token(event: dict) -> str:
    if "choices" in event:  # OpenAI-style chat stream
        choices = event["choices"]
        return (choices[0].get("delta") or {}).get("content") or "" if choices else ""
    return event.get("content", "")  # native /completion stream


def stream_answer(question: str, chunks: list[Chunk]) -> Iterator[str]:
    """Yields generated text pieces. Raises requests.RequestException on failure."""
    url, payload = _request(question, format_context(chunks))

    with requests.post(url, json=payload, stream=True, timeout=(5, 600)) as response:
        response.raise_for_status()
        # Server-sent events: each event is a "data: {...}" line.
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):].decode("utf-8")
            if data == "[DONE]":
                break
            text = _token(json.loads(data))
            if text:
                yield text


def is_ready() -> bool:
    try:
        return requests.get(f"{config.LLM_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False
