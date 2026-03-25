"""
LLM layer using Groq (llama3-8b — fast, free tier).
Falls back to Ollama if Groq key not set.
"""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a knowledgeable, practical fashion expert assistant.

Rules:
- Answer using ONLY the provided context from the knowledge base
- If context contains article links, reference them naturally
- Be specific, trendy, and practical — not generic
- If context doesn't cover the question, say so honestly
- Keep answers conversational but informative
- When listing items, use clean bullet points
- Include source links at the end when relevant
"""


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant content found in the knowledge base."
    parts = []
    for i, c in enumerate(chunks):
        url  = c.get("source_url", "")
        text = c.get("content_text", "")
        parts.append(f"[Source {i+1}]{' ('+url+')' if url else ''}\n{text}")
    return "\n\n---\n\n".join(parts)


def build_messages(query: str, chunks: list[dict], history: list[dict]) -> list:
    context  = build_context(chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or []):
        messages.append({"role": "user",      "content": turn["query"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({
        "role": "user",
        "content": f"Knowledge Base Context:\n{context}\n\nUser Question: {query}"
    })
    return messages


def ask(query: str, chunks: list[dict], history: list[dict] = None,
        image_data: dict = None) -> str:
    messages = build_messages(query, chunks, history)
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def ask_stream(query: str, chunks: list[dict], history: list[dict] = None,
               image_data: dict = None):
    """Generator yielding text tokens for streaming."""
    messages = build_messages(query, chunks, history)
    stream = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
