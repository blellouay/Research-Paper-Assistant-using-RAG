"""Small Ollama-compatible adapter for Groq's Chat Completions API."""

import os
from types import SimpleNamespace

import requests


class GroqClient:
    """Expose Groq through the ``client.chat`` interface used by the generator."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key=None, timeout=90, images=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.timeout = timeout
        self.images = [image for image in (images or []) if image]
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it in PowerShell and restart Streamlit."
            )

    def chat(self, model, messages, options=None, format=None, **_kwargs):
        options = options or {}
        messages = [dict(message) for message in messages]
        if self.images and messages:
            final = messages[-1]
            final["content"] = [
                {"type": "text", "text": str(final.get("content", ""))},
                *[
                    {"type": "image_url", "image_url": {"url": image}}
                    for image in self.images
                ],
            ]
        body = {
            "model": model,
            "messages": messages,
            "temperature": max(float(options.get("temperature", 0.1)), 1e-8),
        }
        if format == "json":
            body["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f" Groq response: {exc.response.text[:500]}"
            raise RuntimeError(f"Groq API request failed.{detail}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Groq returned an unexpected response format.") from exc
        return SimpleNamespace(message=SimpleNamespace(content=content))
