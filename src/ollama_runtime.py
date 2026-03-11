from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class OllamaChatResult:
    model: str
    content: str
    raw: dict[str, Any]


def ollama_chat(
    model_config: dict[str, Any],
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    max_new_tokens: int,
) -> OllamaChatResult:
    model_name = str(model_config["ollama_model"])
    base_url = str(model_config.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")
    timeout_seconds = float(model_config.get("request_timeout_seconds", 600))
    options: dict[str, Any] = {
        "num_predict": int(max_new_tokens),
        "temperature": float(model_config.get("temperature", 0)),
    }
    if "top_p" in model_config:
        options["top_p"] = float(model_config["top_p"])
    if "num_ctx" in model_config:
        options["num_ctx"] = int(model_config["num_ctx"])
    if "seed" in model_config:
        options["seed"] = int(model_config["seed"])
    if "repeat_penalty" in model_config:
        options["repeat_penalty"] = float(model_config["repeat_penalty"])

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": options,
        "keep_alive": str(model_config.get("keep_alive", "10m")),
    }
    if "think" in model_config:
        payload["think"] = bool(model_config["think"])

    response = _post_json(f"{base_url}/api/chat", payload, timeout_seconds=timeout_seconds)
    message = response.get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama returned an empty chat response for model {model_name}: {response}")
    return OllamaChatResult(model=model_name, content=content.strip(), raw=response)


def ollama_version(model_config: dict[str, Any]) -> dict[str, Any]:
    base_url = str(model_config.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")
    timeout_seconds = float(model_config.get("request_timeout_seconds", 30))
    return _get_json(f"{base_url}/api/version", timeout_seconds=timeout_seconds)


def _get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc.reason}") from exc
    return json.loads(body)


def _post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc.reason}") from exc
    return json.loads(body)
