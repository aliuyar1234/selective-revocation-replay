from __future__ import annotations

import json
import os

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.llm_backend import build_backend
from src.ollama_runtime import ollama_chat
from src.pipeline import repo_root, resolve_model_config
from src.utils import load_yaml


def main() -> None:
    root = repo_root()
    model_index = load_yaml(root / "configs" / "models.yaml")
    model_name = os.environ.get("SRR_OLLAMA_MODEL", "qwen2_5_14b_ollama")
    model_cfg = resolve_model_config(root, model_index, model_name, env_var_name="SRR_OLLAMA_MODEL")
    if model_cfg.get("backend") != "ollama":
        raise RuntimeError(f"Model {model_name} is not configured for the Ollama backend: {model_cfg}")

    backend = build_backend("ollama", model_cfg)
    status = backend.warmup()  # type: ignore[attr-defined]
    sample = ollama_chat(
        model_cfg,
        messages=[
            {"role": "system", "content": "Return exactly one JSON object and nothing else."},
            {"role": "user", "content": 'Reply with {"status":"ok"}.'},
        ],
        schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        },
        max_new_tokens=32,
    )
    print(
        f"ollama backend ready: active_model={status['model_name']} ollama_model={status['ollama_model']} "
        f"version={status['ollama_version']}"
    )
    print(f"base_url={status['base_url']}")
    print("sample_output=" + json.dumps(sample.content, ensure_ascii=True))


if __name__ == "__main__":
    main()
