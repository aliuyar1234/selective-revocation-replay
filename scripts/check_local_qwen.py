from __future__ import annotations

import json
import os

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.utils import load_yaml
from src.llm_backend import build_backend
from src.pipeline import repo_root, resolve_model_config
from src.qwen_runtime import generate_chat_text, load_text_only_qwen_bundle


def main() -> None:
    root = repo_root()
    model_index = load_yaml(root / "configs" / "models.yaml")
    local_model_name = os.environ.get("SRR_LOCAL_QWEN_MODEL", "qwen2_5_7b_instruct")
    model_cfg = resolve_model_config(root, model_index, local_model_name, env_var_name="SRR_LOCAL_QWEN_MODEL")
    backend = build_backend("local_qwen", model_cfg)
    status = backend.warmup()  # type: ignore[attr-defined]
    bundle = load_text_only_qwen_bundle(model_cfg)
    sample = generate_chat_text(
        bundle,
        messages=[
            {"role": "system", "content": "Return exactly one JSON object and nothing else."},
            {"role": "user", "content": 'Reply with {"status":"ok"}.'},
        ],
        max_new_tokens=48,
    )
    print(
        f"local_qwen ready: active_model={status['model_name']} model_class={status['model_class']} "
        f"device={status['device']} dtype={status['dtype']}"
    )
    print(f"model_path={status['model_path']}")
    print("sample_output=" + json.dumps(sample, ensure_ascii=True))


if __name__ == "__main__":
    main()
