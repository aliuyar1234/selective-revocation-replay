from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BUNDLE_CACHE: dict[str, "QwenTextBundle"] = {}
_BUNDLE_LOCK = threading.Lock()


@dataclass
class QwenTextBundle:
    model_name: str
    model_path: str
    runtime_loader: str
    tokenizer: Any
    model: Any
    device: Any
    dtype: str
    chat_template_kwargs: dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bootstrap_local_qwen_stack() -> None:
    os.environ.setdefault("TRANSFORMERS_SKIP_RUNTIME_VERSION_CHECK", "1")
    vendor_root = repo_root() / "vendor"
    search_paths = [
        str(vendor_root / "hfdeps"),
        str(vendor_root / "transformers-main" / "src"),
    ]
    for path in reversed(search_paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def load_text_only_qwen_bundle(model_config: dict[str, Any]) -> QwenTextBundle:
    with _BUNDLE_LOCK:
        model_path = str(model_config["local_model_path"])
        runtime_loader = str(model_config.get("runtime_loader", "auto_causal_lm"))
        model_name = str(model_config.get("model_name", Path(model_path).name))
        chat_template_kwargs = dict(model_config.get("chat_template_kwargs", {}))
        cache_key = _bundle_cache_key(model_path, runtime_loader, chat_template_kwargs)
        cached = _BUNDLE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        bootstrap_local_qwen_stack()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        if runtime_loader == "qwen3_5_text":
            config = Qwen3_5TextConfig.from_pretrained(model_path)
            model = Qwen3_5ForCausalLM.from_pretrained(
                model_path,
                config=config,
                dtype="auto",
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        elif runtime_loader == "auto_causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported runtime_loader {runtime_loader}")
        model.eval()
        _normalize_generation_config(model)

        device = next(model.parameters()).device
        bundle = QwenTextBundle(
            model_name=model_name,
            model_path=model_path,
            runtime_loader=runtime_loader,
            tokenizer=tokenizer,
            model=model,
            device=device,
            dtype=str(next(model.parameters()).dtype),
            chat_template_kwargs=chat_template_kwargs,
        )
        _BUNDLE_CACHE[cache_key] = bundle
        return bundle


def _bundle_cache_key(model_path: str, runtime_loader: str, chat_template_kwargs: dict[str, Any]) -> str:
    kwargs_key = json.dumps(chat_template_kwargs, sort_keys=True, ensure_ascii=True, default=str)
    return f"{runtime_loader}::{model_path}::{kwargs_key}"


def generate_chat_text(bundle: QwenTextBundle, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    inputs = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **bundle.chat_template_kwargs,
    ).to(bundle.device)
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    trimmed_ids = output_ids[:, prompt_length:]
    return bundle.tokenizer.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _normalize_generation_config(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = False
    for field in ("temperature", "top_p", "top_k", "min_p", "presence_penalty"):
        if hasattr(generation_config, field):
            setattr(generation_config, field, None)
