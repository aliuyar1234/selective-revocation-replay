from __future__ import annotations

import pytest

from src.pipeline import load_project_settings, repo_root, resolve_model_config
from src.utils import load_yaml


def test_default_active_model_is_heuristic_artifact(monkeypatch) -> None:
    monkeypatch.delenv("SRR_ACTIVE_MODEL", raising=False)
    settings = load_project_settings()
    assert settings["active_model_name"] == "heuristic_artifact"
    assert settings["model"]["model_id"] == "deterministic-heuristic-backend"
    assert settings["model"]["backend"] == "heuristic"


def test_active_model_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("SRR_ACTIVE_MODEL", "qwen3_5_27b_instruct")
    settings = load_project_settings()
    assert settings["active_model_name"] == "qwen3_5_27b_instruct"
    assert settings["model"]["runtime_loader"] == "qwen3_5_text"


def test_active_model_can_be_overridden_to_ollama(monkeypatch) -> None:
    monkeypatch.setenv("SRR_ACTIVE_MODEL", "qwen2_5_14b_ollama")
    settings = load_project_settings()
    assert settings["active_model_name"] == "qwen2_5_14b_ollama"
    assert settings["model"]["backend"] == "ollama"
    assert settings["model"]["ollama_model"] == "qwen2.5:14b"


def test_active_model_override_rejects_unknown_model(monkeypatch) -> None:
    monkeypatch.setenv("SRR_ACTIVE_MODEL", "not_a_real_model")
    with pytest.raises(ValueError, match="Unknown model 'not_a_real_model' from SRR_ACTIVE_MODEL"):
        load_project_settings()


def test_resolve_model_config_rejects_unknown_local_model_name() -> None:
    root = repo_root()
    model_index = load_yaml(root / "configs" / "models.yaml")
    with pytest.raises(ValueError, match="Unknown model 'missing_local_model' from SRR_LOCAL_QWEN_MODEL"):
        resolve_model_config(root, model_index, "missing_local_model", env_var_name="SRR_LOCAL_QWEN_MODEL")
