from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


DEFAULT_TOKEN_REGEX = r"[A-Za-z0-9_./$-]+"


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json_dumps(payload) + "\n", encoding="utf-8")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokenize(text: str, token_regex: str = DEFAULT_TOKEN_REGEX) -> list[str]:
    return re.findall(token_regex, text.lower())


def unique_tokens(text: str, token_regex: str = DEFAULT_TOKEN_REGEX) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text, token_regex=token_regex):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_dict(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def short_text(text: str, max_len: int = 160) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:max_len]


def basename_title(path: str) -> str:
    return Path(path).stem.replace("_", " ")


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [chunk.strip() for chunk in raw if chunk.strip()]


def best_matching_sentences(texts: Sequence[str], query: str, token_regex: str = DEFAULT_TOKEN_REGEX, limit: int = 2) -> list[str]:
    query_tokens = set(unique_tokens(query, token_regex=token_regex))
    scored: list[tuple[int, str]] = []
    for text in texts:
        for sentence in split_sentences(text):
            overlap = len(query_tokens.intersection(set(unique_tokens(sentence, token_regex=token_regex))))
            if overlap > 0:
                scored.append((overlap, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[str] = []
    for _, sentence in scored:
        if sentence not in chosen:
            chosen.append(sentence)
        if len(chosen) >= limit:
            break
    return chosen


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    row_list = list(rows)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def copy_tree(src: str | Path, dst: str | Path) -> None:
    import shutil

    source = Path(src)
    target = Path(dst)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
