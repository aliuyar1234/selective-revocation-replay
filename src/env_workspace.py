from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.schemas import ChainDefinition, MaterializedFile, TaskCatalog
from src.utils import basename_title, ensure_dir, short_text, unique_tokens


@dataclass
class SearchEntry:
    path: str
    title: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "title": self.title, "snippet": self.snippet}


@dataclass
class WorkspacePaths:
    root: Path
    manifest_path: Path
    attack_path: str


def materialize_chain_workspace(catalog: TaskCatalog, chain_id: str, attack_variant: str, output_root: str | Path) -> WorkspacePaths:
    chain = catalog.get_chain(chain_id)
    workspace_root = ensure_dir(Path(output_root) / f"{chain_id}_{attack_variant}")
    for file_def in chain.common_files:
        _write_materialized_file(workspace_root, file_def)
    attack_file = chain.attack_files[attack_variant]
    _write_materialized_file(workspace_root, attack_file)
    manifest = {
        "chain_id": chain_id,
        "attack_variant": attack_variant,
        "attack_path": attack_file.path,
        "files": sorted([entry.path for entry in chain.common_files] + [attack_file.path]),
    }
    manifest_path = workspace_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return WorkspacePaths(root=workspace_root, manifest_path=manifest_path, attack_path=attack_file.path)


def _write_materialized_file(root: Path, file_def: MaterializedFile) -> None:
    target = root / file_def.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(file_def.content.strip() + "\n", encoding="utf-8")


class WorkspaceEnv:
    def __init__(self, chain: ChainDefinition, workspace_root: str | Path, token_regex: str):
        self.chain = chain
        self.workspace_root = Path(workspace_root)
        self.token_regex = token_regex

    def _iter_files(self) -> Iterable[tuple[str, Path]]:
        for path in sorted(self.workspace_root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                rel = path.relative_to(self.workspace_root).as_posix()
                yield rel, path

    def search_docs(self, query: str, k: int, quarantined_paths: set[str] | None = None) -> list[SearchEntry]:
        quarantined_paths = quarantined_paths or set()
        query_tokens = unique_tokens(query, token_regex=self.token_regex)
        results: list[tuple[int, str, SearchEntry]] = []
        allow_restricted = "restricted/vault.txt" in query_tokens or self.chain.restricted_secret.lower() in query.lower()
        for rel_path, file_path in self._iter_files():
            if rel_path in quarantined_paths:
                continue
            if rel_path == "restricted/vault.txt" and not allow_restricted:
                continue
            content = file_path.read_text(encoding="utf-8")
            score = 0
            path_lc = rel_path.lower()
            content_lc = content.lower()
            for token in query_tokens:
                if token in path_lc:
                    score += 3
                if token in content_lc:
                    score += 1
            if score > 0:
                results.append(
                    (
                        score,
                        rel_path,
                        SearchEntry(
                            path=rel_path,
                            title=basename_title(rel_path),
                            snippet=short_text(content, max_len=160),
                        ),
                    )
                )
        results.sort(key=lambda item: (-item[0], item[1]))
        return [entry for _, _, entry in results[:k]]

    def read_doc(
        self,
        path: str,
        searchable_paths_this_session: set[str],
        prompt_visible_texts: list[str],
        quarantined_paths: set[str] | None = None,
    ) -> tuple[str, str]:
        quarantined_paths = quarantined_paths or set()
        if path in quarantined_paths:
            raise PermissionError(f"Path {path} is quarantined")
        allowed_by_search = path in searchable_paths_this_session
        allowed_by_visible_path = any(path in visible_text for visible_text in prompt_visible_texts)
        if not (allowed_by_search or allowed_by_visible_path):
            raise PermissionError(f"Path {path} is not readable in this context")
        target = self.workspace_root / path
        if not target.exists():
            raise FileNotFoundError(path)
        return path, target.read_text(encoding="utf-8")

    @staticmethod
    def serialize_search_results(entries: list[SearchEntry]) -> str:
        return json.dumps([entry.to_dict() for entry in entries], sort_keys=True, ensure_ascii=True)

    @staticmethod
    def deserialize_search_results(content: str) -> list[dict[str, str]]:
        return json.loads(content)

    def score_public_docs_for_query(self, query: str) -> list[str]:
        return [entry.path for entry in self.search_docs(query, 3)]
