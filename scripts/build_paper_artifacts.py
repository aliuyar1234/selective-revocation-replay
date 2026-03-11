from __future__ import annotations

import shutil
from pathlib import Path

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.figures import generate_tables_and_figures
from src.pipeline import load_project_settings
from src.utils import dump_json


def main() -> None:
    settings = load_project_settings()
    root = settings["root"]
    generated = generate_tables_and_figures()
    paper_dir = root / "paper"
    paper_fig_dir = paper_dir / "figures"
    paper_tbl_dir = paper_dir / "tables"
    artifact_dir = root / "artifact"
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    paper_tbl_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for key, path in generated.items():
        if key.startswith("fig"):
            shutil.copy2(path, paper_fig_dir / path.name)
            png = path.with_suffix(".png")
            if png.exists():
                shutil.copy2(png, paper_fig_dir / png.name)
        else:
            shutil.copy2(path, paper_tbl_dir / path.name)

    refs_src = root / settings["paths"]["paper_refs_source"]
    shutil.copy2(refs_src, paper_dir / "refs.bib")

    dump_json(
        artifact_dir / "generated_manifest.json",
        {key: str(path.relative_to(root)).replace("\\", "/") for key, path in generated.items()},
    )
    print("paper and artifact assets assembled")


if __name__ == "__main__":
    main()
