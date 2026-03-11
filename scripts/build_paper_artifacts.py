from __future__ import annotations

import shutil

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from export_latex_tables import main as export_latex_tables_main
from src.figures import generate_tables_and_figures
from src.pipeline import load_project_settings


def main() -> None:
    settings = load_project_settings()
    root = settings["root"]
    generated = generate_tables_and_figures()
    paper_pkg_dir = root / "paper" / "usenix_security26"
    paper_assets_dir = paper_pkg_dir / "assets"
    paper_assets_dir.mkdir(parents=True, exist_ok=True)

    for key in ("fig4", "fig5"):
        path = generated[key]
        shutil.copy2(path, paper_assets_dir / path.name)

    refs_src = root / settings["paths"]["paper_refs_source"]
    shutil.copy2(refs_src, paper_pkg_dir / "refs.bib")
    export_latex_tables_main()
    print("standalone paper package refreshed")


if __name__ == "__main__":
    main()
