from __future__ import annotations

import argparse

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.env_workspace import materialize_chain_workspace
from src.pipeline import load_project_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", required=True)
    parser.add_argument("--attack", choices=["explicit", "stealth"], required=True)
    args = parser.parse_args()

    settings = load_project_settings()
    workspace = materialize_chain_workspace(
        settings["catalog"],
        args.chain,
        args.attack,
        settings["root"] / settings["paths"]["workspace_root"],
    )
    print(workspace.root)
    print(workspace.manifest_path)


if __name__ == "__main__":
    main()
