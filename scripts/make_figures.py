from __future__ import annotations

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.figures import generate_tables_and_figures


def main() -> None:
    paths = generate_tables_and_figures()
    for key in sorted(key for key in paths if key.startswith("fig")):
        print(f"{key}: {paths[key]}")


if __name__ == "__main__":
    main()
