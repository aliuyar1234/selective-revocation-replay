from __future__ import annotations

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.pipeline import run_full_matrix


def main() -> None:
    results = run_full_matrix()
    print(f"wrote {len(results)} result records")


if __name__ == "__main__":
    main()
