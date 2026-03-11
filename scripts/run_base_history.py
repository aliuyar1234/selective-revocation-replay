from __future__ import annotations

import argparse

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.pipeline import create_base_history_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", required=True)
    parser.add_argument("--architecture", choices=["retrieval", "summary"], required=True)
    parser.add_argument("--attack", choices=["explicit", "stealth"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run_id, run_dir = create_base_history_run(args.chain, args.architecture, args.attack, seed=args.seed)
    print(run_id)
    print(run_dir)


if __name__ == "__main__":
    main()
