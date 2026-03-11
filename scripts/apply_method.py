from __future__ import annotations

import argparse

from _bootstrap import bootstrap_repo_path

bootstrap_repo_path()

from src.pipeline import create_method_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-history", required=True)
    parser.add_argument(
        "--method",
        choices=["no_recovery", "root_delete", "full_reset", "coarse_rollback", "selective_replay", "revoke_no_replay"],
        required=True,
    )
    args = parser.parse_args()

    run_id, run_dir, result = create_method_run(args.base_history, args.method)
    print(run_id)
    print(run_dir)
    print(result.to_dict())


if __name__ == "__main__":
    main()
