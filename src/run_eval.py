"""Run each judge once per case and save every call to results/raw.jsonl.

    uv run python -m src.run_eval                # every case in data/cases_qa.csv
    uv run python -m src.run_eval --limit 3      # smoke test

Re-analyze later without new API calls: uv run python -m src.analyze
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from src.judge import Judge
from src.system_one import JevJudge

ROOT = Path(__file__).resolve().parent.parent


def load_cases(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def evaluate(case_index: int, case: dict, judge_name: str, judge: Judge) -> dict:
    """One judge call as a JSONL record. A failed call is recorded, not raised."""
    record = {"case_index": case_index, "source_id": case.get("source_id"), "judge": judge_name, "label": case["label"]}
    evidence, question, answer = case["evidence"], case["question"], case["answer"]
    try:
        result = judge.judge(evidence, question, answer)
    except Exception as error:  # noqa: BLE001 - one bad call must not abort the run
        return record | {"error": f"{type(error).__name__}: {error}"}
    return record | asdict(result) | {"error": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases_qa.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "raw.jsonl")
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = parser.parse_args()

    load_dotenv()
    cases = load_cases(args.cases, args.limit)
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists. Re-run with --force to overwrite it.")

    judges: dict[str, Judge] = {"jev": JevJudge(api_key=os.environ["TYPESAFE_API_KEY"])}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    try:
        with args.out.open("w", encoding="utf-8") as f:
            for i, case in enumerate(cases):
                for name, judge in judges.items():
                    record = evaluate(i, case, name, judge)
                    errors += record["error"] is not None
                    f.write(json.dumps(record) + "\n")
                    f.flush()
    finally:
        for judge in judges.values():
            judge.close()
    print(f"Wrote {len(cases) * len(judges)} calls ({errors} errors) to {args.out}")


if __name__ == "__main__":
    main()
