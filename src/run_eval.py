"""Run each judge once per case. Every run gets its own folder, results/<datetime>/, with one
JSONL file per judge (jev.jsonl, llm.jsonl) so the two can be compared line by line.

    uv run python -m src.run_eval                    # both judges on every case in data/cases_qa.csv
    uv run python -m src.run_eval --judge jev        # only Jev (or: --judge llm)
    uv run python -m src.run_eval --limit 3          # smoke test

Re-analyze later without new API calls: uv run python -m src.analyze
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.judge import Judge
from src.llm import LLMJudge
from src.system_one import JevJudge

ROOT = Path(__file__).resolve().parent.parent
JUDGES = {"jev": JevJudge, "llm": LLMJudge}


def _load_cases(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def _evaluate(case_index: int, case: dict, judge_name: str, judge: Judge) -> dict:
    """One judge call as a JSONL record. A failed call is recorded, not raised."""
    record = {
        "case_index": case_index,
        "source_id": case.get("source_id"),
        "judge": judge_name,
        "label": case["label"],
    }
    evidence, question, answer = case["evidence"], case["question"], case["answer"]
    try:
        result = judge.judge(evidence, question, answer)
    except Exception as error:  # noqa: BLE001 - one bad call must not abort the run
        return record | {"error": f"{type(error).__name__}: {error}"}
    return record | {"verdict": result.verdict} | asdict(result) | {"error": None}


def run_judge(name: str, judge: Judge, cases: list[dict], path: Path) -> int:
    """Judge every case, writing one JSONL record per case. Returns the number of failed calls."""
    errors = 0
    with path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            record = _evaluate(i, case, name, judge)
            errors += record["error"] is not None
            f.write(json.dumps(record) + "\n")
            f.flush()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "cases_qa.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="folder that holds the run folders",
    )
    parser.add_argument(
        "--judge", choices=sorted(JUDGES), help="run only this judge (default: both)"
    )
    parser.add_argument("--limit", type=int, help="only the first N cases (smoke test)")
    args = parser.parse_args()

    load_dotenv()
    cases = _load_cases(args.cases, args.limit)

    judges: dict[str, Judge] = {
        name: JUDGES[name]() for name in ([args.judge] if args.judge else JUDGES)
    }

    run_dir = args.out_dir / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True)
    try:
        for name, judge in judges.items():
            path = run_dir / f"{name}.jsonl"
            errors = run_judge(name, judge, cases, path)
            print(f"{name}: {len(cases)} calls, {errors} errors -> {path}")
    finally:
        for judge in judges.values():
            judge.close()


if __name__ == "__main__":
    main()
