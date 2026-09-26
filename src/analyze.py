"""Statistics for every run in results/: one table per run, judges side by side.

    uv run python -m src.analyze
    uv run python -m src.analyze > results/summary.md

Reads only the JSONL files written by run_eval.py; makes no API calls. Failed calls
(`error` set) are left out of the statistics and counted as "Failed calls".
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tomllib
from bisect import bisect_left, bisect_right
from pathlib import Path

from src.judges.base import PASS_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
JUDGE_NAMES = {"jev": "Jev", "llm": "LLM"}
PRICING_PATH = ROOT / "pricing.toml"


def _load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_pricing(path: Path) -> dict[str, dict[str, float]]:
    """USD per million tokens per judge, or {} when the file is missing."""
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _cost_per_100(records: list[dict], price: dict[str, float]) -> float:
    """Estimated USD for 100 calls, from the mean tokens per call in these records."""
    input_tokens = statistics.mean(r["input_tokens"] or 0 for r in records)
    output_tokens = statistics.mean(r["output_tokens"] or 0 for r in records)
    per_call = (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6
    return per_call * 100


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean, resampling cases with a fixed seed.

    With per-case 0/1 correctness this is the CI for accuracy; with per-case
    differences between two judges it is the CI for the paired accuracy difference.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    samples = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_SAMPLES)
    )
    low = samples[int(0.025 * BOOTSTRAP_SAMPLES)]
    high = samples[int(0.975 * BOOTSTRAP_SAMPLES) - 1]
    return low, high


def _auc(good: list[float], bad: list[float]) -> float:
    """Chance that a random good answer scores above a random bad one (ties count half)."""
    bad = sorted(bad)
    wins = sum(
        bisect_left(bad, s) + 0.5 * (bisect_right(bad, s) - bisect_left(bad, s))
        for s in good
    )
    return wins / (len(good) * len(bad))


def _auc_ci(good: list[float], bad: list[float]) -> tuple[float, float]:
    """Bootstrap 95% CI for the AUC, resampling good and bad cases separately.

    Resampling each label on its own keeps the pass/fail balance of the data set in
    every resample, and guarantees both groups are present.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        _auc(rng.choices(good, k=len(good)), rng.choices(bad, k=len(bad)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    low = samples[int(0.025 * BOOTSTRAP_SAMPLES)]
    high = samples[int(0.975 * BOOTSTRAP_SAMPLES) - 1]
    return low, high


def _format_percent(hits: int, total: int) -> str:
    """hits / total as display text, e.g. "80.0%", or "-" when total is 0."""
    return f"{hits / total:.1%}" if total else "-"


def _is_correct(record: dict) -> bool:
    verdict = "pass" if record["score"] >= PASS_THRESHOLD else "fail"
    return verdict == record["label"]


def _judge_stats(records: list[dict], price: dict[str, float] | None) -> dict[str, str]:
    ok = [r for r in records if not r.get("error")]
    stats = {"Cases judged": str(len(ok)), "Failed calls": str(len(records) - len(ok))}
    if not ok:
        return stats

    verdicts = ["pass" if r["score"] >= PASS_THRESHOLD else "fail" for r in ok]
    labels = [r["label"] for r in ok]
    correct = [_is_correct(r) for r in ok]
    low, high = _mean_ci(correct)
    good_accepted = sum(v == y == "pass" for v, y in zip(verdicts, labels))
    bad_rejected = sum(v == y == "fail" for v, y in zip(verdicts, labels))
    good_scores = [r["score"] for r in ok if r["label"] == "pass"]
    bad_scores = [r["score"] for r in ok if r["label"] == "fail"]

    stats |= {
        "Accuracy": _format_percent(sum(correct), len(ok)),
        "Accuracy 95% CI": f"{low:.1%} - {high:.1%}",
        "Good answers accepted": _format_percent(good_accepted, labels.count("pass")),
        "Bad answers rejected": _format_percent(bad_rejected, labels.count("fail")),
    }
    if good_scores and bad_scores:
        auc_low, auc_high = _auc_ci(good_scores, bad_scores)
        stats |= {
            "AUC": f"{_auc(good_scores, bad_scores):.3f}",
            "AUC 95% CI": f"{auc_low:.3f} - {auc_high:.3f}",
        }
    stats |= {
        "Median latency": f"{statistics.median(r['latency_s'] for r in ok):.2f} s",
        "Mean input tokens": f"{statistics.mean(r['input_tokens'] or 0 for r in ok):.0f}",
        "Mean output tokens": f"{statistics.mean(r['output_tokens'] or 0 for r in ok):.1f}",
    }
    if price:
        stats["Cost per 100 calls"] = f"${_cost_per_100(ok, price):.4f}"
    return stats


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """A markdown table whose source is also aligned, so it reads well in a terminal."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        first, *rest = cells
        padded = [first.ljust(widths[0])] + [
            c.rjust(w) for c, w in zip(rest, widths[1:])
        ]
        return "| " + " | ".join(padded) + " |"

    divider = (
        "|"
        + "|".join(["-" * (widths[0] + 2)] + ["-" * (w + 1) + ":" for w in widths[1:]])
        + "|"
    )
    return "\n".join([line(header), divider, *(line(row) for row in rows)])


def _paired_difference(jev: list[dict], llm: list[dict]) -> str:
    """Jev accuracy minus LLM accuracy on the cases both judged, with a bootstrap CI."""
    jev_ok = {r["case_index"]: r for r in jev if not r.get("error")}
    llm_ok = {r["case_index"]: r for r in llm if not r.get("error")}
    shared = sorted(jev_ok.keys() & llm_ok.keys())
    if not shared:
        return "Paired accuracy difference (Jev - LLM): no case was judged by both."
    diffs = [_is_correct(jev_ok[i]) - _is_correct(llm_ok[i]) for i in shared]
    low, high = _mean_ci(diffs)
    return (
        f"Paired accuracy difference (Jev - LLM): {statistics.mean(diffs) * 100:+.1f} points, "
        f"95% CI {low * 100:+.1f} to {high * 100:+.1f}, on the {len(shared)} cases both judged."
    )


def _cost_ratio(llm: list[dict], jev: list[dict], pricing: dict) -> str | None:
    """How many times more the LLM costs per call than Jev, if both are priced."""
    llm_ok = [r for r in llm if not r.get("error")]
    jev_ok = [r for r in jev if not r.get("error")]
    if not (llm_ok and jev_ok and "llm" in pricing and "jev" in pricing):
        return None
    jev_cost = _cost_per_100(jev_ok, pricing["jev"])
    if not jev_cost:
        return None
    ratio = _cost_per_100(llm_ok, pricing["llm"]) / jev_cost
    return f"Cost: the LLM costs {ratio:.1f}x as much as Jev per call, at the prices in pricing.toml."


def summarize_run(run_dir: Path, pricing: dict) -> str | None:
    """Markdown statistics for one run folder, or None if it has no JSONL files."""
    records = {p.stem: _load_records(p) for p in sorted(run_dir.glob("*.jsonl"))}
    if not records:
        return None
    by_judge = {
        judge: _judge_stats(recs, pricing.get(judge)) for judge, recs in records.items()
    }
    metrics = list(dict.fromkeys(m for stats in by_judge.values() for m in stats))
    header = ["Metric"] + [JUDGE_NAMES.get(j, j) for j in by_judge]
    rows = [[m] + [stats.get(m, "-") for stats in by_judge.values()] for m in metrics]
    table = f"## {run_dir.name}\n\n{_markdown_table(header, rows)}"
    if "jev" in records and "llm" in records:
        table += "\n\n" + _paired_difference(records["jev"], records["llm"])
        if ratio := _cost_ratio(records["llm"], records["jev"], pricing):
            table += "\n\n" + ratio
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--pricing", type=Path, default=PRICING_PATH)
    args = parser.parse_args()
    pricing = _load_pricing(args.pricing)

    summaries = [
        summarize_run(d, pricing)
        for d in sorted(p for p in args.results.iterdir() if p.is_dir())
    ]
    summaries = [s for s in summaries if s]
    if not summaries:
        raise SystemExit(f"No run folders with JSONL files in {args.results}.")

    print("\n\n".join(summaries))
    print(f"\nA judge's verdict is `pass` when its score is >= {PASS_THRESHOLD}.")
    print("Good answers accepted: share of `pass`-labelled cases the judge passed.")
    print("Bad answers rejected: share of `fail`-labelled cases the judge failed.")
    print("AUC: chance a random good answer scores above a random bad one (ties count")
    print("half); it uses the raw scores, so it does not depend on the threshold.")
    print("Paired accuracy difference: both judges scored on the same cases, so the CI")
    print("resamples cases and compares the judges case by case.")
    if pricing:
        print(
            "Cost per 100 calls: mean tokens per call times the USD per million tokens"
        )
        print(f"in {args.pricing.name}; an estimate, not a billed amount.")


if __name__ == "__main__":
    main()
