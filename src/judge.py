"""Shared interface both judges implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class JudgeResult:
    """One judge call, with everything run_eval.py needs to log metrics."""

    score: float
    """Probability/score in [0, 1] that the answer is fully supported by the evidence"""
    latency_s: float
    """Wall-clock time of the API call, in seconds"""
    input_tokens: int | None
    """Input tokens billed for the call"""
    output_tokens: int | None
    """Output tokens billed for the call"""
    model: str
    """The model name that actually served the call"""
    extra_info: dict[str, Any] = field(default_factory=dict)
    """Judge-specific extras (request id, hyperparameters, etc.)"""


class Judge(Protocol):
    """judge(evidence, question, answer) -> JudgeResult, for the single faithfulness criterion:
    "Is the answer fully supported by the evidence?" """

    def judge(self, evidence: str, question: str, answer: str) -> JudgeResult: ...

    def close(self) -> None: ...
