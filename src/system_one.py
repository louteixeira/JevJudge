"""JevJudge: tests TypeSafe's Jev via the Noul primitive.

Noul returns one float in [0, 1]: the probability of "yes". See
https://docs.typesafe.ai/primitives/noul and https://docs.typesafe.ai/concepts/state.
"""

from __future__ import annotations

import time

from typesafe_sdk import Noul, RetryPolicy, TypeSafeClient

from src.judge import JudgeResult

CRITERION = Noul(
    instructions=(
        """Is the answer fully supported by the evidence? Every claim in the answer must be stated or directly implied by the evidence. An unsupported detail, a changed number or entity, or a hedge turned into an absolute all count as not fully supported, even if the claim happens to be true in the real world."""
    ),
    criteria={
        "true": "Every claim the answer makes is stated or directly implied by the evidence.",
        "false": "The answer makes at least one claim that the evidence does not state or imply.",
    },
)


class JevJudge:
    """Faithfulness judge backed by Jev's Noul primitive."""

    def __init__(self, api_key: str, model: str = "jev-1.13.0") -> None:
        if not api_key:
            raise ValueError("Typesafe API key is required")

        self._client = TypeSafeClient(api_key=api_key, model=model, retry=RetryPolicy(max_retries=0))

    def judge(self, evidence: str, question: str, answer: str) -> JudgeResult:
        state = {"evidence": evidence, "question": question, "answer": answer}
        start = time.perf_counter()
        response = self._client.system_one(state=state, questions={"faithful": CRITERION})
        latency_s = time.perf_counter() - start
        return JudgeResult(
            score=response.nouls["faithful"].noul,
            latency_s=latency_s,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            extra_info={"request_id": response.request_id},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JevJudge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
