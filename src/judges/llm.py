"""LLMJudge: the same faithfulness criterion as JevJudge, asked of an Azure OpenAI chat
deployment with structured output.
"""

from __future__ import annotations

import os
import time

from openai import AzureOpenAI
from pydantic import BaseModel, Field

from src.judges.base import JudgeResult
from src.judges.prompts import LLM_JUDGE_HUMAN_PROMPT, LLM_JUDGE_SYSTEM_PROMPT


class Verdict(BaseModel):
    score: float = Field(
        description=(
            "Probability between 0 and 1 that the answer is fully supported by the evidence: "
            "1 means every claim is supported, 0 means it clearly is not."
        )
    )


class LLMJudge:
    """Faithfulness judge backed by an LLM."""

    def __init__(self) -> None:
        self._deployment = os.environ["AZURE_DEPLOYMENT"]
        self._client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_API_VERSION"],
            max_retries=0,
        )

    def judge(self, evidence: str, question: str, answer: str) -> JudgeResult:
        human_prompt = LLM_JUDGE_HUMAN_PROMPT.format(
            evidence=evidence, question=question, answer=answer
        )
        start = time.perf_counter()
        response = self._client.chat.completions.parse(
            model=self._deployment,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": human_prompt},
            ],
            response_format=Verdict,
        )

        latency_s = time.perf_counter() - start
        choice = response.choices[0]
        if choice.message.parsed is None:
            raise ValueError(
                f"No verdict returned (refusal={choice.message.refusal!r}, finish_reason={choice.finish_reason!r})"
            )
        score = choice.message.parsed.score
        if not 0 <= score <= 1:
            raise ValueError(f"Score {score} is outside [0, 1]")
        usage = response.usage
        return JudgeResult(
            score=score,
            latency_s=latency_s,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            model=response.model,
            extra_info={"request_id": response.id},
        )

    def close(self) -> None:
        self._client.close()
