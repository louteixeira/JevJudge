CRITERION_PARTIAL_PROMPT = """Is the answer fully supported by the evidence? Every claim in the answer must be stated or directly implied by the evidence. An unsupported detail, a changed number or entity, or a hedge turned into an absolute all count as not fully supported, even if the claim happens to be true in the real world."""

LLM_JUDGE_SYSTEM_PROMPT = "You are a strict evaluator of grounded question answering. Use only the evidence you are given. Do not use outside knowledge."

LLM_JUDGE_HUMAN_PROMPT = f"""Evidence:
{{evidence}}

Question:
{{question}}

Answer:
{{answer}}

{CRITERION_PARTIAL_PROMPT}
"""
