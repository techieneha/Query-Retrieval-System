"""
rag_pipeline/llm_reasoner.py
Wraps Mistral API – generates answers from retrieved chunks.
"""
import os
from loguru import logger
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

SYSTEM = """You are a helpful insurance policy assistant.
Answer the user's question using ONLY the provided policy context.
Be concise and precise (2-4 sentences). Cite the specific clause or section when possible.
If the answer is not in the context, say: "This information is not found in your policy document."
End with: "Is there anything else you'd like to know, or would you like to file a claim?"
"""

class LLMReasoner:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> Mistral:
        if not self._client:
            self._client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def reason(self, question: str, chunks: list[dict]) -> tuple[str, float, list[str]]:
        """Returns (answer, confidence, sources)."""
        if not chunks:
            return "No relevant policy content found for this question.", 0.0, []

        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        scores  = [c["score"] for c in chunks]
        variance = max(scores) - min(scores) if len(scores) > 1 else 0
        confidence = round(min(1.0, scores[0] * (1 + variance * 0.5)), 3)

        try:
            resp = self.client.chat.complete(
                model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": f"Policy context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Unable to generate answer: {e}"
            confidence = 0.0

        sources = [c["text"][:120] + "…" for c in chunks[:2]]
        return answer, confidence, sources