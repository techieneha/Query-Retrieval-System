import json
import asyncio
from difflib import SequenceMatcher
from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
from rag_pipeline.llm_reasoner import answer_with_llm

EVAL_PATH = "tests/eval_set.json"
PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

def similarity(a, b):
    """Fuzzy similarity score between 0 and 1"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

async def run_evaluation():
    print("📥 Loading and chunking PDF...")
    docs = await load_pdf(PDF_URL)
    vectorstore = embed_chunks(docs)

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total = len(test_cases)
    exact, threshold = 0, 0
    threshold_limit = 0.75

    print(f"\n🧪 Running evaluation on {total} questions...\n")

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected = case["expected_answer"]

        print(f"[{i}] Q: {question}")
        chunks = await retrieve_with_rerank(question, vectorstore)
        predicted = await answer_with_llm(question, chunks)
        sim = similarity(predicted, expected)

        print(f"   ✅ Predicted: {predicted}")
        print(f"   🎯 Expected:  {expected}")
        print(f"   🔁 Similarity: {sim:.2f}\n")

        if predicted.strip().lower() == expected.strip().lower():
            exact += 1
        if sim >= threshold_limit:
            threshold += 1

    print("📊 Summary:")
    print(f"Total Questions:  {total}")
    print(f"Exact Matches:    {exact} ({exact / total:.1%})")
    print(f"Similarity ≥ 75%: {threshold} ({threshold / total:.1%})")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
