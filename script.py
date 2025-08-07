import asyncio
from rag_pipeline.retriever import load_pdf, embed_chunks, retrieve_with_rerank
from rag_pipeline.llm_reasoner import answer_with_llm

async def main():
    # PDF URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    print("[1] Loading and chunking PDF...")
    docs = await load_pdf(pdf_url)

    print(f"[2] Loaded {len(docs)} chunks. Embedding...")
    vectorstore = embed_chunks(docs)

    questions = [
    "Does the policy cover maternity?",
    "What is the waiting period for pre-existing diseases?",
    "Is ambulance service covered?"
]

    for query in questions:
        print(f"\n🔍 Query: {query}")
        top_docs = await retrieve_with_rerank(query, vectorstore)
        answer = await answer_with_llm(query, top_docs)
        print(f"✅ Answer: {answer}")


    print("[3] Retrieving top documents...")
    top_docs = await retrieve_with_rerank(query, vectorstore)

    print(f"[4] Retrieved {len(top_docs)} top chunks. Reasoning with LLM...")
    answer = await answer_with_llm(query, top_docs)

    print("\n✅ Final Answer:", answer)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
