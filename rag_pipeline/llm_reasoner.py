from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

from time import perf_counter

async def answer_with_llm(query: str, top_docs):
    try:
        t0 = perf_counter()
        clauses = [doc.page_content for doc in top_docs]
        prompt = _build_concise_prompt(query, clauses)

        llm = ChatOllama(
            model="mistral",
            temperature=0.1,
            num_ctx=2048
        )
        chain = llm | StrOutputParser()
        response = await chain.ainvoke(prompt)
        t1 = perf_counter()
        logger.info(f"🧠 LLM answered in {t1 - t0:.2f}s")
        return _clean_response(response)

    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return "Answer unavailable."


def _build_concise_prompt(query: str, clauses: list) -> str:
    """Builds a strict prompt for 1-2 line answers"""
    return f"""<s>[INST] You are an insurance policy expert. Answer in EXACTLY 1-2 sentences using ONLY these clauses:
    
{chr(10).join(clauses)}

Question: {query}

Rules:
1. Answer in 1-2 sentences MAX
2. Start with Yes/No if applicable
3. Include ONLY key details
4. Never say "as per clause" or "refer to"
5. Never list multiple conditions
6. Format: [Answer]. [Optional brief condition]

Answer: [/INST]"""

def _clean_response(response: str) -> str:
    """Ensures consistent answer formatting"""
    response = response.strip()
    if not response.endswith('.'):
        response += '.'
    return response