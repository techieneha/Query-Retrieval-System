from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import logging
from functools import lru_cache
from typing import List, Any

logger = logging.getLogger(__name__)

LLM = ChatOllama(
    model="mistral",
    temperature=0.1,
    num_ctx=1024,
    num_thread=4,
    repeat_penalty=1.0
)

@lru_cache(maxsize=100)
def _build_concise_prompt(query: str, clauses: tuple) -> str:
    """Cacheable prompt builder"""
    clauses_str = chr(10).join(clauses)
    return f"""<s>[INST] You are an insurance policy expert. Answer in 1-2 sentences:
    
{clauses_str}

Question: {query}

Rules:
1. Be concise (1-2 sentences)
2. Start with Yes/No if applicable
3. Include only key details
4. Never say "refer to document"

Answer: [/INST]"""

async def answer_with_llm(query: str, top_docs: List[Any]) -> str:
    """Optimized LLM answering"""
    try:
        clauses = tuple(doc.page_content for doc in top_docs)
        prompt = _build_concise_prompt(query, clauses)
        chain = LLM | StrOutputParser()
        response = await chain.ainvoke(prompt)
        return response.strip() + ('' if response.endswith('.') else '.')
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return "Answer unavailable."