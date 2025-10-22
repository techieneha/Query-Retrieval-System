from mistralai import Mistral
import os
import logging
from typing import List

logger = logging.getLogger(__name__)


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = "mistral-small-latest"  
async def answer_with_llm(query: str, context_clauses: List[str]) -> str:
    """Generate precise answers using Mistral API"""
    try:
        if not context_clauses:
            return "No relevant policy clauses found."
        
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    context="\n".join(context_clauses[:3]),
                    question=query
                )
            }
        ]
        
        
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )
        
        
        answer = response.choices[0].message.content
        return answer.strip() + ('' if answer.endswith('.') else '.')
    except Exception as e:
        logger.error(f"Mistral API error: {str(e)}")
        return "Answer unavailable due to API error."

SYSTEM_PROMPT = """You are an insurance policy expert. Provide concise answers:
1. Extract exact numbers/dates/amounts
2. 1-2 sentences maximum
3. Never say "refer to document"
4. If unsure: "Not specified in policy\""""

USER_PROMPT = """**Relevant Policy Clauses:**
{context}

**Question:**
{question}

**Answer:**"""