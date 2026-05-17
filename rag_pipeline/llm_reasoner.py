from mistralai import Mistral
import os
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = "mistral-tiny"

SYSTEM_PROMPT = """You are an insurance policy expert. Provide clear, conversational answers in plain text.

IMPORTANT RULES:
1. Use plain text ONLY - absolutely NO markdown formatting
2. NO asterisks (**), NO bold, NO italic, NO special symbols
3. Write naturally like you're explaining to a friend
4. Be specific with numbers, dates, and amounts
5. Keep answers concise (1-3 sentences)
6. If unsure, say "This is not specified in the policy"

GOOD EXAMPLES:
✓ "The deductible is $500 per claim period."
✓ "Coverage includes emergency hospitalization, OPD treatment, and personal accident benefits."
✓ "Return air ambulance transportation is not covered."

BAD EXAMPLES (DO NOT DO THIS):
✗ "The **deductible** is *$500*."
✗ "Coverage includes: **emergency care**, **OPD**, etc."
✗ "**Not covered**: return flights"
"""

USER_PROMPT = """Policy Information:
{context}

Question: {question}

Answer in plain, conversational text (no formatting):"""


def clean_markdown(text: str) -> str:
    """Remove all markdown formatting from text"""
    
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
   
    text = re.sub(r'#{1,6}\s+', '', text)
    
    
    text = re.sub(r'`{1,3}(.+?)`{1,3}', r'\1', text)
    
    
    text = re.sub(r'^\s*[-\*]\s+', '', text, flags=re.MULTILINE)
    
   
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


async def answer_with_llm(query: str, retrieval_result: Dict) -> Dict:
    """
    Generate answer with confidence and citations
    
    Args:
        query: User question
        retrieval_result: Output from retrieve_with_rerank()
        
    Returns:
        {
            'answer': str (plain text, no markdown),
            'confidence': float,
            'sources': List[dict],
            'quality': str
        }
    """
    try:
        chunks = retrieval_result['chunks']
        
        if not chunks:
            return {
                'answer': "No relevant policy clauses found.",
                'confidence': 0.0,
                'sources': [],
                'quality': 'poor'
            }
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                context="\n\n".join(chunks[:3]),
                question=query
            )}
        ]
        
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )
        
        answer = response.choices[0].message.content.strip()
        
       
        answer = clean_markdown(answer)
        
      
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        
        avg_score = sum(retrieval_result['scores']) / len(retrieval_result['scores'])
        
        if avg_score >= 0.8:
            quality = 'excellent'
        elif avg_score >= 0.6:
            quality = 'good'
        elif avg_score >= 0.4:
            quality = 'fair'
        else:
            quality = 'poor'
        
        
        sources = []
        for i, (chunk, meta) in enumerate(zip(chunks, retrieval_result['metadata'])):
            sources.append({
                'index': i + 1,
                'page': meta.get('page', 'Unknown'),
                'relevance': round(meta['score'] * 100, 1),
                'excerpt': chunk[:150] + '...' if len(chunk) > 150 else chunk
            })
        
        logger.info(f"✅ Generated clean answer: {answer[:50]}...")
        
        return {
            'answer': answer,
            'confidence': retrieval_result['confidence'],
            'sources': sources,
            'quality': quality
        }
        
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return {
            'answer': "Unable to generate an answer at this time.",
            'confidence': 0.0,
            'sources': [],
            'quality': 'error'
        }


async def stream_llm_answer(query: str, retrieval_result: Dict):
    """Stream tokens from Mistral with markdown cleaning"""
    try:
        chunks = retrieval_result['chunks']
        
        if not chunks:
            yield "No relevant policy clauses found."
            return
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                context="\n\n".join(chunks[:3]),
                question=query
            )}
        ]
        
        response = client.chat.stream(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )
        
        full_text = ""
        for chunk in response:
            if chunk.data.choices[0].delta.content:
                token = chunk.data.choices[0].delta.content
                full_text += token
                
                
                clean_token = token.replace('**', '').replace('*', '')
                yield clean_token
                
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield "Error generating answer."