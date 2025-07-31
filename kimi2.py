from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio
import os
import logging
import requests
import tempfile
import fitz  # Only for PDF text extraction
from dotenv import load_dotenv
import time
import threading
import re
import hashlib

# Minimal dependencies - no ML libraries!
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


# --- Multi-key pool for GROQ API keys ---
class GroqAPIKeyPool:
    def __init__(self, keys, cooldown=60):
        self.keys = keys
        self.cooldown = cooldown
        self.lock = threading.Lock()
        self.cooling = {}  # key: available_time
        self.next_key_idx = 0

    def get_key(self):
        with self.lock:
            now = time.time()
            n = len(self.keys)
            for _ in range(n):
                idx = self.next_key_idx
                key = self.keys[idx]
                if key not in self.cooling or self.cooling[key] <= now:
                    self.next_key_idx = (idx + 1) % n
                    return key
                self.next_key_idx = (idx + 1) % n
            # All keys cooling, pick soonest
            soonest_key = min(self.keys, key=lambda k: self.cooling.get(k, 0))
            wait = max(0, self.cooling[soonest_key] - now)
        if wait > 0:
            logger.warning(f"[Jay121305] All API keys cooling down, waiting {wait:.1f}s...")
            time.sleep(wait)
        return soonest_key

    def mark_rate_limited(self, key):
        with self.lock:
            self.cooling[key] = time.time() + self.cooldown


# List your Groq API keys here
groq_keys = [
    "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",
    "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",
    "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",
    "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"
]
groq_pool = GroqAPIKeyPool(groq_keys)

app = FastAPI(title="LLM-Based Semantic Insurance Assistant", version="5.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]


# --- LLM-Based Semantic Retrieval System (No Embeddings Needed) ---

def extract_section_header(text: str) -> str:
    """Extract likely section header from text"""
    lines = text.split('\n')[:3]  # First 3 lines
    for line in lines:
        if any(keyword in line.lower() for keyword in ['clause', 'section', 'article', 'chapter']):
            return line.strip()
        if re.match(r'^\d+\.', line.strip()):
            return line.strip()
    return lines[0][:100] if lines else ""


def smart_chunk_document(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict]:
    """Smart chunking for better semantic retrieval"""
    chunks = []

    # Split by logical sections (clauses, numbered items, etc.)
    section_patterns = [
        r'\n\d+\.\d+\s',  # 1.1, 2.3, etc.
        r'\nClause\s+\d+',  # Clause 1, Clause 2
        r'\nSection\s+\d+',  # Section 1, Section 2
        r'\n[A-Z][A-Z\s]{10,50}:',  # ALL CAPS headers
        r'\n\([a-z]\)',  # (a), (b), (c)
        r'\n[A-Z]{2,}\s*[-:]',  # DEFINITIONS:, BENEFITS:, etc.
        r'\n\d+\.\s',  # 1. 2. 3. etc.
    ]

    # First try to split by sections
    sections = [text]
    for pattern in section_patterns:
        new_sections = []
        for section in sections:
            parts = re.split(pattern, section)
            new_sections.extend([s.strip() for s in parts if s.strip()])
        sections = new_sections

    # Now chunk each section
    for i, section in enumerate(sections):
        if len(section) <= chunk_size:
            chunks.append({
                'text': section,
                'section_id': i,
                'metadata': {
                    'section_header': extract_section_header(section),
                    'chunk_type': 'complete_section',
                    'word_count': len(section.split())
                }
            })
        else:
            # Split large sections with overlap
            words = section.split()
            chunk_word_size = chunk_size // 5  # Approximate words
            overlap_words = overlap // 5

            for j in range(0, len(words), chunk_word_size - overlap_words):
                chunk_words = words[j:j + chunk_word_size]
                chunk_text = ' '.join(chunk_words)

                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'section_id': i,
                        'metadata': {
                            'section_header': extract_section_header(section),
                            'chunk_type': 'partial_section',
                            'chunk_index': j,
                            'word_count': len(chunk_words)
                        }
                    })

    logger.info(f"[Jay121305] Smart chunking created {len(chunks)} chunks")
    return chunks


async def llm_based_relevance_scoring(query: str, chunks: List[Dict], top_k: int = 6) -> List[Dict]:
    """Use LLM to score chunk relevance - no embeddings needed"""
    try:
        # Create a batch of chunks for evaluation
        chunk_texts = []
        for i, chunk in enumerate(chunks[:20]):  # Limit to first 20 chunks for efficiency
            chunk_preview = chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text']
            chunk_texts.append(f"CHUNK {i + 1}: {chunk_preview}")

        combined_chunks = "\n\n".join(chunk_texts)

        payload = {
            "model": "llama-3.1-70b-versatile",  # Fast model for relevance scoring
            "messages": [
                {
                    "role": "system",
                    "content": """You are a document relevance analyzer. Given a query and document chunks, score each chunk's relevance to the query on a scale of 1-10.

Return ONLY a JSON array of scores like: [8, 3, 9, 2, 7, 1, 4, 6, 5, 3]

Score criteria:
- 9-10: Directly answers the query with specific details
- 7-8: Contains relevant information related to the query
- 5-6: Somewhat related but missing key details
- 3-4: Tangentially related
- 1-2: Not relevant"""
                },
                {
                    "role": "user",
                    "content": f"""QUERY: {query}

DOCUMENT CHUNKS:
{combined_chunks}

Score each chunk's relevance (1-10). Return only the JSON array of scores."""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 500
        }

        api_key = groq_pool.get_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=20
        )
        response.raise_for_status()

        result = response.json()["choices"][0]["message"]["content"]
        parsed_result = json.loads(result)

        # Extract scores array
        scores = parsed_result.get("scores", []) or list(parsed_result.values())[0] if parsed_result else []

        # Combine chunks with scores
        scored_chunks = []
        for i, chunk in enumerate(chunks[:len(scores)]):
            score = scores[i] if i < len(scores) else 5
            scored_chunks.append({
                'chunk': chunk,
                'relevance_score': score
            })

        # Sort by relevance score and return top chunks
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.info(
            f"[Jay121305] LLM relevance scoring completed, top score: {scored_chunks[0]['relevance_score'] if scored_chunks else 0}")

        return [item['chunk'] for item in scored_chunks[:top_k]]

    except Exception as e:
        logger.error(f"[Jay121305] LLM relevance scoring failed: {e}")
        # Fallback to keyword-based selection
        return chunks[:top_k]


class LLMSemanticRetriever:
    def __init__(self):
        self.chunks = []
        self.is_indexed = False
        self.document_hash = None

    async def index_document(self, pdf_text: str):
        """Index document using smart chunking"""
        # Check if already indexed
        current_hash = hashlib.md5(pdf_text.encode()).hexdigest()[:16]
        if self.document_hash == current_hash and self.is_indexed:
            logger.info(f"[Jay121305] Document already indexed (hash: {current_hash})")
            return

        logger.info("[Jay121305] Starting LLM-based semantic indexing...")
        start_time = time.time()

        # Chunk the document
        self.chunks = smart_chunk_document(pdf_text)
        self.is_indexed = True
        self.document_hash = current_hash

        elapsed = time.time() - start_time
        logger.info(f"[Jay121305] Indexed {len(self.chunks)} chunks in {elapsed:.2f}s")

    async def retrieve_relevant_chunks(self, query: str, top_k: int = 6) -> str:
        """Retrieve most relevant chunks using LLM-based scoring"""
        if not self.is_indexed:
            logger.warning("[Jay121305] Document not indexed, using fallback")
            return "Document not properly indexed for semantic search."

        # Use LLM to score relevance
        relevant_chunks = await llm_based_relevance_scoring(query, self.chunks, top_k)

        if not relevant_chunks:
            # Fallback to keyword matching
            relevant_chunks = self.keyword_fallback(query, top_k)

        # Combine relevant chunks with metadata
        combined_sections = []
        for i, chunk in enumerate(relevant_chunks):
            section_text = f"[Rank: {i + 1}] [Section: {chunk['metadata'].get('section_header', 'Unknown')}]\n{chunk['text']}"
            combined_sections.append(section_text)

        return "\n\n--- RELEVANT SECTION ---\n\n".join(combined_sections)

    def keyword_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Keyword-based fallback for chunk selection"""
        query_words = [word.lower().strip('.,!?') for word in query.split()
                       if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'how', 'the', 'and', 'for']]

        scored_chunks = []
        for chunk in self.chunks:
            score = sum(1 for word in query_words if word in chunk['text'].lower())
            if re.search(r'\d+', chunk['text']):  # Boost chunks with numbers
                score += 2
            if score > 0:
                scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:top_k]]


# Initialize global retriever
semantic_retriever = LLMSemanticRetriever()

# Enhanced system prompt
ENHANCED_SYSTEM_PROMPT = """
You are a domain-specific document analyst. Extract the most accurate answer based strictly on the provided document sections.

CRITICAL: Each section is marked with [Rank: X] showing relevance ranking and [Section: Header] showing document structure.

RESPONSE FORMAT (JSON):
{
  "answer": "Detailed answer with specific numbers, percentages, time periods, and all relevant conditions",
  "confidence": "high | medium | low",
  "supporting_clauses": ["Section references"],
  "matched_text": ["Exact supporting text excerpts"],
  "reasoning": "Step-by-step explanation referencing ranked sections"
}

RULES:
- Prioritize higher-ranked sections ([Rank: 1] is most relevant)
- Include ALL specific details: numbers, percentages, conditions, limits
- Extract exact text that supports your answer in "matched_text"
- Be comprehensive and precise with numerical values
- Look for information in tables, lists, definitions, and examples
- Cross-reference multiple sections if needed
- Never say "not mentioned" - analyze all provided sections thoroughly
"""


def extract_pdf_text_only(pdf_url: str) -> str:
    """Lightweight PDF extraction - no ML processing"""
    try:
        logger.info(f"[Jay121305] Downloading PDF: {pdf_url[:50]}...")

        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)
        text = "\n\n".join([page.get_text() for page in doc])
        doc.close()
        os.unlink(temp_path)

        logger.info(f"[Jay121305] Extracted {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"[Jay121305] PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")


def optimize_context_for_tokens(context: str, max_chars: int = 90000) -> str:
    """Optimize context to fit token limits"""
    if len(context) <= max_chars:
        return context

    # Remove boilerplate patterns
    boilerplate_patterns = [
        r'This document is.*?\n',
        r'Disclaimer:.*?\n',
        r'Copyright.*?\n',
        r'All rights reserved.*?\n',
        r'For internal use only.*?\n'
    ]

    for pattern in boilerplate_patterns:
        context = re.sub(pattern, '', context, flags=re.IGNORECASE)

    # If still too long, keep only top-ranked sections
    sections = context.split("--- RELEVANT SECTION ---")
    if len(sections) > 1:
        # Keep top 4 sections
        optimized_context = "\n\n--- RELEVANT SECTION ---\n\n".join(sections[:5])
        if len(optimized_context) <= max_chars:
            return optimized_context

    # Final fallback: simple truncation
    return context[:max_chars] + "\n\n[Context truncated for token limits...]"


async def enhanced_cloud_analysis(query: str, relevant_context: str, model_id: str = "moonshotai/kimi-k2-instruct") -> \
Dict[str, Any]:
    """Enhanced cloud analysis with LLM-ranked context"""
    try:
        # Optimize context for token efficiency
        optimized_context = optimize_context_for_tokens(relevant_context)

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": ENHANCED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
QUESTION: {query}

DOCUMENT SECTIONS (Ranked by LLM Relevance):
{optimized_context}

Analyze this question based on the LLM-ranked document sections and provide your response in the specified JSON format."""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.05,
            "max_tokens": 2000
        }

        t0 = time.time()
        api_key = groq_pool.get_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"[Jay121305] Using API key: ...{api_key[-5:]} for enhanced analysis")

        max_attempts = 4
        last_exception = None
        tokens_used = None

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                try:
                    tokens_used = response.json().get("usage", {}).get("total_tokens")
                except Exception:
                    tokens_used = None
                break

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if hasattr(response, "status_code") and response.status_code == 429:
                    logger.warning(f"[Jay121305] Rate limited, marking key and retrying...")
                    groq_pool.mark_rate_limited(api_key)
                    if attempt < max_attempts - 1:
                        continue
                logger.error(f"[Jay121305] HTTP error in enhanced analysis: {str(e)}")
                raise e
            except Exception as ex:
                logger.error(f"[Jay121305] Unexpected error in enhanced analysis: {str(ex)}")
                last_exception = ex
        else:
            logger.error(f"[Jay121305] All attempts failed in enhanced analysis.")
            if last_exception:
                raise last_exception
            raise Exception("Unknown error in enhanced analysis")

        elapsed = time.time() - t0
        result = response.json()["choices"][0]["message"]["content"]

        try:
            parsed_result = json.loads(result)
            confidence_label = parsed_result.get("confidence", "medium")
            confidence_score = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(confidence_label, 0.6)

            return {
                "answer": parsed_result.get("answer", "Unable to determine"),
                "confidence": f"{confidence_label} ({confidence_score})",
                "confidence_score": confidence_score,
                "supporting_clauses": parsed_result.get("supporting_clauses", []),
                "matched_text": parsed_result.get("matched_text", []),
                "reasoning": parsed_result.get("reasoning", ""),
                "model_used": model_id,
                "processing_type": "llm_semantic",
                "context_length": len(optimized_context),
                "success": True,
                "tokens_used": tokens_used,
                "time_used_seconds": round(elapsed, 2),
                "api_key_used": api_key[-5:]
            }
        except json.JSONDecodeError:
            return {
                "answer": result,
                "confidence": "medium (0.6)",
                "confidence_score": 0.6,
                "model_used": model_id,
                "processing_type": "llm_semantic_fallback",
                "success": True,
                "tokens_used": tokens_used,
                "time_used_seconds": round(elapsed, 2),
                "api_key_used": api_key[-5:]
            }

    except Exception as e:
        logger.error(f"[Jay121305] Enhanced analysis failed: {str(e)}")
        return {
            "answer": f"Enhanced analysis failed: {str(e)}",
            "confidence": "low (0.3)",
            "confidence_score": 0.3,
            "success": False,
            "tokens_used": None,
            "time_used_seconds": None,
            "api_key_used": None,
            "model_used": model_id
        }


@app.get("/")
def root():
    return {
        "message": "LLM-Based Semantic Insurance Assistant v5.1",
        "status": "running",
        "architecture": "Pure cloud processing with LLM-based semantic retrieval",
        "features": ["Smart chunking", "LLM relevance scoring", "Enhanced context optimization"],
        "memory_usage": "< 100MB",
        "dependencies": ["FastAPI", "Requests", "PyMuPDF"],
        "user": "Jay121305",
        "timestamp": "2025-07-31 18:45:30"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "processing_type": "llm_semantic_cloud",
        "api_available": True,
        "semantic_retriever": semantic_retriever.is_indexed,
        "timestamp": "2025-07-31 18:45:30"
    }


@app.post("/api/v1/hackrx/run")
async def hackathon_endpoint(
        request: HackathonRequest,
        authorization: Optional[str] = Header(None)
):
    """
    Enhanced hackathon endpoint with LLM-based semantic retrieval
    """
    try:
        if authorization:
            logger.info(f"[Jay121305] Authorization header received: {authorization[:20]}...")
        else:
            logger.warning(f"[Jay121305] No authorization header provided")

        logger.info(f"[Jay121305] LLM-semantic hackathon request: {len(request.questions)} questions")
        logger.info(f"[Jay121305] Document URL: {request.documents[:50]}...")

        pdf_text = extract_pdf_text_only(request.documents)
        logger.info(f"[Jay121305] PDF extracted successfully: {len(pdf_text)} characters")

        # Index document for LLM-based semantic search
        await semantic_retriever.index_document(pdf_text)

        answers = []

        # Tracking stats
        success_count = 0
        fallback_count = 0
        failed_count = 0

        # Define models
        primary_model = "moonshotai/kimi-k2-instruct"
        fallback_model = "llama-3.1-70b-versatile"

        for i, question in enumerate(request.questions):
            logger.info(f"[Jay121305] Processing question {i + 1}/{len(request.questions)}: {question[:100]}...")

            # Use LLM-based semantic retrieval
            relevant_context = await semantic_retriever.retrieve_relevant_chunks(question, top_k=6)

            # Try with primary model first
            result_primary = await enhanced_cloud_analysis(question, relevant_context, model_id=primary_model)

            if result_primary.get("success", False) and result_primary.get("confidence_score", 0.0) >= 0.4:
                answer_text = result_primary.get("answer", "Unable to determine from the policy document.")
                success_count += 1

                logger.info(
                    f"[Jay121305] [Q{i + 1}/{len(request.questions)}] SUCCESS | "
                    f"confidence: {result_primary.get('confidence')} | "
                    f"time: {result_primary.get('time_used_seconds')}s"
                )
                answers.append(answer_text)

            elif result_primary.get("success", False):
                # Fallback to secondary model if confidence is low
                logger.info(f"[Jay121305] Low confidence, trying fallback model")
                result_fallback = await enhanced_cloud_analysis(question, relevant_context, model_id=fallback_model)

                if (result_fallback.get("success", False) and
                        result_fallback.get("confidence_score", 0.0) > result_primary.get("confidence_score", 0.0)):
                    answer_text = result_fallback.get("answer", "Unable to determine from the policy document.")
                    fallback_count += 1
                    logger.info(f"[Jay121305] [Q{i + 1}] FALLBACK BETTER")
                    answers.append(answer_text)
                else:
                    answer_text = result_primary.get("answer", "Unable to determine from the policy document.")
                    logger.info(f"[Jay121305] [Q{i + 1}] PRIMARY KEPT")
                    answers.append(answer_text)
            else:
                answer_text = "Analysis failed: Could not process with available models."
                failed_count += 1
                logger.error(f"[Jay121305] [Q{i + 1}] FAILED")
                answers.append(answer_text)

        logger.info(f"[Jay121305] All {len(request.questions)} questions processed")
        logger.info(
            f"[Jay121305] Stats -- Success: {success_count}, Fallback: {fallback_count}, Failed: {failed_count}")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"[Jay121305] LLM-semantic endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"[Jay121305] Starting LLM-Based Semantic Insurance Assistant v5.1")
    uvicorn.run(app, host="0.0.0.0", port=8000)
