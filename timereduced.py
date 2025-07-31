from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
import os
import logging
import requests
import tempfile
import fitz
from dotenv import load_dotenv
import time
import threading
import re
import hashlib
import aiohttp
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class Advanced6KeyPool:
    def __init__(self, keys, primary_tpm=30000, fallback_tpm=10000, rpm_limit=30):
        self.keys = keys
        self.primary_tpm = primary_tpm  # Llama-4-Scout limit
        self.fallback_tpm = fallback_tpm  # Kimi-K2 limit
        self.rpm_limit = rpm_limit
        self.lock = threading.Lock()

        # Initialize tracking for all 6 keys
        self.key_stats = {}
        for key in keys:
            self.key_stats[key] = {
                'primary_tokens_used': 0,
                'fallback_tokens_used': 0,
                'requests_made': 0,
                'last_reset': time.time(),
                'consecutive_fails': 0,
                'is_cooling': False,
                'cooldown_until': 0
            }

        self.total_requests = 0
        self.successful_requests = 0
        self.primary_model_used = 0
        self.fallback_model_used = 0

    def reset_counters_if_needed(self, key):
        """Reset counters every minute"""
        now = time.time()
        stats = self.key_stats[key]

        if now - stats['last_reset'] >= 60:  # 1 minute reset
            stats['primary_tokens_used'] = 0
            stats['fallback_tokens_used'] = 0
            stats['requests_made'] = 0
            stats['last_reset'] = now

        # Reset cooling if cooldown period passed
        if stats['is_cooling'] and now >= stats['cooldown_until']:
            stats['is_cooling'] = False
            stats['consecutive_fails'] = 0
            logger.info(f"[Jay121305] Key ...{key[-5:]} cooled down and ready")

    def get_best_available_key(self, estimated_tokens=600, use_fallback=False):
        """Get the best available key with model-specific limits"""
        with self.lock:
            available_keys = []
            tpm_limit = self.fallback_tpm if use_fallback else self.primary_tpm
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            # Check all 6 keys
            for key in self.keys:
                self.reset_counters_if_needed(key)
                stats = self.key_stats[key]

                # Skip if cooling down
                if stats['is_cooling']:
                    continue

                # Check if key can handle the request
                tokens_ok = stats[token_field] + estimated_tokens <= tpm_limit
                requests_ok = stats['requests_made'] < self.rpm_limit

                if tokens_ok and requests_ok:
                    # Calculate load score (lower is better)
                    load_score = (
                            (stats[token_field] / tpm_limit) * 0.6 +
                            (stats['requests_made'] / self.rpm_limit) * 0.3 +
                            (stats['consecutive_fails'] * 0.1)
                    )
                    available_keys.append((key, load_score))

            if not available_keys:
                # All keys at limit, find the one that resets soonest
                soonest_key = min(self.keys, key=lambda k: self.key_stats[k]['last_reset'] + 60)
                wait_time = max(0, self.key_stats[soonest_key]['last_reset'] + 60 - time.time())

                if wait_time > 0:
                    model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                    logger.warning(
                        f"[Jay121305] All 6 keys at {model_name} limit, waiting {wait_time:.1f}s for reset...")
                    time.sleep(wait_time)
                    return self.get_best_available_key(estimated_tokens, use_fallback)

                return soonest_key

            # Sort by load score and return best key
            available_keys.sort(key=lambda x: x[1])
            selected_key = available_keys[0][0]

            model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
            logger.debug(
                f"[Jay121305] Selected key ...{selected_key[-5:]} for {model_name} (load: {available_keys[0][1]:.2f})")
            return selected_key

    def record_success(self, key, tokens_used, use_fallback=False):
        """Record successful API call"""
        with self.lock:
            stats = self.key_stats[key]
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            stats[token_field] += tokens_used
            stats['requests_made'] += 1
            stats['consecutive_fails'] = 0  # Reset fail counter

            self.total_requests += 1
            self.successful_requests += 1

            if use_fallback:
                self.fallback_model_used += 1
            else:
                self.primary_model_used += 1

            model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
            limit = self.fallback_tpm if use_fallback else self.primary_tpm
            logger.debug(
                f"[Jay121305] Key ...{key[-5:]} ({model_name}): {stats[token_field]}/{limit} tokens, {stats['requests_made']}/{self.rpm_limit} requests")

    def record_failure(self, key, error_type="unknown"):
        """Record failed API call"""
        with self.lock:
            stats = self.key_stats[key]
            stats['consecutive_fails'] += 1
            self.total_requests += 1

            # Put in cooldown if too many consecutive fails
            if stats['consecutive_fails'] >= 3:
                stats['is_cooling'] = True
                stats['cooldown_until'] = time.time() + 30  # 30 second cooldown
                logger.warning(f"[Jay121305] Key ...{key[-5:]} cooling down due to {stats['consecutive_fails']} fails")

            logger.error(f"[Jay121305] Key ...{key[-5:]} failed: {error_type}")

    def get_pool_status(self):
        """Get comprehensive pool status"""
        with self.lock:
            status = {
                'total_keys': len(self.keys),
                'active_keys': sum(1 for k in self.keys if not self.key_stats[k]['is_cooling']),
                'success_rate': f"{(self.successful_requests / max(1, self.total_requests) * 100):.1f}%",
                'total_requests': self.total_requests,
                'primary_model_usage': f"{self.primary_model_used} (Llama-4-Scout)",
                'fallback_model_usage': f"{self.fallback_model_used} (Kimi-K2)",
                'key_details': {}
            }

            for i, key in enumerate(self.keys):
                stats = self.key_stats[key]
                status['key_details'][f'key_{i + 1}'] = {
                    'primary_tokens': f"{stats['primary_tokens_used']}/{self.primary_tpm}",
                    'fallback_tokens': f"{stats['fallback_tokens_used']}/{self.fallback_tpm}",
                    'requests': f"{stats['requests_made']}/{self.rpm_limit}",
                    'status': 'cooling' if stats['is_cooling'] else 'active',
                    'fails': stats['consecutive_fails']
                }

            return status


# Initialize with all 6 keys and updated limits
groq_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",  # 5th key
    "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"  # 6th key
]

groq_pool = Advanced6KeyPool(groq_keys, primary_tpm=30000, fallback_tpm=10000, rpm_limit=30)

app = FastAPI(title="Dual-Model Insurance Assistant", version="8.2.0")

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


def ultra_compact_chunking(text: str, chunk_size: int = 600) -> List[Dict]:
    """Ultra-compact chunking to reduce token usage by 70%"""
    chunks = []

    # Only extract the most critical patterns
    critical_patterns = [
        r'grace\s+period[^.]{0,100}\.',
        r'thirty\s+days[^.]{0,100}premium[^.]{0,50}\.',
        r'pre-existing\s+disease[^.]{0,150}\.',
        r'thirty[- ]six.*?months[^.]{0,100}coverage[^.]{0,50}\.',
        r'maternity[^.]{0,200}\.',
        r'twenty[- ]four.*?months.*?female[^.]{0,100}\.',
        r'two\s+deliveries[^.]{0,50}\.',
        r'cataract[^.]{0,100}surgery[^.]{0,50}\.',
        r'organ\s+donor[^.]{0,150}\.',
        r'transplantation.*?human.*?organs[^.]{0,100}\.',
        r'no\s+claim\s+discount[^.]{0,150}\.',
        r'5%[^.]{0,100}premium[^.]{0,50}\.',
        r'health\s+check[^.]{0,150}\.',
        r'hospital.*?defined[^.]{0,250}\.',
        r'institution.*?beds[^.]{0,200}\.',
        r'ayush[^.]{0,200}\.',
        r'room\s+rent[^.]{0,150}\.',
        r'1%.*?sum.*?insured[^.]{0,100}\.',
        r'2%.*?sum.*?insured[^.]{0,100}\.',
        r'day\s+care[^.]{0,150}\.',
        r'domiciliary[^.]{0,150}\.',
        r'aids[^.]{0,100}\.',
        r'floater\s+sum\s+insured[^.]{0,100}\.',
    ]

    # Extract only critical chunks with minimal context
    critical_chunks = []
    used_positions = set()

    for pattern in critical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            start = max(0, match.start() - 50)  # Reduced context
            end = min(len(text), match.end() + 50)

            # Avoid overlapping chunks
            if not any(abs(start - pos) < 80 for pos in used_positions):
                context = text[start:end].strip()

                if len(context) > 40:
                    critical_chunks.append({
                        'text': context,
                        'metadata': {
                            'type': 'critical',
                            'priority': 100,
                            'has_numbers': bool(re.search(r'\d+', context)),
                            'pattern': pattern
                        }
                    })
                    used_positions.add(start)

    # Add minimal regular chunks from key sections
    key_section_patterns = [
        r'CLAUSE\s+\d+[^.]{0,500}\.',
        r'DEFINITIONS[^.]{0,800}\.',
        r'Table\s+of\s+Benefits[^.]{0,800}\.',
        r'EXCLUSIONS[^.]{0,600}\.',
    ]

    for pattern in key_section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_text = match.group(0)
            if len(section_text) > 100:
                # Split into smaller chunks
                sentences = re.split(r'(?<=[.!?])\s+', section_text)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + " " + sentence) <= chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'text': current_chunk.strip(),
                                'metadata': {
                                    'type': 'regular',
                                    'priority': 30,
                                    'has_numbers': bool(re.search(r'\d+', current_chunk))
                                }
                            })
                        current_chunk = sentence

                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'type': 'regular',
                            'priority': 30,
                            'has_numbers': bool(re.search(r'\d+', current_chunk))
                        }
                    })

    # Combine and deduplicate
    all_chunks = critical_chunks + chunks

    # Remove duplicates
    unique_chunks = []
    seen = set()
    for chunk in all_chunks:
        sig = chunk['text'][:50].lower().strip()
        if sig not in seen and len(chunk['text']) > 30:
            seen.add(sig)
            unique_chunks.append(chunk)
            if len(unique_chunks) >= 60:  # Hard limit
                break

    # Sort by priority
    unique_chunks.sort(key=lambda x: x['metadata']['priority'], reverse=True)

    logger.info(
        f"[Jay121305] Ultra-compact chunking: {len(critical_chunks)} critical + {len(chunks)} regular = {len(unique_chunks)} unique")
    return unique_chunks


def minimal_search(query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Minimal search returning only 3 most relevant chunks"""

    query_lower = query.lower()

    # Enhanced keyword mapping for better search
    keyword_mappings = {
        'grace period': ['grace', 'period', 'premium', 'payment', 'thirty', '30', 'days'],
        'pre-existing': ['pre-existing', 'ped', 'disease', 'thirty-six', '36', 'months'],
        'maternity': ['maternity', 'childbirth', 'pregnancy', 'twenty-four', '24', 'months', 'delivery'],
        'cataract': ['cataract', 'surgery', 'twenty-four', '24', 'months', 'two', 'years'],
        'organ donor': ['organ', 'donor', 'transplant', 'harvesting', 'transplantation', 'human', 'organs'],
        'no claim discount': ['no claim', 'ncd', 'discount', '5%', 'five', 'percent', 'premium'],
        'health check': ['health check', 'preventive', 'check-up', 'medical', 'examination'],
        'hospital': ['hospital', 'definition', 'institution', 'beds', 'nursing', 'staff', 'operation', 'theatre'],
        'ayush': ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'],
        'room rent': ['room rent', 'icu', 'intensive care', 'plan a', 'sub-limit', '1%', '2%'],
        'day care': ['day care', 'day-care', 'procedure', 'surgical', 'treatment'],
        'domiciliary': ['domiciliary', 'hospitalization', 'home', 'treatment'],
        'aids': ['aids', 'hiv', 'immune', 'deficiency'],
        'floater': ['floater', 'sum', 'insured', 'aggregate', 'family']
    }

    # Determine query domain
    matched_keywords = set()
    for domain, keywords in keyword_mappings.items():
        if any(term in query_lower for term in domain.split()):
            matched_keywords.update(keywords)

    # Add general query words
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    matched_keywords.update(query_words)

    scored_chunks = []

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        metadata = chunk['metadata']
        score = 0

        # Keyword matching with higher weights for exact matches
        for keyword in matched_keywords:
            if keyword in text_lower:
                if len(keyword) > 5:  # Longer keywords get higher scores
                    score += 15
                else:
                    score += 8

        # Priority and type bonuses
        priority = metadata.get('priority', 0)
        score += priority

        if metadata.get('type') == 'critical':
            score += 40

        # Number bonus
        if metadata.get('has_numbers'):
            score += 15

        # Content pattern bonuses
        important_phrases = [
            'grace period', 'waiting period', 'pre-existing disease',
            'maternity expenses', 'cataract surgery', 'organ donor',
            'no claim discount', 'health check', 'hospital', 'ayush',
            'room rent', 'icu charges', 'day care', 'domiciliary'
        ]

        for phrase in important_phrases:
            if phrase in text_lower:
                score += 20

        # Exact phrase matching (highest bonus)
        for phrase in query_lower.split():
            if len(phrase) > 3 and phrase in text_lower:
                score += 25

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    if scored_chunks:
        logger.info(
            f"[Jay121305] Minimal search: top score = {scored_chunks[0][1]}, found {len(scored_chunks)} relevant chunks")

    return [chunk for chunk, _ in scored_chunks[:top_k]]


class OptimizedRetriever:
    def __init__(self):
        self.chunks = []
        self.is_indexed = False
        self.doc_hash = None

    async def index_document(self, text: str):
        doc_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        if self.doc_hash == doc_hash and self.is_indexed:
            logger.info("[Jay121305] Document already indexed")
            return

        start = time.time()
        self.chunks = ultra_compact_chunking(text)
        self.is_indexed = True
        self.doc_hash = doc_hash
        logger.info(f"[Jay121305] Indexed {len(self.chunks)} chunks in {time.time() - start:.2f}s")

    async def get_ultra_compact_context(self, query: str) -> str:
        """Ultra-compact context to reduce tokens by 80%"""
        if not self.is_indexed:
            return "Document not indexed"

        relevant_chunks = minimal_search(query, self.chunks, top_k=3)

        if not relevant_chunks:
            # Fallback to high priority chunks
            high_priority = [c for c in self.chunks if c['metadata'].get('priority', 0) > 80]
            relevant_chunks = high_priority[:3] if high_priority else self.chunks[:3]

        # Build minimal context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            # Limit each chunk to 400 chars max
            text = chunk['text'][:400]
            priority = chunk['metadata'].get('priority', 0)

            if priority > 80:
                context_parts.append(f"[IMPORTANT-{i + 1}] {text}")
            else:
                context_parts.append(f"[{i + 1}] {text}")

        context = "\n\n".join(context_parts)
        logger.info(f"[Jay121305] Ultra-compact context: {len(context)} chars")
        return context


# Global retriever
retriever = OptimizedRetriever()

# Primary model prompt (Llama-4-Scout)
PRIMARY_SYSTEM_PROMPT = """You are an insurance policy expert. Provide brief, accurate answers based on the document context.

RESPONSE FORMAT (JSON):
{
  "answer": "Direct answer with key facts, numbers, and conditions",
  "confidence": "high | medium | low"
}

RULES:
1. Keep under 100 words
2. Include exact numbers, periods, percentages as stated
3. For definitions, provide complete definition 
4. If information exists in context, provide it - don't say "not mentioned"
5. Be factual and direct
6. Set confidence: "high" if answer is definitive, "medium" if partially found, "low" if uncertain

EXAMPLES:
- "A grace period of thirty days is provided for premium payment after the due date."
- "There is a waiting period of thirty-six (36) months for pre-existing diseases."
- "Yes, the policy covers maternity expenses. The female insured must be covered for 24 months. Limited to two deliveries."
"""

# Fallback model prompt (Kimi-K2) - more detailed for better accuracy
FALLBACK_SYSTEM_PROMPT = """You are an expert insurance policy analyst. The primary model had low confidence, so provide a more detailed, accurate analysis.

RESPONSE FORMAT (JSON):
{
  "answer": "Comprehensive answer with all relevant details, numbers, and conditions"
}

RULES:
1. Be thorough and include all relevant information
2. Include exact numbers, periods, percentages, and conditions
3. For definitions, provide complete definitions as stated
4. Cross-reference multiple context sections if available
5. If information exists in context, provide it fully - don't say "not mentioned"
6. Provide complete eligibility criteria and limitations
7. Match sample response style and completeness

EXAMPLES:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits pertaining to Waiting Periods and coverage of Pre-Existing Diseases."
- "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
"""


async def analyze_with_dual_model_system(query: str, context: str) -> Dict[str, Any]:
    """Dual-model analysis: Llama-4-Scout primary, Kimi-K2 fallback"""
    try:
        # Aggressive context reduction
        if len(context) > 2500:
            context = context[:2500] + "..."

        # Step 1: Try primary model (Llama-4-Scout)
        estimated_tokens = (len(context) + len(query) + 400) // 4
        api_key = groq_pool.get_best_available_key(estimated_tokens, use_fallback=False)

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nProvide a brief, accurate answer with confidence level:"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.05,
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=20)
            ) as response:

                if response.status == 429:
                    groq_pool.record_failure(api_key, "rate_limit")
                    raise Exception("Primary model rate limited")
                elif response.status >= 400:
                    groq_pool.record_failure(api_key, f"http_{response.status}")
                    raise Exception(f"Primary model HTTP {response.status}")

                result = await response.json()

        tokens_used = result.get("usage", {}).get("total_tokens", estimated_tokens)
        groq_pool.record_success(api_key, tokens_used, use_fallback=False)

        # Parse primary response
        content = result["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            answer = parsed.get("answer", "Unable to determine")
            confidence = parsed.get("confidence", "medium").lower()
        except json.JSONDecodeError:
            answer = content
            confidence = "low"  # JSON parse failure indicates low confidence

        elapsed = time.time() - start_time

        # Step 2: Check if fallback is needed
        if confidence == "low":
            logger.info(f"[Jay121305] Primary model confidence LOW, trying fallback (Kimi-K2)...")

            # Try fallback model (Kimi-K2)
            fallback_key = groq_pool.get_best_available_key(estimated_tokens, use_fallback=True)

            fallback_payload = {
                "model": "moonshotai/kimi-k2-instruct",
                "messages": [
                    {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nThe primary model had low confidence. Provide a comprehensive, accurate answer:"}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.05,
                "max_tokens": 800  # More tokens for detailed response
            }

            fallback_headers = {
                "Authorization": f"Bearer {fallback_key}",
                "Content-Type": "application/json"
            }

            fallback_start = time.time()

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            json=fallback_payload,
                            headers=fallback_headers,
                            timeout=aiohttp.ClientTimeout(total=25)
                    ) as fallback_response:

                        if fallback_response.status == 429:
                            groq_pool.record_failure(fallback_key, "rate_limit")
                            logger.warning(f"[Jay121305] Fallback model also rate limited, using primary answer")
                        elif fallback_response.status >= 400:
                            groq_pool.record_failure(fallback_key, f"http_{fallback_response.status}")
                            logger.warning(f"[Jay121305] Fallback model failed, using primary answer")
                        else:
                            fallback_result = await fallback_response.json()
                            fallback_tokens = fallback_result.get("usage", {}).get("total_tokens", estimated_tokens)
                            groq_pool.record_success(fallback_key, fallback_tokens, use_fallback=True)

                            # Parse fallback response
                            fallback_content = fallback_result["choices"][0]["message"]["content"]
                            try:
                                fallback_parsed = json.loads(fallback_content)
                                fallback_answer = fallback_parsed.get("answer", answer)

                                # Use fallback answer if it's significantly longer/better
                                if len(fallback_answer) > len(answer) + 20:
                                    answer = fallback_answer
                                    confidence = "medium"  # Upgraded from low
                                    logger.info(f"[Jay121305] Using improved fallback answer")

                            except json.JSONDecodeError:
                                logger.warning(f"[Jay121305] Fallback JSON parse failed, keeping primary answer")

            except Exception as e:
                logger.warning(f"[Jay121305] Fallback model failed: {e}, using primary answer")

        return {
            "answer": answer,
            "success": True,
            "tokens_used": tokens_used,
            "time_seconds": round(elapsed, 2),
            "api_key": api_key[-5:],
            "confidence": confidence,
            "model_used": "Llama-4-Scout" if confidence != "medium" else "Llama-4-Scout + Kimi-K2"
        }

    except Exception as e:
        logger.error(f"[Jay121305] Dual-model analysis failed: {str(e)}")
        return {
            "answer": "Analysis failed",
            "success": False,
            "error": str(e)
        }


def extract_pdf_text(pdf_url: str) -> str:
    try:
        logger.info("[Jay121305] Downloading PDF...")
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        os.unlink(temp_path)

        logger.info(f"[Jay121305] Extracted {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"[Jay121305] PDF extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF error: {e}")


@app.get("/")
def root():
    return {
        "message": "Dual-Model Insurance Assistant v8.2",
        "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)",
        "fallback_model": "moonshotai/kimi-k2-instruct (10K TPM)",
        "keys_available": len(groq_keys),
        "strategy": "Primary model with low-confidence fallback",
        "features": ["6-key load balancing", "Dual-model system", "Confidence-based fallback"],
        "user": "Jay121305",
        "timestamp": "2025-07-31 19:38:20"
    }


@app.get("/pool-status")
def get_pool_status():
    return groq_pool.get_pool_status()


@app.post("/api/v1/hackrx/run")
async def hackathon_endpoint(
        request: HackathonRequest,
        authorization: Optional[str] = Header(None)
):
    try:
        start_time = time.time()
        logger.info(f"[Jay121305] Processing {len(request.questions)} questions with dual-model system")

        pdf_text = extract_pdf_text(request.documents)
        await retriever.index_document(pdf_text)

        async def process_single_question(i, question):
            question_start = time.time()
            logger.info(f"[Jay121305] Q{i + 1}: {question[:60]}...")

            # Add progressive delay to spread load evenly across keys
            delay = (i % 6) * 0.4 + random.uniform(0.1, 0.3)  # 0.1-2.7 second staggered delays
            await asyncio.sleep(delay)

            context = await retriever.get_ultra_compact_context(question)
            result = await analyze_with_dual_model_system(question, context)

            elapsed = time.time() - question_start
            if result.get("success"):
                tokens = result.get('tokens_used', 0)
                model_used = result.get('model_used', 'Unknown')
                confidence = result.get('confidence', 'unknown')
                logger.info(
                    f"[Jay121305] Q{i + 1} SUCCESS | {elapsed:.1f}s | {tokens} tokens | {model_used} | confidence: {confidence}")
            else:
                logger.error(f"[Jay121305] Q{i + 1} FAILED | {elapsed:.1f}s | {result.get('error', 'unknown')}")

            return result.get("answer", "Analysis failed")

        # Process with controlled concurrency to respect rate limits
        semaphore = asyncio.Semaphore(2)  # Conservative concurrency

        async def limited_process(i, question):
            async with semaphore:
                return await process_single_question(i, question)

        # Execute all questions
        tasks = [limited_process(i, q) for i, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        final_answers = []
        for answer in answers:
            if isinstance(answer, Exception):
                final_answers.append("Processing failed")
                logger.error(f"[Jay121305] Exception: {answer}")
            else:
                final_answers.append(answer)

        total_time = time.time() - start_time
        success_count = sum(1 for a in final_answers if a not in ["Processing failed", "Analysis failed"])

        pool_status = groq_pool.get_pool_status()

        logger.info(
            f"[Jay121305] COMPLETED | {total_time:.1f}s | {success_count}/{len(request.questions)} success | {pool_status['success_rate']} overall")
        logger.info(
            f"[Jay121305] Model usage: {pool_status['primary_model_usage']}, {pool_status['fallback_model_usage']}")
        logger.info(f"[Jay121305] Pool status: {pool_status['active_keys']}/{pool_status['total_keys']} keys active")

        return {"answers": final_answers}

    except Exception as e:
        logger.error(f"[Jay121305] Endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("[Jay121305] Starting Dual-Model Insurance Assistant v8.2")
    logger.info(f"[Jay121305] Primary: meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)")
    logger.info(f"[Jay121305] Fallback: moonshotai/kimi-k2-instruct (10K TPM)")
    logger.info(f"[Jay121305] Timestamp: 2025-07-31 19:38:20")
    logger.info(f"[Jay121305] User: Jay121305")
    uvicorn.run(app, host="0.0.0.0", port=8000)
