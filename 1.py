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
import hashlib
import aiohttp
import random
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def clean_answer_formatting(answer: str) -> str:
    """Clean answer formatting"""
    if not answer or not isinstance(answer, str):
        return answer

    cleaned = answer.replace('\\n', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\.{2,}', '.', cleaned)
    cleaned = re.sub(r'\s+\.', '.', cleaned)
    cleaned = re.sub(r'\.([A-Z])', r'. \1', cleaned)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)

    return cleaned.strip()


class AdaptiveKeyPool:
    def __init__(self, keys, primary_tpm=30000, fallback_tpm=10000, rpm_limit=30):
        self.keys = keys
        self.primary_tpm = primary_tpm
        self.fallback_tpm = fallback_tpm
        self.rpm_limit = rpm_limit
        self.lock = threading.Lock()

        self.key_stats = {}
        for key in keys:
            self.key_stats[key] = {
                'primary_tokens_used': 0,
                'fallback_tokens_used': 0,
                'requests_made': 0,
                'last_reset': time.time(),
                'consecutive_fails': 0,
                'is_cooling': False,
                'cooldown_until': 0,
                'last_used': 0,
                'avg_response_time': 1.0,
                'complex_queries': 0
            }

        self.total_requests = 0
        self.successful_requests = 0
        self.primary_model_used = 0
        self.fallback_model_used = 0

    def reset_counters_if_needed(self, key):
        now = time.time()
        stats = self.key_stats[key]

        if stats['is_cooling'] and now >= stats['cooldown_until']:
            stats['is_cooling'] = False
            stats['consecutive_fails'] = 0

        if now - stats['last_reset'] >= 60:
            stats['primary_tokens_used'] = 0
            stats['fallback_tokens_used'] = 0
            stats['requests_made'] = 0
            stats['last_reset'] = now

    def get_adaptive_key(self, estimated_tokens=1000, use_fallback=False, is_complex=False):
        with self.lock:
            tpm_limit = self.fallback_tpm if use_fallback else self.primary_tpm
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            if is_complex:
                estimated_tokens = int(estimated_tokens * 1.3)

            usage_limit = 0.85

            for key in self.keys:
                self.reset_counters_if_needed(key)

            usable_keys = []
            for key in self.keys:
                stats = self.key_stats[key]
                if stats['is_cooling']:
                    continue

                tokens_ok = stats[token_field] + estimated_tokens <= tpm_limit * usage_limit
                requests_ok = stats['requests_made'] < self.rpm_limit

                if tokens_ok and requests_ok:
                    load_score = (stats[token_field] / tpm_limit) + (stats['requests_made'] / self.rpm_limit)
                    usable_keys.append((key, load_score))

            if usable_keys:
                usable_keys.sort(key=lambda x: x[1])
                return usable_keys[0][0]

            # Emergency recovery
            for key in self.keys:
                stats = self.key_stats[key]
                stats['is_cooling'] = False
                stats['consecutive_fails'] = 0
                stats[token_field] = 0
                stats['requests_made'] = 0

            return self.keys[0]

    def record_success(self, key, tokens_used, response_time, use_fallback=False, was_complex=False):
        with self.lock:
            stats = self.key_stats[key]
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            stats[token_field] += tokens_used
            stats['requests_made'] += 1
            stats['consecutive_fails'] = 0
            stats['last_used'] = time.time()
            stats['avg_response_time'] = stats['avg_response_time'] * 0.7 + response_time * 0.3

            if was_complex:
                stats['complex_queries'] += 1

            self.total_requests += 1
            self.successful_requests += 1

            if use_fallback:
                self.fallback_model_used += 1
            else:
                self.primary_model_used += 1

    def record_failure(self, key, error_type="unknown"):
        with self.lock:
            stats = self.key_stats[key]
            stats['consecutive_fails'] += 1
            self.total_requests += 1

            if stats['consecutive_fails'] >= 3:
                stats['is_cooling'] = True
                stats['cooldown_until'] = time.time() + 25

    def get_pool_status(self):
        with self.lock:
            return {
                'total_keys': len(self.keys),
                'active_keys': sum(1 for k in self.keys if not self.key_stats[k]['is_cooling']),
                'success_rate': f"{(self.successful_requests / max(1, self.total_requests) * 100):.1f}%",
                'total_requests': self.total_requests,
                'primary_model_usage': f"{self.primary_model_used} (DeepSeek-R1)",
                'fallback_model_usage': f"{self.fallback_model_used} (Mistral-Small)"
            }


openrouter_keys = [
   os.getenv("OPEN_API_KEY_1"),
  os.getenv("OPEN_API_KEY_2"),
  os.getenv("OPEN_API_KEY_3"),
  os.getenv("OPEN_API_KEY_4"),
  os.getenv("OPEN_API_KEY_5"),
  os.getenv("OPEN_API_KEY_6"),
  os.getenv("OPEN_API_KEY_"),
]

openrouter_pool = AdaptiveKeyPool(openrouter_keys)

app = FastAPI(title="Accuracy-Focused Document Assistant", version="11.0.0")

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


def analyze_query_complexity(query: str) -> tuple[bool, str]:
    """Analyze query complexity"""
    query_lower = query.lower()

    complexity_indicators = [
        'definition', 'define', 'what does', 'what is', 'meaning',
        'conditions', 'requirements', 'what are the', 'list', 'explain',
        'all', 'comprehensive', 'complete', 'detailed',
        'how does', 'process', 'steps'
    ]

    simple_indicators = [
        'is', 'does', 'can', 'will', 'covered', 'period', 'limit', 'amount'
    ]

    complexity_score = 0

    for indicator in complexity_indicators:
        if indicator in query_lower:
            complexity_score += 1

    if len(query.split()) > 12:
        complexity_score += 1

    if query.count('?') > 1 or ' and ' in query or ' or ' in query:
        complexity_score += 1

    for simple in simple_indicators:
        if simple in query_lower and len(query.split()) <= 8:
            complexity_score = max(0, complexity_score - 1)
            break

    is_complex = complexity_score >= 2
    reason = 'complex' if is_complex else 'simple'

    return is_complex, reason


def precise_chunking(text: str) -> List[Dict]:
    """Create precise chunks focused on finding specific policy information"""
    chunks = []
    chunk_id = 0

    # Split by major sections first
    major_sections = []

    # Look for section markers
    section_patterns = [
        r'\n\s*(?:SECTION|Section|CLAUSE|Clause|ARTICLE|Article|PART|Part)\s*[A-Z0-9][A-Z0-9\.\-]*[:\s]',
        r'\n\s*\d+\.\s*[A-Z][^.]*(?::|\.)',
        r'\n\s*[A-Z]{3,}[:\s]',
        r'\n\s*(?:BENEFITS|EXCLUSIONS|CONDITIONS|DEFINITIONS|WAITING|GRACE|LIMIT|COVERAGE)[^\n]*[:]\s*',
    ]

    # Find all section breaks
    section_breaks = [0]
    for pattern in section_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            section_breaks.append(match.start())

    section_breaks = sorted(set(section_breaks))
    section_breaks.append(len(text))

    # Create chunks from sections
    for i in range(len(section_breaks) - 1):
        start = section_breaks[i]
        end = section_breaks[i + 1]
        section_text = text[start:end].strip()

        if len(section_text) < 50:
            continue

        # Determine chunk priority based on content
        priority = calculate_chunk_priority(section_text)
        chunk_type = determine_chunk_type(section_text)

        # Split large sections
        if len(section_text) > 2000:
            # Split by paragraphs
            paragraphs = section_text.split('\n\n')
            current_chunk = []
            current_length = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if current_length + len(para) > 1500 and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(create_precise_chunk(chunk_text, chunk_id, priority, chunk_type))
                    chunk_id += 1
                    current_chunk = [para]
                    current_length = len(para)
                else:
                    current_chunk.append(para)
                    current_length += len(para)

            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(create_precise_chunk(chunk_text, chunk_id, priority, chunk_type))
                chunk_id += 1
        else:
            chunks.append(create_precise_chunk(section_text, chunk_id, priority, chunk_type))
            chunk_id += 1

    # If we don't have enough chunks, add paragraph-based ones
    if len(chunks) < 30:
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) < 100:
                continue

            # Check if already covered
            already_covered = False
            for chunk in chunks:
                if para in chunk['text']:
                    already_covered = True
                    break

            if not already_covered:
                priority = calculate_chunk_priority(para)
                chunk_type = determine_chunk_type(para)
                chunks.append(create_precise_chunk(para, chunk_id, priority, chunk_type))
                chunk_id += 1

    # Sort by priority
    chunks.sort(key=lambda x: x['metadata']['priority'], reverse=True)

    logger.info(f"[PRECISE_CHUNKING] Created {len(chunks)} precision-focused chunks")

    return chunks[:150]  # Return more chunks for better coverage


def calculate_chunk_priority(text: str) -> int:
    """Calculate chunk priority based on content indicators"""
    text_lower = text.lower()
    priority = 50  # Base priority

    # High priority terms
    high_priority_terms = [
        'grace period', 'waiting period', 'pre-existing', 'maternity', 'cataract',
        'no claim discount', 'ncd', 'health check', 'hospital definition', 'ayush',
        'room rent', 'icu charges', 'sub-limit', 'organ donor', 'thirty days',
        '30 days', '36 months', '24 months', '5%', '10%', '1%', '2%'
    ]

    for term in high_priority_terms:
        if term in text_lower:
            priority += 20

    # Numbers and percentages
    if re.search(r'\d+\s*(?:days?|months?|years?)', text_lower):
        priority += 15

    if re.search(r'\d+\s*%', text_lower):
        priority += 15

    # Section markers
    if re.search(r'\b(?:section|clause|article|definition|exclusion|benefit)\b', text_lower):
        priority += 10

    # Specific amounts
    if re.search(r'(?:₹|rs\.?|rupees)\s*\d+', text_lower):
        priority += 10

    return min(priority, 100)


def determine_chunk_type(text: str) -> str:
    """Determine chunk type based on content"""
    text_lower = text.lower()

    if any(term in text_lower for term in ['definition', 'means', 'defined as', 'shall mean']):
        return 'definition'
    elif any(term in text_lower for term in ['exclusion', 'exclude', 'not covered', 'not pay']):
        return 'exclusion'
    elif any(term in text_lower for term in ['benefit', 'coverage', 'covered', 'pay', 'reimburse']):
        return 'benefit'
    elif any(term in text_lower for term in ['waiting period', 'grace period', 'period']):
        return 'period'
    elif any(term in text_lower for term in ['limit', 'maximum', 'minimum', 'sub-limit']):
        return 'limit'
    elif any(term in text_lower for term in ['condition', 'requirement', 'provided', 'subject to']):
        return 'condition'
    elif any(term in text_lower for term in ['procedure', 'process', 'claim', 'documents']):
        return 'procedure'
    else:
        return 'general'


def create_precise_chunk(text: str, chunk_id: int, priority: int, chunk_type: str) -> Dict:
    """Create a precise chunk with detailed metadata"""

    # Extract key information
    has_numbers = bool(re.search(r'\d+', text))
    has_percentages = bool(re.search(r'\d+\s*%', text))
    has_periods = bool(re.search(r'\d+\s*(?:day|month|year|hour)s?\b', text.lower()))
    has_amounts = bool(re.search(r'(?:₹|rs\.?|rupees|amount|sum|premium|limit)', text.lower()))

    # Key terms extraction
    key_terms = []

    # Policy-specific terms
    policy_patterns = [
        r'\b(?:grace\s+period|waiting\s+period|pre[\-\s]existing|maternity|cataract|no\s+claim\s+discount|ncd|health\s+check|ayush|organ\s+donor)\b',
        r'\b(?:room\s+rent|icu\s+charges|sub[\-\s]limit|hospital|definition|exclusion|benefit|coverage)\b',
        r'\b(?:thirty|30)\s+days?\b',
        r'\b(?:thirty[\-\s]six|36)\s+months?\b',
        r'\b(?:twenty[\-\s]four|24)\s+months?\b'
    ]

    for pattern in policy_patterns:
        matches = re.findall(pattern, text.lower())
        key_terms.extend(matches)

    # Extract numerical values
    numerical_values = re.findall(r'\d+(?:\.\d+)?\s*(?:%|days?|months?|years?|₹|rs\.?)', text.lower())

    return {
        'text': text,
        'metadata': {
            'id': chunk_id,
            'type': chunk_type,
            'priority': priority,
            'has_numbers': has_numbers,
            'has_percentages': has_percentages,
            'has_periods': has_periods,
            'has_amounts': has_amounts,
            'key_terms': key_terms[:10],
            'numerical_values': numerical_values[:5],
            'text_length': len(text)
        }
    }


def accuracy_focused_search(query: str, chunks: List[Dict], top_k: int = 15) -> List[Dict]:
    """Accuracy-focused search that finds the most relevant chunks"""
    query_lower = query.lower()
    query_words = [w.strip().lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 2]

    # Extract key search terms from query
    key_search_terms = extract_key_search_terms(query_lower)

    scored_chunks = []

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        metadata = chunk['metadata']
        score = 0

        # 1. EXACT PHRASE MATCHING (Highest Priority)
        exact_phrases = extract_exact_phrases(query_lower)
        for phrase in exact_phrases:
            if phrase in text_lower:
                score += 2000
                logger.debug(f"[EXACT_PHRASE] Found '{phrase}' in chunk {metadata.get('id', 0)}")

        # 2. KEY SEARCH TERMS MATCHING
        for term in key_search_terms:
            if term in text_lower:
                # Count occurrences
                count = text_lower.count(term)
                term_score = len(term) * 50 * count
                score += term_score
                logger.debug(f"[KEY_TERM] Found '{term}' {count} times: +{term_score}")

        # 3. QUERY WORD MATCHING with proximity
        matched_words = 0
        word_positions = {}

        for word in query_words:
            if word in text_lower:
                matched_words += 1
                positions = [m.start() for m in re.finditer(re.escape(word), text_lower)]
                word_positions[word] = positions

                # Base score for word match
                word_score = len(word) * 10 * len(positions)
                score += word_score

        # Word coverage bonus
        if matched_words > 0:
            coverage_ratio = matched_words / len(query_words)
            score += coverage_ratio * 1000

            if coverage_ratio >= 0.8:  # Most words matched
                score += 500
        else:
            # Heavy penalty for no query words
            continue

        # 4. PROXIMITY SCORING
        if len(word_positions) >= 2:
            proximity_bonus = calculate_proximity_bonus(word_positions)
            score += proximity_bonus

        # 5. NUMERICAL VALUE MATCHING
        if has_numerical_query(query):
            numerical_bonus = calculate_numerical_bonus(query, text_lower)
            score += numerical_bonus

        # 6. CHUNK TYPE RELEVANCE
        chunk_type = metadata.get('type', '')
        type_bonus = calculate_type_bonus(query_lower, chunk_type)
        score += type_bonus

        # 7. PRIORITY BONUS
        priority = metadata.get('priority', 0)
        score += priority * 5

        # 8. KEY TERMS IN METADATA
        chunk_key_terms = metadata.get('key_terms', [])
        for term in chunk_key_terms:
            if any(qt in term for qt in query_words):
                score += 100

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    if scored_chunks:
        logger.info(f"[ACCURACY_SEARCH] Found {len(scored_chunks)} relevant chunks")
        # Log top matches
        for i, (chunk, score) in enumerate(scored_chunks[:5]):
            preview = chunk['text'][:100].replace('\n', ' ')
            chunk_type = chunk['metadata'].get('type', 'unknown')
            logger.info(f"[ACCURACY_SEARCH] Rank {i + 1}: Score={score}, Type={chunk_type}, Preview='{preview}...'")
    else:
        logger.warning(f"[ACCURACY_SEARCH] No relevant chunks found for: '{query}'")

    # Return top chunks
    result_chunks = []
    for chunk, score in scored_chunks[:top_k]:
        chunk_copy = {
            'text': chunk['text'],
            'metadata': {**chunk['metadata'], 'search_score': score}
        }
        result_chunks.append(chunk_copy)

    return result_chunks


def extract_key_search_terms(query: str) -> List[str]:
    """Extract key search terms from query"""
    key_terms = []

    # Specific policy terms
    policy_terms = [
        'grace period', 'waiting period', 'pre-existing', 'maternity', 'cataract',
        'no claim discount', 'ncd', 'health check', 'hospital', 'ayush',
        'room rent', 'icu charges', 'sub-limit', 'organ donor'
    ]

    for term in policy_terms:
        if term in query:
            key_terms.append(term)

    # Time periods
    time_patterns = [
        r'(?:thirty|30)\s+days?',
        r'(?:thirty[\-\s]six|36)\s+months?',
        r'(?:twenty[\-\s]four|24)\s+months?',
        r'\d+\s+(?:days?|months?|years?)'
    ]

    for pattern in time_patterns:
        matches = re.findall(pattern, query)
        key_terms.extend(matches)

    # Percentages and amounts
    percentage_patterns = [
        r'\d+\s*%',
        r'(?:five|5)\s*percent',
        r'(?:ten|10)\s*percent'
    ]

    for pattern in percentage_patterns:
        matches = re.findall(pattern, query)
        key_terms.extend(matches)

    return key_terms


def extract_exact_phrases(query: str) -> List[str]:
    """Extract exact phrases that should be matched"""
    phrases = []

    # Common exact phrases in insurance queries
    exact_patterns = [
        r'grace\s+period\s+for\s+premium\s+payment',
        r'waiting\s+period\s+for\s+pre[\-\s]existing',
        r'no\s+claim\s+discount',
        r'health\s+check[\-\s]up',
        r'room\s+rent.*sub[\-\s]limit',
        r'icu\s+charges.*sub[\-\s]limit',
        r'organ\s+donor.*expenses',
        r'definition.*hospital'
    ]

    for pattern in exact_patterns:
        matches = re.findall(pattern, query)
        phrases.extend(matches)

    return phrases


def calculate_proximity_bonus(word_positions: Dict[str, List[int]]) -> int:
    """Calculate bonus for word proximity"""
    bonus = 0
    words_list = list(word_positions.keys())

    for i in range(len(words_list)):
        for j in range(i + 1, len(words_list)):
            word1, word2 = words_list[i], words_list[j]

            for pos1 in word_positions[word1]:
                for pos2 in word_positions[word2]:
                    distance = abs(pos1 - pos2)
                    if distance <= 50:  # Very close
                        bonus += 200
                    elif distance <= 100:  # Close
                        bonus += 100
                    elif distance <= 200:  # Moderate
                        bonus += 50

    return bonus


def has_numerical_query(query: str) -> bool:
    """Check if query contains numerical elements"""
    return bool(re.search(r'\d+|percent|%|days?|months?|years?|amount|limit|sub-limit', query.lower()))


def calculate_numerical_bonus(query: str, text: str) -> int:
    """Calculate bonus for numerical matching"""
    bonus = 0

    # Extract numbers from query
    query_numbers = re.findall(r'\d+', query)
    text_numbers = re.findall(r'\d+', text)

    for qnum in query_numbers:
        for tnum in text_numbers:
            if qnum == tnum:
                bonus += 300  # Exact number match

    # Time period matching
    time_terms = ['days', 'months', 'years', 'period']
    for term in time_terms:
        if term in query.lower() and term in text.lower():
            bonus += 200

    # Percentage matching
    if '%' in query or 'percent' in query.lower():
        if '%' in text or 'percent' in text.lower():
            bonus += 200

    return bonus


def calculate_type_bonus(query: str, chunk_type: str) -> int:
    """Calculate bonus based on chunk type relevance"""
    type_bonuses = {
        'definition': ['define', 'definition', 'what is', 'what does', 'meaning'],
        'exclusion': ['exclude', 'exclusion', 'not covered', 'not pay'],
        'benefit': ['benefit', 'coverage', 'covered', 'pay', 'reimburse'],
        'period': ['period', 'waiting', 'grace', 'time'],
        'limit': ['limit', 'maximum', 'minimum', 'sub-limit', 'cap'],
        'condition': ['condition', 'requirement', 'provided', 'subject to']
    }

    if chunk_type in type_bonuses:
        for term in type_bonuses[chunk_type]:
            if term in query:
                return 400

    return 0


class AccuracyRetriever:
    def __init__(self):
        self.chunks = []
        self.is_indexed = False
        self.doc_hash = None

    async def index_document(self, text: str):
        doc_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        if self.doc_hash == doc_hash and self.is_indexed:
            logger.info("[ACCURACY_RETRIEVER] Document already indexed")
            return

        start = time.time()
        self.chunks = precise_chunking(text)
        self.is_indexed = True
        self.doc_hash = doc_hash

        # Log chunk statistics
        chunk_types = defaultdict(int)
        for chunk in self.chunks:
            chunk_type = chunk['metadata'].get('type', 'unknown')
            chunk_types[chunk_type] += 1

        indexing_time = time.time() - start
        logger.info(f"[ACCURACY_RETRIEVER] Indexed {len(self.chunks)} chunks in {indexing_time:.2f}s")
        logger.info(f"[ACCURACY_RETRIEVER] Types: {dict(chunk_types)}")

    async def get_accurate_context(self, query: str, top_k: int = 15) -> str:
        """Get accurate context for the query"""
        if not self.is_indexed:
            return "Document not indexed"

        start_time = time.time()

        # Use accuracy-focused search
        relevant_chunks = accuracy_focused_search(query, self.chunks, top_k=top_k)

        if not relevant_chunks:
            logger.warning("[ACCURACY_RETRIEVER] No relevant chunks found, using high-priority fallbacks")
            high_priority_chunks = sorted(
                self.chunks,
                key=lambda x: x['metadata'].get('priority', 0),
                reverse=True
            )[:top_k]
            relevant_chunks = high_priority_chunks

        # Build context with focus on accuracy
        context_parts = []
        total_chars = 0
        max_context = 6000  # Larger context for accuracy
        chunks_used = 0

        for i, chunk in enumerate(relevant_chunks):
            text = chunk['text']
            metadata = chunk['metadata']
            search_score = metadata.get('search_score', 0)
            chunk_type = metadata.get('type', 'general')

            # Use more text for high-scoring chunks
            if search_score > 1500:
                chunk_text = text[:2000] if len(text) > 2000 else text
                label = f"[EXACT-{i + 1}]"
            elif search_score > 800:
                chunk_text = text[:1500] if len(text) > 1500 else text
                label = f"[HIGH-{i + 1}]"
            elif search_score > 400:
                chunk_text = text[:1000] if len(text) > 1000 else text
                label = f"[MED-{i + 1}]"
            else:
                chunk_text = text[:800] if len(text) > 800 else text
                label = f"[{chunk_type.upper()}-{i + 1}]"

            if total_chars + len(chunk_text) > max_context:
                remaining = max_context - total_chars
                if remaining > 300:
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(f"{label} {chunk_text}")
                    chunks_used += 1
                break

            context_parts.append(f"{label} {chunk_text}")
            total_chars += len(chunk_text)
            chunks_used += 1

        context = "\n\n".join(context_parts)
        retrieval_time = time.time() - start_time

        logger.info(
            f"[ACCURACY_RETRIEVER] Context: {len(context)} chars from {chunks_used} chunks in {retrieval_time:.2f}s")

        return context


# Global retriever
retriever = AccuracyRetriever()

# ACCURACY-FOCUSED PROMPTS
ACCURACY_PRIMARY_PROMPT = """You are an expert insurance policy analyst. Your job is to extract EXACT information from policy documents with complete accuracy.

RESPONSE FORMAT (JSON):
{
  "answer": "Precise answer with exact numbers, timeframes, and conditions as stated in the document",
  "confidence": "high | medium | low"
}

CRITICAL ACCURACY RULES:
1. **EXACT NUMBERS**: Use the EXACT numbers, percentages, and timeframes from the document (e.g., "thirty (30) days", "5%", "36 months")
2. **SPECIFIC REFERENCES**: Include section numbers, clause references when available
3. **COMPLETE CONDITIONS**: Include ALL conditions, limitations, and requirements mentioned
4. **DIRECT QUOTES**: When possible, use direct quotes from the policy text
5. **NO ASSUMPTIONS**: If specific information isn't in the context, say so clearly

EXAMPLES OF ACCURATE ANSWERS:
✓ "The grace period for premium payment is thirty (30) days from the due date, as specified in the policy terms."
✓ "Pre-existing diseases have a waiting period of thirty-six (36) months of continuous coverage."
✓ "The No Claim Discount is 5% per claim-free year, with a maximum cumulative discount of 50%."
✓ "Room rent is limited to 1% of the Sum Insured per day for Plan A."

CONFIDENCE LEVELS:
- HIGH: Information is explicitly stated with exact details
- MEDIUM: Information is clearly implied or partially stated
- LOW: Information is unclear or requires interpretation

Focus on COMPLETE ACCURACY. Extract the exact information as written in the policy document."""

ACCURACY_FALLBACK_PROMPT = """You are conducting a thorough analysis of an insurance policy document. Provide the most accurate and complete answer possible.

RESPONSE FORMAT (JSON):
{
  "answer": "Comprehensive, accurate answer with all specific details from the document"
}

THOROUGH ANALYSIS REQUIREMENTS:
1. **EXAMINE ALL CONTEXT**: Carefully review every section provided
2. **EXTRACT SPECIFICS**: Find exact numbers, percentages, timeframes, amounts
3. **IDENTIFY CONDITIONS**: List all qualifying conditions and requirements
4. **REFERENCE SECTIONS**: Cite specific clauses, sections, or articles
5. **COMPLETE COVERAGE**: Address all aspects of the question

ACCURACY PRIORITIES:
- Use exact wording from the policy document
- Include all numerical values and timeframes
- Mention all conditions and limitations
- Reference specific policy sections
- Provide complete, not partial, information

This is a detailed analysis requiring maximum accuracy. Extract and present all relevant information from the document context."""


async def analyze_with_accuracy_focus(query: str, context: str, is_complex: bool = False) -> Dict[str, Any]:
    """Accuracy-focused analysis"""
    try:
        # Ensure adequate context
        if len(context) > 15000:
            context = context[:15000] + "..."

        base_tokens = int(len(context + query) / 3.5) + 200
        estimated_tokens = int(base_tokens * 1.3) if is_complex else base_tokens

        # Primary analysis
        api_key = openrouter_pool.get_adaptive_key(estimated_tokens, use_fallback=False, is_complex=is_complex)

        timeout = 50
        max_tokens = 600

        payload = {
            "model": "tngtech/deepseek-r1t2-chimera:free",
            "messages": [
                {"role": "system", "content": ACCURACY_PRIMARY_PROMPT},
                {"role": "user",
                 "content": f"POLICY QUESTION: {query}\n\nPOLICY DOCUMENT CONTEXT:\n{context}\n\nExtract the exact information with complete accuracy:"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,  # Maximum determinism for accuracy
            "max_tokens": max_tokens
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:

                if response.status == 429:
                    openrouter_pool.record_failure(api_key, "rate_limit")
                    raise Exception("Primary model rate limited")
                elif response.status >= 400:
                    openrouter_pool.record_failure(api_key, f"http_{response.status}")
                    raise Exception(f"Primary model HTTP {response.status}")

                result = await response.json()

        tokens_used = result.get("usage", {}).get("total_tokens", estimated_tokens)
        response_time = time.time() - start_time
        openrouter_pool.record_success(api_key, tokens_used, response_time, use_fallback=False, was_complex=is_complex)

        # Parse response
        content = result["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            raw_answer = parsed.get("answer", "Unable to determine from provided context")
            answer = clean_answer_formatting(raw_answer)
            confidence = parsed.get("confidence", "medium").lower()
        except json.JSONDecodeError:
            answer = clean_answer_formatting(content)
            confidence = "low"

        elapsed = time.time() - start_time

        # Use fallback for low confidence or vague answers
        accuracy_issues = [
            "not mentioned", "not specified", "not available", "not provided",
            "unable to determine", "cannot determine", "not clearly stated",
            "document does not specify", "not explicitly mentioned",
            "not found", "insufficient information", "not described"
        ]

        should_use_fallback = (
                confidence == "low" or
                len(answer) < 40 or
                any(issue in answer.lower() for issue in accuracy_issues)
        )

        if should_use_fallback:
            logger.info(f"[ACCURACY_ANALYSIS] Using fallback for better accuracy - Primary confidence: {confidence}")

            try:
                fallback_key = openrouter_pool.get_adaptive_key(estimated_tokens, use_fallback=True, is_complex=True)

                fallback_payload = {
                    "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                    "messages": [
                        {"role": "system", "content": ACCURACY_FALLBACK_PROMPT},
                        {"role": "user",
                         "content": f"QUESTION: {query}\n\nPOLICY CONTEXT:\n{context}\n\nProvide thorough, accurate analysis - primary analysis had {confidence} confidence:"}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                    "max_tokens": 600
                }

                fallback_headers = {
                    "Authorization": f"Bearer {fallback_key}",
                    "Content-Type": "application/json"
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            json=fallback_payload,
                            headers=fallback_headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                    ) as fallback_response:

                        if fallback_response.status >= 400:
                            openrouter_pool.record_failure(fallback_key, f"http_{fallback_response.status}")
                        else:
                            fallback_result = await fallback_response.json()
                            fallback_tokens_used = fallback_result.get("usage", {}).get("total_tokens",
                                                                                        estimated_tokens)
                            fallback_response_time = time.time() - start_time
                            openrouter_pool.record_success(fallback_key, fallback_tokens_used, fallback_response_time,
                                                     use_fallback=True, was_complex=True)

                            fallback_content = fallback_result["choices"][0]["message"]["content"]
                            try:
                                fallback_parsed = json.loads(fallback_content)
                                fallback_answer = clean_answer_formatting(fallback_parsed.get("answer", answer))

                                # Use fallback if it's more specific and accurate
                                if (len(fallback_answer) > len(answer) + 20 or
                                        not any(issue in fallback_answer.lower() for issue in accuracy_issues)):
                                    answer = fallback_answer
                                    confidence = "medium"
                                    logger.info("[ACCURACY_ANALYSIS] Using fallback analysis for better accuracy")

                            except json.JSONDecodeError:
                                logger.warning("[ACCURACY_ANALYSIS] Fallback JSON parse failed")

            except Exception as e:
                logger.warning(f"[ACCURACY_ANALYSIS] Fallback failed: {e}")

        model_used = "DeepSeek-R1"
        if should_use_fallback and confidence == "medium":
            model_used = "DeepSeek-R1 + Accuracy-Mistral-Small"

        return {
            "answer": answer,
            "success": True,
            "tokens_used": tokens_used,
            "time_seconds": round(elapsed, 2),
            "confidence": confidence,
            "model_used": model_used
        }

    except Exception as e:
        logger.error(f"[ACCURACY_ANALYSIS] Analysis failed: {str(e)}")
        return {
            "answer": "Analysis failed",
            "success": False,
            "error": str(e)
        }


def extract_pdf_text(pdf_url: str) -> str:
    try:
        logger.info("[PDF_EXTRACTOR] Downloading PDF...")
        response = requests.get(pdf_url, timeout=90)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        os.unlink(temp_path)

        logger.info(f"[PDF_EXTRACTOR] Extracted {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"[PDF_EXTRACTOR] Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")


@app.get("/")
def root():
    return {
        "message": "Accuracy-Focused Document Assistant v11.0",
        "primary_model": "tngtech/deepseek-r1t2-chimera:free",
        "fallback_model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "design": "Maximum accuracy with precise information extraction",
        "features": [
            "Precision-focused chunking",
            "Accuracy-focused search",
            "Exact information extraction",
            "Comprehensive context analysis",
            "Zero temperature for determinism"
        ],
        "improvements": [
            "Better chunk prioritization for policy terms",
            "Enhanced search for exact matches",
            "Focused on specific numbers and conditions",
            "Comprehensive context coverage",
            "Deterministic analysis"
        ],
        "focus": "Complete accuracy over speed or brevity",
        "keys_available": len(openrouter_keys),
        "user": "Jay121305",
        "timestamp": "2025-08-04 16:11:47"
    }


@app.get("/pool-status")
def get_pool_status():
    return openrouter_pool.get_pool_status()


@app.post("/api/v1/hackrx/run")
async def hackathon_endpoint(
        request: HackathonRequest,
        authorization: Optional[str] = Header(None)
):
    try:
        start_time = time.time()
        num_questions = len(request.questions)
        logger.info(f"[ACCURACY_SYSTEM] Processing {num_questions} questions with maximum accuracy focus")

        # Extract and index document
        pdf_text = extract_pdf_text(request.documents)
        await retriever.index_document(pdf_text)

        async def process_single_question(i, question):
            question_start = time.time()

            is_complex, complexity_reason = analyze_query_complexity(question)

            complexity_indicator = "🔍" if is_complex else "🎯"
            logger.info(f"[Q{i + 1}] {complexity_indicator} {question[:80]}...")

            # Get accurate context
            context = await retriever.get_accurate_context(question, top_k=15)

            # Analyze with accuracy focus
            result = await analyze_with_accuracy_focus(question, context, is_complex)

            answer = result.get("answer", "")

            # Quality check - retry if answer is clearly inadequate
            accuracy_issues = [
                "not mentioned", "not specified", "unable to determine",
                "document does not specify", "not explicitly mentioned",
                "not found", "insufficient information"
            ]

            if any(issue in answer.lower() for issue in accuracy_issues) and len(answer) < 60:
                logger.warning(f"[Q{i + 1}] Accuracy issue detected, attempting enhanced retrieval")

                # Try with maximum context
                enhanced_context = await retriever.get_accurate_context(question, top_k=15)
                enhanced_result = await analyze_with_accuracy_focus(question, enhanced_context, True)

                enhanced_answer = enhanced_result.get("answer", "")
                if (len(enhanced_answer) > len(answer) + 30 and
                        not any(issue in enhanced_answer.lower() for issue in accuracy_issues[:3])):
                    answer = enhanced_answer
                    result = enhanced_result
                    logger.info(f"[Q{i + 1}] Enhanced retrieval provided more accurate answer")

            final_answer = clean_answer_formatting(answer)

            elapsed = time.time() - question_start
            tokens = result.get('tokens_used', 0)
            model_used = result.get('model_used', 'Unknown')
            confidence = result.get('confidence', 'unknown')

            timing_emoji = "🕐" if elapsed > 3.0 else "⚡"
            logger.info(f"[Q{i + 1}] ✓ {timing_emoji} {elapsed:.1f}s | {tokens}t | {confidence} | {model_used}")

            return final_answer

        # Sequential processing for maximum accuracy
        semaphore = asyncio.Semaphore(4)  # Limited concurrency for accuracy

        async def limited_process(i, question):
            async with semaphore:
                # Add delay for stability
                await asyncio.sleep(i * 0.5)
                return await process_single_question(i, question)

        tasks = [limited_process(i, q) for i, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        final_answers = []
        for answer in answers:
            if isinstance(answer, Exception):
                final_answers.append("Processing failed")
                logger.error(f"[ACCURACY_SYSTEM] Exception: {answer}")
            else:
                final_answers.append(str(answer))

        total_time = time.time() - start_time
        success_count = sum(1 for a in final_answers if a != "Processing failed")

        logger.info(f"[ACCURACY_SYSTEM] COMPLETED - ACCURACY-FOCUSED PROCESSING")
        logger.info(f"[ACCURACY_SYSTEM] ⏱️  Total: {total_time:.1f}s ({total_time / num_questions:.2f}s/q)")
        logger.info(
            f"[ACCURACY_SYSTEM] ✅ Success: {success_count}/{num_questions} ({success_count / num_questions * 100:.1f}%)")
        logger.info(f"[ACCURACY_SYSTEM] 🎯 Maximum accuracy processing with precise information extraction")

        return {"answers": final_answers}

    except Exception as e:
        logger.error(f"[ACCURACY_SYSTEM] Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Accuracy-focused processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("[ACCURACY_SYSTEM] Starting Accuracy-Focused Document Assistant v11.0")
    logger.info("[ACCURACY_SYSTEM] Design: Maximum accuracy with precise information extraction")
    logger.info("[ACCURACY_SYSTEM] Focus: Complete accuracy over speed - deterministic analysis")
    logger.info("[ACCURACY_SYSTEM] Features: Precision chunking, exact matching, comprehensive context")
    logger.info(f"[ACCURACY_SYSTEM] User: Jay121305 | Time: 2025-08-04 16:11:47")
    uvicorn.run(app, host="0.0.0.0", port=8000)
