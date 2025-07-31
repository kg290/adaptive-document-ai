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


def clean_answer_formatting(answer: str) -> str:
    """Clean up answer formatting to remove literal \n characters and improve readability"""
    if not answer or not isinstance(answer, str):
        return answer

    # Replace literal \n with actual spaces
    cleaned = answer.replace('\\n', ' ')

    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Clean up common formatting issues
    cleaned = cleaned.replace(' - ', '. ')  # Replace bullet-like dashes
    cleaned = cleaned.replace(' • ', '. ')  # Replace bullet points

    # Remove excessive punctuation
    cleaned = re.sub(r'\.{2,}', '.', cleaned)  # Multiple dots to single
    cleaned = re.sub(r'\s+\.', '.', cleaned)  # Space before period

    # Ensure proper spacing after periods
    cleaned = re.sub(r'\.([A-Z])', r'. \1', cleaned)

    # Remove markdown-style formatting
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold markers
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)  # Remove italic markers

    # Clean up any remaining whitespace issues
    cleaned = cleaned.strip()

    return cleaned


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
                'avg_response_time': 1.0,  # Track response times for adaptive timing
                'complex_queries': 0  # Track complex query handling
            }

        self.total_requests = 0
        self.successful_requests = 0
        self.primary_model_used = 0
        self.fallback_model_used = 0
        self.current_key_index = 0

    def reset_counters_if_needed(self, key):
        """Reset counters every minute"""
        now = time.time()
        stats = self.key_stats[key]

        if now - stats['last_reset'] >= 60:
            stats['primary_tokens_used'] = 0
            stats['fallback_tokens_used'] = 0
            stats['requests_made'] = 0
            stats['last_reset'] = now
            logger.info(f"[Jay121305] Key ...{key[-5:]} tokens reset for new minute")

        if stats['is_cooling'] and now >= stats['cooldown_until']:
            stats['is_cooling'] = False
            stats['consecutive_fails'] = 0
            logger.info(f"[Jay121305] Key ...{key[-5:]} cooled down and ready")

    def get_adaptive_key(self, estimated_tokens=1000, use_fallback=False, is_complex=False):
        """Adaptive key selection with complexity awareness"""
        with self.lock:
            tpm_limit = self.fallback_tpm if use_fallback else self.primary_tpm
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            # Adjust token estimation for complex queries
            if is_complex:
                estimated_tokens = int(estimated_tokens * 1.3)  # 30% more tokens for complex queries

            # Round-robin selection with complexity awareness
            for _ in range(len(self.keys)):
                key_idx = self.current_key_index
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                key = self.keys[key_idx]

                self.reset_counters_if_needed(key)
                stats = self.key_stats[key]

                if stats['is_cooling']:
                    continue

                # Adaptive limits based on complexity
                usage_limit = 0.85 if not is_complex else 0.75  # More conservative for complex queries
                tokens_ok = stats[token_field] + estimated_tokens <= tpm_limit * usage_limit
                requests_ok = stats['requests_made'] < self.rpm_limit

                if tokens_ok and requests_ok:
                    model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                    complexity_note = " (complex)" if is_complex else ""
                    logger.debug(f"[Jay121305] Selected key ...{key[-5:]} for {model_name}{complexity_note}")
                    return key

            # Fallback selection
            available_keys = []
            for key in self.keys:
                self.reset_counters_if_needed(key)
                stats = self.key_stats[key]

                if stats['is_cooling']:
                    continue

                usage_limit = 0.85 if not is_complex else 0.75
                tokens_ok = stats[token_field] + estimated_tokens <= tpm_limit * usage_limit
                requests_ok = stats['requests_made'] < self.rpm_limit

                if tokens_ok and requests_ok:
                    load_score = stats[token_field] / tmp_limit + stats['requests_made'] / self.rpm_limit
                    available_keys.append((key, load_score))

            if available_keys:
                available_keys.sort(key=lambda x: x[1])
                return available_keys[0][0]

            # Wait with adaptive timing
            soonest_key = min(self.keys, key=lambda k: self.key_stats[k]['last_reset'] + 60)
            wait_time = max(0, self.key_stats[soonest_key]['last_reset'] + 60 - time.time())

            if wait_time > 0:
                model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                max_wait = 15 if is_complex else 10  # Longer wait for complex queries if needed
                logger.warning(f"[Jay121305] All keys at {model_name} limit, waiting {wait_time:.1f}s...")
                time.sleep(min(wait_time, max_wait))
                return self.get_adaptive_key(estimated_tokens, use_fallback, is_complex)

            return soonest_key

    def record_success(self, key, tokens_used, response_time, use_fallback=False, was_complex=False):
        """Record successful API call with timing and complexity tracking"""
        with self.lock:
            stats = self.key_stats[key]
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            stats[token_field] += tokens_used
            stats['requests_made'] += 1
            stats['consecutive_fails'] = 0
            stats['last_used'] = time.time()

            # Update average response time with exponential moving average
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
        """Record failed API call"""
        with self.lock:
            stats = self.key_stats[key]
            stats['consecutive_fails'] += 1
            self.total_requests += 1

            if stats['consecutive_fails'] >= 3:
                stats['is_cooling'] = True
                stats['cooldown_until'] = time.time() + 25
                logger.warning(f"[Jay121305] Key ...{key[-5:]} cooling down (25s)")

    def get_pool_status(self):
        """Get pool status with adaptive metrics"""
        with self.lock:
            total_primary_capacity = len(self.keys) * self.primary_tpm
            total_fallback_capacity = len(self.keys) * self.fallback_tpm

            avg_response_time = sum(stats['avg_response_time'] for stats in self.key_stats.values()) / len(self.keys)
            total_complex_queries = sum(stats['complex_queries'] for stats in self.key_stats.values())

            return {
                'total_keys': len(self.keys),
                'active_keys': sum(1 for k in self.keys if not self.key_stats[k]['is_cooling']),
                'success_rate': f"{(self.successful_requests / max(1, self.total_requests) * 100):.1f}%",
                'total_requests': self.total_requests,
                'avg_response_time': f"{avg_response_time:.1f}s",
                'complex_queries_handled': total_complex_queries,
                'primary_model_usage': f"{self.primary_model_used} (Llama-4-Scout)",
                'fallback_model_usage': f"{self.fallback_model_used} (Kimi-K2)",
                'total_capacity': f"Primary: {total_primary_capacity:,} TPM, Fallback: {total_fallback_capacity:,} TPM"
            }


# Initialize with all 6 keys
groq_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

groq_pool = AdaptiveKeyPool(groq_keys, primary_tpm=30000, fallback_tpm=10000, rpm_limit=30)

app = FastAPI(title="Concise Adaptive Document Assistant", version="9.3.0")

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
    """Analyze if a query is complex and might need more thinking time"""

    query_lower = query.lower()

    # Indicators of complex queries that might need more time
    complexity_indicators = {
        'definition_requests': [
            'define', 'definition', 'what does', 'what is', 'meaning of', 'term', 'how does',
            'what qualifies as', 'what constitutes'
        ],
        'multi_part_questions': [
            'conditions', 'requirements', 'what are the', 'list', 'explain',
            'exclusions', 'limitations', 'process for', 'steps', 'how to'
        ],
        'comparative_analysis': [
            'difference between', 'compare', 'versus', 'vs', 'which is better',
            'pros and cons', 'advantages'
        ],
        'comprehensive_coverage': [
            'all', 'comprehensive', 'complete', 'entire', 'full coverage',
            'everything about', 'detailed'
        ],
        'complex_medical_terms': [
            'psychiatric', 'infertility', 'transplant', 'refractive', 'robotic surgery',
            'stem cell', 'domiciliary', 'ayush', 'modern treatment'
        ]
    }

    # Simple queries that usually don't need extra time
    simple_indicators = [
        'is', 'does', 'can', 'will', 'covered', 'period', 'limit', 'amount',
        'yes or no', 'true or false'
    ]

    complexity_score = 0
    complexity_reasons = []

    # Check for complexity indicators
    for category, indicators in complexity_indicators.items():
        for indicator in indicators:
            if indicator in query_lower:
                complexity_score += 1
                complexity_reasons.append(category)
                break

    # Check query length (longer queries often more complex)
    if len(query.split()) > 12:
        complexity_score += 1
        complexity_reasons.append('long_query')

    # Check for multiple questions in one
    question_count = query.count('?') + query.count(' and ') + query.count(' or ')
    if question_count > 1:
        complexity_score += 1
        complexity_reasons.append('multi_question')

    # Reduce score for simple indicators
    for simple in simple_indicators:
        if simple in query_lower and len(query.split()) <= 8:
            complexity_score = max(0, complexity_score - 1)
            break

    is_complex = complexity_score >= 2
    reason = ', '.join(complexity_reasons) if complexity_reasons else 'simple'

    return is_complex, reason


def comprehensive_document_chunking(text: str, chunk_size: int = 1000) -> List[Dict]:
    """Comprehensive chunking designed for maximum information extraction"""
    chunks = []

    # Universal patterns that work for any document type
    critical_patterns = [
        # Time periods and numbers
        r'(?:grace|waiting)\s+period[^.]{0,200}\.',
        r'(?:thirty|30|twenty-four|24|thirty-six|36|two|2)\s+(?:days?|months?|years?)[^.]{0,200}\.',
        r'\d+\s*%[^.]{0,150}\.',
        r'\d+\s*(?:days?|months?|years?)[^.]{0,200}\.',

        # Coverage and benefits patterns
        r'(?:covers?|coverage|benefit|indemnif)[^.]{0,300}\.',
        r'(?:maternity|pregnancy|childbirth|delivery)[^.]{0,300}\.',
        r'(?:hospital|medical|treatment|expenses?)[^.]{0,300}\.',
        r'(?:discount|bonus|ncd)[^.]{0,200}\.',
        r'(?:check-up|examination|preventive)[^.]{0,200}\.',

        # Exclusions and limitations
        r'(?:exclusion|excluded|not\s+covered|limit)[^.]{0,300}\.',
        r'(?:sub-limit|capped|maximum)[^.]{0,200}\.',

        # Definitions and specific terms
        r'(?:defined|definition|means)[^.]{0,400}\.',
        r'(?:ayush|homeopathy|ayurveda)[^.]{0,300}\.',
        r'(?:donor|transplant|organ)[^.]{0,300}\.',
        r'(?:room\s+rent|icu|intensive)[^.]{0,200}\.',
        r'(?:outpatient|opd|day\s+care)[^.]{0,200}\.',
        r'(?:floater|sum\s+insured|aggregate)[^.]{0,200}\.',

        # Common document sections
        r'(?:clause|section|article)\s+\d+[^.]{0,500}\.',
        r'(?:table|schedule|annexure)[^.]{0,500}\.',
    ]

    # Extract critical information with broader context
    critical_chunks = []
    used_positions = set()

    for pattern in critical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            start = max(0, match.start() - 150)  # Larger context window
            end = min(len(text), match.end() + 150)

            # Allow some overlap for comprehensive coverage
            if not any(abs(start - pos) < 200 for pos in used_positions):
                context = text[start:end].strip()

                if len(context) > 80:
                    critical_chunks.append({
                        'text': context,
                        'metadata': {
                            'type': 'critical',
                            'priority': 100,
                            'has_numbers': bool(re.search(r'\d+', context)),
                            'has_percentages': bool(re.search(r'\d+\s*%', context)),
                            'coverage_score': len(context),
                            'pattern_type': 'critical'
                        }
                    })
                    used_positions.add(start)

    # Enhanced sectional chunking for comprehensive coverage
    section_delimiters = [
        r'\n(?=(?:CLAUSE|SECTION|ARTICLE|CHAPTER)\s+\d+)',
        r'\n(?=(?:DEFINITIONS?|BENEFITS?|EXCLUSIONS?|CONDITIONS?))',
        r'\n(?=(?:Table|Schedule|Annexure))',
        r'\n(?=[A-Z][A-Z\s]{10,}:)',  # ALL CAPS headers
        r'\n(?=\d+\.\d+\s)',  # Numbered sections
        r'\n(?=\w+\s+means\s)',  # Definition patterns
    ]

    # Split document into logical sections
    sections = [text]
    for delimiter in section_delimiters:
        new_sections = []
        for section in sections:
            parts = re.split(delimiter, section)
            new_sections.extend([s.strip() for s in parts if s.strip()])
        sections = new_sections

    # Process each section with overlap
    regular_chunks = []
    for i, section in enumerate(sections):
        if len(section) < 100:
            continue

        # Determine section priority
        section_lower = section.lower()
        priority = 20  # Base priority

        # Higher priority for important sections
        important_keywords = [
            'definition', 'benefit', 'coverage', 'waiting period', 'grace period',
            'maternity', 'discount', 'hospital', 'exclusion', 'limit'
        ]

        for keyword in important_keywords:
            if keyword in section_lower:
                priority += 15
                break

        # Split long sections with overlap
        if len(section) <= chunk_size:
            regular_chunks.append({
                'text': section,
                'metadata': {
                    'type': 'section',
                    'priority': priority,
                    'has_numbers': bool(re.search(r'\d+', section)),
                    'has_percentages': bool(re.search(r'\d+\s*%', section)),
                    'coverage_score': len(section),
                    'section_id': i
                }
            })
        else:
            # Sliding window with overlap
            start = 0
            overlap = 200

            while start < len(section):
                end = min(start + chunk_size, len(section))
                chunk_text = section[start:end]

                if len(chunk_text.strip()) > 100:
                    regular_chunks.append({
                        'text': chunk_text.strip(),
                        'metadata': {
                            'type': 'section',
                            'priority': priority,
                            'has_numbers': bool(re.search(r'\d+', chunk_text)),
                            'has_percentages': bool(re.search(r'\d+\s*%', chunk_text)),
                            'coverage_score': len(chunk_text),
                            'section_id': i,
                            'chunk_part': start // chunk_size
                        }
                    })

                if end >= len(section):
                    break
                start += chunk_size - overlap

    # Combine all chunks
    all_chunks = critical_chunks + regular_chunks

    # Advanced deduplication using content similarity
    unique_chunks = []
    seen_content = []

    for chunk in all_chunks:
        # Create content fingerprint
        words = set(chunk['text'].lower().split())
        content_words = [w for w in words if len(w) > 3]

        # Check similarity with existing chunks
        is_duplicate = False
        for existing_words in seen_content:
            if len(content_words) > 0 and len(existing_words) > 0:
                similarity = len(set(content_words) & set(existing_words)) / len(
                    set(content_words) | set(existing_words))
                if similarity > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break

        if not is_duplicate and len(chunk['text']) > 50:
            unique_chunks.append(chunk)
            seen_content.append(content_words)

            # Limit total chunks to prevent overwhelming
            if len(unique_chunks) >= 150:
                break

    # Sort by priority and coverage score for best results
    unique_chunks.sort(key=lambda x: (
        x['metadata']['priority'],
        x['metadata']['coverage_score'],
        x['metadata'].get('has_numbers', False),
        x['metadata'].get('has_percentages', False)
    ), reverse=True)

    logger.info(
        f"[Jay121305] Comprehensive chunking: {len(critical_chunks)} critical + {len(regular_chunks)} regular = {len(unique_chunks)} unique")
    return unique_chunks


def multi_strategy_search(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """Multi-strategy search for maximum accuracy across any document type"""

    query_lower = query.lower()
    query_words = [w.lower() for w in query.split() if len(w) > 2]

    # Extract key terms and patterns from query
    key_terms = set()

    # Common document terms that might appear in any policy/document
    universal_mappings = {
        'grace': ['grace', 'period', 'premium', 'payment', 'due', 'renew'],
        'waiting': ['waiting', 'period', 'months', 'coverage', 'continuous'],
        'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'female', 'months'],
        'cataract': ['cataract', 'surgery', 'waiting', 'period', 'months', 'years'],
        'organ': ['organ', 'donor', 'transplant', 'harvesting', 'expenses'],
        'discount': ['discount', 'ncd', 'claim', 'premium', 'bonus', 'renewal'],
        'health': ['health', 'check', 'preventive', 'examination', 'medical'],
        'hospital': ['hospital', 'definition', 'institution', 'beds', 'nursing', 'qualified'],
        'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'naturopathy'],
        'room': ['room', 'rent', 'icu', 'intensive', 'charges', 'limit', 'percentage'],
        'day': ['day', 'care', 'procedure', 'treatment', 'surgical', 'hours'],
        'domiciliary': ['domiciliary', 'home', 'treatment', 'hospitalization'],
        'aids': ['aids', 'hiv', 'immune', 'deficiency', 'treatment'],
        'floater': ['floater', 'sum', 'insured', 'aggregate', 'family']
    }

    # Identify relevant term groups
    for term, related_terms in universal_mappings.items():
        if term in query_lower:
            key_terms.update(related_terms)

    # Add query words
    key_terms.update(query_words)

    # Advanced scoring algorithm
    scored_chunks = []

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        metadata = chunk['metadata']
        score = 0

        # Exact query phrase matching (highest weight)
        if query_lower in text_lower:
            score += 100

        # Key term matching with weights
        term_matches = 0
        for term in key_terms:
            if term in text_lower:
                # Weight by term length and frequency
                term_weight = len(term) + text_lower.count(term) * 5
                score += min(term_weight, 25)  # Cap per term
                term_matches += 1

        # Term density bonus
        if term_matches > 0:
            density_bonus = min(term_matches * 8, 40)
            score += density_bonus

        # Priority and type bonuses
        priority = metadata.get('priority', 0)
        score += priority

        if metadata.get('type') == 'critical':
            score += 60

        # Numeric content bonus (important for policy documents)
        if metadata.get('has_numbers'):
            score += 25

        if metadata.get('has_percentages'):
            score += 30

        # Coverage score (longer chunks often more complete)
        coverage_score = metadata.get('coverage_score', 0)
        score += min(coverage_score // 50, 20)

        # Pattern-specific bonuses
        important_patterns = [
            'grace period', 'waiting period', 'maternity', 'discount',
            'health check', 'hospital', 'room rent', 'coverage',
            'benefit', 'exclusion', 'definition', 'limit'
        ]

        for pattern in important_patterns:
            if pattern in text_lower:
                score += 20

        # Number and percentage pattern matching
        number_patterns = [r'\d+\s*(?:days?|months?|years?)', r'\d+\s*%', r'two\s+(?:years?|months?)', r'thirty']
        for pattern in number_patterns:
            if re.search(pattern, text_lower):
                score += 15

        # Penalize very short chunks
        if len(chunk['text']) < 80:
            score -= 15

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    if scored_chunks:
        logger.info(
            f"[Jay121305] Multi-strategy search: top score = {scored_chunks[0][1]}, found {len(scored_chunks)} relevant chunks")
        # Log top 3 for debugging
        for i, (chunk, score) in enumerate(scored_chunks[:3]):
            preview = chunk['text'][:100].replace('\n', ' ')
            logger.debug(f"[Jay121305] Rank {i + 1}: score={score}, text='{preview}...'")

    return [chunk for chunk, _ in scored_chunks[:top_k]]


class AdaptiveRetriever:
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
        self.chunks = comprehensive_document_chunking(text)
        self.is_indexed = True
        self.doc_hash = doc_hash
        logger.info(f"[Jay121305] Indexed {len(self.chunks)} chunks in {time.time() - start:.2f}s")

    async def get_adaptive_context(self, query: str, is_complex: bool = False) -> str:
        """Get context adapted to query complexity"""
        if not self.is_indexed:
            return "Document not indexed"

        # Adjust chunk count based on complexity
        chunk_count = 6 if is_complex else 5
        relevant_chunks = multi_strategy_search(query, self.chunks, top_k=chunk_count)

        if not relevant_chunks:
            # Fallback to highest priority chunks
            high_priority = sorted(
                [c for c in self.chunks if c['metadata'].get('priority', 0) > 50],
                key=lambda x: x['metadata']['priority'],
                reverse=True
            )
            relevant_chunks = high_priority[:chunk_count] if high_priority else self.chunks[:chunk_count]

        # Build adaptive context with complexity-based limits
        context_parts = []
        total_chars = 0
        max_context = 6000 if is_complex else 5000  # More context for complex queries

        for i, chunk in enumerate(relevant_chunks):
            text = chunk['text']
            priority = chunk['metadata'].get('priority', 0)

            # Adaptive text inclusion based on complexity and priority
            if is_complex:
                # More generous limits for complex queries
                if priority > 80:
                    chunk_text = text[:1000] if len(text) > 1000 else text
                    label = f"[CRITICAL-{i + 1}]"
                elif priority > 50:
                    chunk_text = text[:700] if len(text) > 700 else text
                    label = f"[IMPORTANT-{i + 1}]"
                else:
                    chunk_text = text[:500] if len(text) > 500 else text
                    label = f"[{i + 1}]"
            else:
                # Standard limits for simple queries
                if priority > 80:
                    chunk_text = text[:800] if len(text) > 800 else text
                    label = f"[CRITICAL-{i + 1}]"
                elif priority > 50:
                    chunk_text = text[:600] if len(text) > 600 else text
                    label = f"[IMPORTANT-{i + 1}]"
                else:
                    chunk_text = text[:400] if len(text) > 400 else text
                    label = f"[{i + 1}]"

            # Check if adding this chunk exceeds limit
            if total_chars + len(chunk_text) > max_context:
                # Truncate to fit but ensure minimum useful content
                remaining = max_context - total_chars
                if remaining > 150:  # Only add if meaningful amount left
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(f"{label} {chunk_text}")
                break

            context_parts.append(f"{label} {chunk_text}")
            total_chars += len(chunk_text)

        context = "\n\n".join(context_parts)
        complexity_note = " (complex query)" if is_complex else ""
        logger.info(
            f"[Jay121305] Adaptive context: {len(context)} chars from {len(relevant_chunks)} chunks{complexity_note}")
        return context


# Global retriever
retriever = AdaptiveRetriever()

# Concise adaptive system prompts
ADAPTIVE_PRIMARY_PROMPT = """You are an expert document analyst specializing in policy documents. Provide accurate, concise answers based strictly on the document context.

RESPONSE FORMAT (JSON):
{
  "answer": "Clear, concise answer with essential facts, numbers, and conditions",
  "confidence": "high | medium | low",
  "needs_more_time": "true | false"
}

FORMATTING & ACCURACY RULES:
1. Keep answers concise (60-120 words) - avoid excessive detail
2. Include ALL exact numbers, percentages, time periods as stated
3. For definitions, provide focused definitions without extensive elaboration
4. If information exists in context, provide it - never say "not mentioned" without thorough analysis
5. Use clear, professional language - DO NOT use \\n or line break characters
6. Include key conditions using phrases like "provided that", "subject to", "limited to"
7. Set confidence HIGH if answer is definitive, MEDIUM if partially found, LOW if uncertain
8. Set needs_more_time to TRUE only if question is very complex and you need more analysis time
9. Format numbers clearly (e.g., "thirty (30) days", "24 months", "5%")
10. Write in flowing prose without literal newlines or formatting characters
11. Avoid bullet points, excessive sub-sections, or overly detailed explanations

CONCISENESS PRIORITY: Provide complete but succinct information. Avoid verbose explanations unless absolutely necessary.

COMPLEXITY ASSESSMENT:
- Set needs_more_time=true for: complex definitions, multi-part questions, comprehensive coverage requests
- Set needs_more_time=false for: simple yes/no, basic coverage, straightforward facts

BALANCE: Provide complete information while maintaining readability and professional conciseness."""

ADAPTIVE_FALLBACK_PROMPT = """You are a senior document expert handling a complex query that needs thorough analysis. Provide comprehensive but concise answers.

RESPONSE FORMAT (JSON):
{
  "answer": "Comprehensive answer with complete information, properly formatted and clearly structured"
}

EXPERT ANALYSIS FOR COMPLEX QUERIES:
1. Conduct thorough analysis of ALL context sections
2. Include ALL numerical details, percentages, time periods, conditions
3. Provide complete eligibility criteria and exceptions
4. Cross-reference multiple sections for comprehensive coverage
5. Use clear, professional language with logical structure - NO \\n characters
6. Include all qualifying conditions and procedural requirements
7. Format for maximum clarity while being thorough but concise
8. Ensure no important detail is omitted
9. Provide complete benefit structures and limitations
10. Write in flowing, professional prose without special formatting characters
11. Avoid excessive elaboration - focus on essential information

CONCISENESS: Even for complex queries, maintain professional brevity while ensuring completeness.

THOROUGHNESS: This is a complex query requiring detailed analysis - be comprehensive while maintaining clarity and avoiding literal newline characters or excessive verbosity."""


async def analyze_with_adaptive_timing(query: str, context: str, is_complex: bool = False,
                                       complexity_reason: str = "") -> Dict[str, Any]:
    """Adaptive analysis that adjusts timing based on query complexity"""
    try:
        # Adaptive context limits
        context_limit = 6000 if is_complex else 5000
        if len(context) > context_limit:
            context = context[:context_limit] + "..."

        # Adaptive token estimation
        base_tokens = (len(context) + len(query) + 600) // 4
        estimated_tokens = int(base_tokens * 1.2) if is_complex else base_tokens

        # Step 1: Primary model with adaptive settings
        api_key = groq_pool.get_adaptive_key(estimated_tokens, use_fallback=False, is_complex=is_complex)

        # Adaptive timeout and max_tokens - reduced for conciseness
        timeout = 40 if is_complex else 30
        max_tokens = 700 if is_complex else 600  # Reduced for more concise answers

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": ADAPTIVE_PRIMARY_PROMPT},
                {"role": "user",
                 "content": f"QUESTION: {query}\n\nDOCUMENT CONTEXT:\n{context}\n\nProvide concise, accurate answer (complexity: {complexity_reason}):"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.02,
            "max_tokens": max_tokens
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
                    timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:

                if response.status == 429:
                    groq_pool.record_failure(api_key, "rate_limit")
                    raise Exception("Primary model rate limited")
                elif response.status >= 400:
                    groq_pool.record_failure(api_key, f"http_{response.status}")
                    raise Exception(f"Primary model HTTP {response.status}")

                result = await response.json()

        tokens_used = result.get("usage", {}).get("total_tokens", estimated_tokens)
        response_time = time.time() - start_time
        groq_pool.record_success(api_key, tokens_used, response_time, use_fallback=False, was_complex=is_complex)

        # Parse primary response with cleaning
        content = result["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            raw_answer = parsed.get("answer", "Unable to determine from provided context")
            answer = clean_answer_formatting(raw_answer)  # Apply cleaning
            confidence = parsed.get("confidence", "medium").lower()
            needs_more_time = parsed.get("needs_more_time", "false").lower() == "true"
        except json.JSONDecodeError:
            raw_content = content
            answer = clean_answer_formatting(raw_content)  # Apply cleaning
            confidence = "low"
            needs_more_time = is_complex

        elapsed = time.time() - start_time

        # Step 2: Adaptive fallback logic
        should_use_fallback = (
                confidence == "low" or
                len(answer) < 40 or
                "not mentioned" in answer.lower() or
                needs_more_time or
                (is_complex and confidence == "medium")
        )

        if should_use_fallback:
            complexity_note = f" (complex: {complexity_reason})" if is_complex else ""
            logger.info(f"[Jay121305] Primary {confidence} confidence{complexity_note}, using adaptive fallback...")

            try:
                fallback_key = groq_pool.get_adaptive_key(estimated_tokens, use_fallback=True, is_complex=is_complex)

                fallback_timeout = 45 if is_complex else 35
                fallback_tokens = 900 if is_complex else 800  # Reduced for conciseness

                fallback_payload = {
                    "model": "moonshotai/kimi-k2-instruct",
                    "messages": [
                        {"role": "system", "content": ADAPTIVE_FALLBACK_PROMPT},
                        {"role": "user",
                         "content": f"QUESTION: {query}\n\nDOCUMENT CONTEXT:\n{context}\n\nConcise analysis needed ({complexity_reason}). Primary had {confidence} confidence:"}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.02,
                    "max_tokens": fallback_tokens
                }

                fallback_headers = {
                    "Authorization": f"Bearer {fallback_key}",
                    "Content-Type": "application/json"
                }

                fallback_start = time.time()

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            json=fallback_payload,
                            headers=fallback_headers,
                            timeout=aiohttp.ClientTimeout(total=fallback_timeout)
                    ) as fallback_response:

                        if fallback_response.status >= 400:
                            groq_pool.record_failure(fallback_key, f"http_{fallback_response.status}")
                            logger.warning(f"[Jay121305] Adaptive fallback failed, using primary answer")
                        else:
                            fallback_result = await fallback_response.json()
                            fallback_tokens_used = fallback_result.get("usage", {}).get("total_tokens",
                                                                                        estimated_tokens)
                            fallback_response_time = time.time() - fallback_start
                            groq_pool.record_success(fallback_key, fallback_tokens_used, fallback_response_time,
                                                     use_fallback=True, was_complex=is_complex)

                            fallback_content = fallback_result["choices"][0]["message"]["content"]
                            try:
                                fallback_parsed = json.loads(fallback_content)
                                raw_fallback_answer = fallback_parsed.get("answer", answer)
                                fallback_answer = clean_answer_formatting(raw_fallback_answer)  # Apply cleaning

                                # Use fallback if significantly better or primary had issues
                                if (len(fallback_answer) > len(answer) + 30 or
                                        "not mentioned" in answer.lower() or
                                        len(answer) < 40 or
                                        needs_more_time):
                                    answer = fallback_answer
                                    confidence = "medium"
                                    logger.info(f"[Jay121305] Using adaptive fallback answer")

                            except json.JSONDecodeError:
                                logger.warning(f"[Jay121305] Adaptive fallback JSON parse failed")

            except Exception as e:
                logger.warning(f"[Jay121305] Adaptive fallback failed: {e}")

        model_note = ""
        if is_complex:
            model_note = f" (adaptive-complex)"
        elif confidence == "medium":
            model_note = f" (with-fallback)"

        return {
            "answer": answer,
            "success": True,
            "tokens_used": tokens_used,
            "time_seconds": round(elapsed, 2),
            "api_key": api_key[-5:],
            "confidence": confidence,
            "complexity": complexity_reason if is_complex else "simple",
            "model_used": f"Llama-4-Scout{model_note}" if confidence != "medium" else f"Llama-4-Scout + Adaptive-Kimi-K2{model_note}"
        }

    except Exception as e:
        logger.error(f"[Jay121305] Adaptive analysis failed: {str(e)}")
        return {
            "answer": "Analysis failed",
            "success": False,
            "error": str(e)
        }


def extract_pdf_text(pdf_url: str) -> str:
    try:
        logger.info("[Jay121305] Downloading PDF...")
        response = requests.get(pdf_url, timeout=90)
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
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")


@app.get("/")
def root():
    return {
        "message": "Concise Adaptive Document Assistant v9.3",
        "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)",
        "fallback_model": "moonshotai/kimi-k2-instruct (10K TPM)",
        "design": "Adaptive timing with concise, clean responses",
        "features": [
            "Smart complexity detection",
            "Adaptive response timing",
            "Concise answer formatting",
            "Automatic \\n cleaning",
            "Performance optimization"
        ],
        "timing_strategy": "Only increases time when genuinely needed",
        "answer_style": "Professional, concise, clean formatting",
        "keys_available": len(groq_keys),
        "user": "Jay121305",
        "timestamp": "2025-07-31 20:27:03"
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
        num_questions = len(request.questions)
        logger.info(
            f"[Jay121305] CONCISE ADAPTIVE: Processing {num_questions} questions with smart complexity + clean formatting")

        pdf_text = extract_pdf_text(request.documents)
        await retriever.index_document(pdf_text)

        async def process_single_question(i, question):
            question_start = time.time()

            # Analyze query complexity
            is_complex, complexity_reason = analyze_query_complexity(question)

            complexity_indicator = "🔍" if is_complex else "⚡"
            logger.info(f"[Jay121305] Q{i + 1} {complexity_indicator}: {question[:60]}... [{complexity_reason}]")

            # Adaptive delays - only longer for complex queries
            if is_complex:
                delay = (i % 6) * 0.4 + random.uniform(0.2, 0.5)  # 0.2-4.0s for complex
            else:
                delay = (i % 6) * 0.2 + random.uniform(0.05, 0.2)  # 0.05-1.4s for simple

            await asyncio.sleep(delay)

            context = await retriever.get_adaptive_context(question, is_complex)
            result = await analyze_with_adaptive_timing(question, context, is_complex, complexity_reason)

            elapsed = time.time() - question_start
            if result.get("success"):
                tokens = result.get('tokens_used', 0)
                model_used = result.get('model_used', 'Unknown')
                confidence = result.get('confidence', 'unknown')
                complexity = result.get('complexity', 'simple')

                timing_emoji = "🕐" if elapsed > 2.0 else "⚡"
                logger.info(
                    f"[Jay121305] Q{i + 1} ✓ {timing_emoji} | {elapsed:.1f}s | {tokens}t | {model_used} | {confidence} | {complexity}")
            else:
                logger.error(f"[Jay121305] Q{i + 1} ✗ | {elapsed:.1f}s | {result.get('error', 'unknown')}")

            return result.get("answer", "Analysis failed")

        # Adaptive concurrency based on complexity distribution
        complex_count = sum(1 for q in request.questions if analyze_query_complexity(q)[0])
        simple_count = num_questions - complex_count

        if complex_count > num_questions * 0.6:  # Mostly complex queries
            concurrency = 2
        elif num_questions <= 30:
            concurrency = 3
        else:
            concurrency = 2

        semaphore = asyncio.Semaphore(concurrency)
        logger.info(
            f"[Jay121305] Adaptive concurrency: {concurrency} | Complex: {complex_count}, Simple: {simple_count}")

        async def limited_process(i, question):
            async with semaphore:
                return await process_single_question(i, question)

        # Execute all questions
        tasks = [limited_process(i, q) for i, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results with additional cleaning
        final_answers = []
        for answer in answers:
            if isinstance(answer, Exception):
                final_answers.append("Processing failed")
                logger.error(f"[Jay121305] Exception: {answer}")
            else:
                # Apply final cleaning to all answers
                cleaned_answer = clean_answer_formatting(str(answer))
                final_answers.append(cleaned_answer)

        total_time = time.time() - start_time
        success_count = sum(1 for a in final_answers if a not in ["Processing failed", "Analysis failed"])

        pool_status = groq_pool.get_pool_status()

        logger.info(f"[Jay121305] CONCISE ADAPTIVE COMPLETED")
        logger.info(f"[Jay121305] ⏱️  Total time: {total_time:.1f}s ({total_time / num_questions:.2f}s per question)")
        logger.info(
            f"[Jay121305] ✅ Success: {success_count}/{num_questions} ({(success_count / num_questions) * 100:.1f}%)")
        logger.info(f"[Jay121305] 🎯 Adaptive timing + concise formatting + \\n cleaning applied")
        logger.info(f"[Jay121305] 📊 Complexity distribution: {complex_count} complex, {simple_count} simple")
        logger.info(
            f"[Jay121305] 🔧 Model usage: {pool_status['primary_model_usage']}, {pool_status['fallback_model_usage']}")
        logger.info(
            f"[Jay121305] 🔑 Keys: {pool_status['active_keys']}/{pool_status['total_keys']} active | Avg response: {pool_status['avg_response_time']}")

        return {"answers": final_answers}

    except Exception as e:
        logger.error(f"[Jay121305] Concise adaptive endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Concise adaptive processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("[Jay121305] Starting Concise Adaptive Document Assistant v9.3")
    logger.info(f"[Jay121305] Smart timing: Only increases time when complexity requires it")
    logger.info(f"[Jay121305] Concise responses: 60-120 words with clean formatting")
    logger.info(f"[Jay121305] Auto-cleaning: Removes \\n characters and excessive formatting")
    logger.info(f"[Jay121305] Performance: Maintains ~1s for simple queries, extends for complex ones")
    logger.info(f"[Jay121305] Primary: meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)")
    logger.info(f"[Jay121305] Concise Fallback: moonshotai/kimi-k2-instruct (10K TPM)")
    logger.info(f"[Jay121305] User: Jay121305 | Time: 2025-07-31 20:27:03")
    uvicorn.run(app, host="0.0.0.0", port=8000)