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
import yake
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Optional
import re  # Make sure this is included


def metadata_weight_score(chunk: Dict) -> float:
    meta = chunk.get("metadata", {})
    priority = meta.get("priority", 0)
    coverage = meta.get("coverage_score", 0)
    has_numbers = 1 if meta.get("has_numbers") else 0
    has_percent = 1 if meta.get("has_percentages") else 0
    clause_like = 1 if re.search(r'\b(clause|section|article)\b', chunk["text"].lower()) else 0

    # Combined score (normalize coverage)
    return (
        0.4 * priority +
        0.3 * (coverage / 1000) +  # normalize large values
        0.2 * has_numbers +
        0.1 * has_percent +
        0.3 * clause_like
    )


def extract_dynamic_keywords(text: str, max_keywords: int = 20) -> list[str]:
    """
    Extracts top keywords from the given text using YAKE.
    Returns a list of strings.
    """
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=1,  # unigrams only (set n=2 for phrases like 'room rent')
        dedupLim=0.9,
        top=max_keywords,
        features=None
    )
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, _ in keywords]


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
        """Robust adaptive key selection with better cooldown handling and load balancing"""
        with self.lock:
            tpm_limit = self.fallback_tpm if use_fallback else self.primary_tpm
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            # Adjust estimated token budget for complex queries
            if is_complex:
                estimated_tokens = int(estimated_tokens * 1.3)

            usage_limit = 0.85 if not is_complex else 0.75  # Conservative cap for complex queries

            # 🔁 Ensure cooldown and usage counters are reset for ALL keys
            for key in self.keys:
                self.reset_counters_if_needed(key)

            # ⚖️ Build list of all usable keys with load score
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

            # ✅ If usable keys exist, choose the least loaded one
            if usable_keys:
                usable_keys.sort(key=lambda x: x[1])  # lowest load_score first
                selected_key = usable_keys[0][0]
                model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                logger.debug(f"[Jay121305] Selected key ...{selected_key[-5:]} for {model_name}")
                return selected_key

            # 🔄 All keys are overloaded or cooling — determine soonest available
            soonest_key = min(self.keys, key=lambda k: self.key_stats[k]['last_reset'] + 60)
            wait_time = max(0, self.key_stats[soonest_key]['last_reset'] + 60 - time.time())
            max_wait = 15 if is_complex else 10

            model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
            logger.warning(f"[Jay121305] All keys at {model_name} limit, waiting {wait_time:.1f}s...")

            if wait_time > 0:
                time.sleep(min(wait_time, max_wait))
                return self.get_adaptive_key(estimated_tokens, use_fallback, is_complex)

            return soonest_key  # fallback to soonest-reset key

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


def smart_section_chunking(text: str) -> List[Dict]:
    """Simple but effective chunking that actually works"""
    chunks = []

    # Step 1: Try to find obvious section breaks first
    obvious_breaks = []

    # Look for numbered sections, exclusions, benefits, etc.
    section_markers = [
        r'\n\s*(?:EXCLUSION|BENEFIT|CLAUSE|SECTION)\s*\d+',
        r'\n\s*\d+\.\d+\s+',
        r'\n\s*\d+\.\s+[A-Z]',
        r'\n\s*[A-Z]{4,}\s*:',
        r'\n\s*Table\s+\d+',
        r'\n\s*Schedule\s+[A-Z]'
    ]

    for pattern in section_markers:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            obvious_breaks.append(match.start())

    # Add start and end positions
    obvious_breaks = [0] + sorted(set(obvious_breaks)) + [len(text)]

    # Step 2: Create chunks from obvious breaks
    for i in range(len(obvious_breaks) - 1):
        start = obvious_breaks[i]
        end = obvious_breaks[i + 1]
        section_text = text[start:end].strip()

        if len(section_text) > 100:  # Only meaningful chunks
            chunks.append(create_chunk_simple(section_text, i))

    # Step 3: If we don't have enough chunks, split by paragraphs
    if len(chunks) < 20:
        logger.warning(f"[SMART_CHUNKING] Only {len(chunks)} chunks from sections, adding paragraph chunks")

        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        paragraph_chunks = []

        current_chunk = []
        current_length = 0
        max_chunk_size = 800

        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip tiny paragraphs
                continue

            # If adding this paragraph would make chunk too big
            if current_length + len(para) > max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                paragraph_chunks.append(create_chunk_simple(chunk_text, len(paragraph_chunks)))
                current_chunk = [para]
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            paragraph_chunks.append(create_chunk_simple(chunk_text, len(paragraph_chunks)))

        # Add paragraph chunks to our collection
        chunks.extend(paragraph_chunks)
        logger.info(f"[SMART_CHUNKING] Added {len(paragraph_chunks)} paragraph chunks")

    # Step 4: If still not enough, split by single newlines
    if len(chunks) < 30:
        logger.warning(f"[SMART_CHUNKING] Only {len(chunks)} chunks total, adding line-based chunks")

        # Split by single newlines and group
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        line_chunks = []

        current_chunk = []
        current_length = 0
        max_chunk_size = 600

        for line in lines:
            if current_length + len(line) > max_chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text.strip()) > 100:  # Only meaningful chunks
                    line_chunks.append(create_chunk_simple(chunk_text, len(line_chunks)))
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line)

        # Last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text.strip()) > 100:
                line_chunks.append(create_chunk_simple(chunk_text, len(line_chunks)))

        chunks.extend(line_chunks)
        logger.info(f"[SMART_CHUNKING] Added {len(line_chunks)} line-based chunks")

    # Step 5: Remove duplicates and sort by priority
    unique_chunks = []
    seen_starts = set()

    for chunk in chunks:
        # Simple dedup based on first 50 characters
        chunk_start = chunk['text'][:50].lower()
        if chunk_start not in seen_starts:
            unique_chunks.append(chunk)
            seen_starts.add(chunk_start)

    # Sort by priority
    unique_chunks.sort(key=lambda x: x['metadata']['priority'], reverse=True)

    # Limit to reasonable number
    final_chunks = unique_chunks[:100]

    # Log results
    chunk_types = {}
    for chunk in final_chunks:
        chunk_type = chunk['metadata']['type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

    logger.info(f"[SMART_CHUNKING] Final result: {len(final_chunks)} chunks")
    logger.info(f"[SMART_CHUNKING] Types: {dict(chunk_types)}")

    return final_chunks


def create_chunk_simple(text: str, section_id: int) -> Dict:
    """Create a chunk with smart priority based on content - works for ANY document type"""
    text_lower = text.lower()

    # Step 1: Extract dynamic keywords from the chunk itself
    chunk_keywords = extract_dynamic_keywords(text, max_keywords=10)

    # Step 2: Determine document type and set priorities dynamically
    priority = 50  # Base priority
    chunk_type = 'general'

    # Document type detection patterns
    doc_type_indicators = {
        'legal': ['article', 'section', 'clause', 'constitution', 'law', 'act', 'regulation', 'legal', 'court',
                  'justice'],
        'medical': ['treatment', 'surgery', 'hospital', 'medical', 'patient', 'doctor', 'disease', 'medicine'],
        'insurance': ['coverage', 'claim', 'policy', 'premium', 'benefit', 'exclusion', 'deductible', 'insured'],
        'physics': ['force', 'motion', 'gravity', 'mass', 'acceleration', 'velocity', 'newton', 'physics'],
        'technical': ['engine', 'brake', 'tire', 'manual', 'specification', 'maintenance', 'repair', 'system'],
        'programming': ['code', 'function', 'variable', 'javascript', 'python', 'algorithm', 'programming'],
        'financial': ['payment', 'cost', 'price', 'amount', 'money', 'fee', 'charge', 'budget']
    }

    # Detect document type
    detected_types = []
    for doc_type, indicators in doc_type_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            detected_types.append(doc_type)

    # Step 3: Universal high-priority patterns (work across all document types)
    universal_high_priority = [
        # Structural elements
        r'\b(?:article|section|clause|chapter|part)\s+\d+',
        r'\b(?:definition|means|defined as|refers to)\b',
        r'\b(?:exclusion|exception|limitation|restriction)\b',
        r'\b(?:requirement|condition|criteria|rule)\b',

        # Quantitative information
        r'\d+\s*(?:days?|months?|years?|hours?)',
        r'\d+\s*%|\d+\s*percent',
        r'(?:maximum|minimum|limit|cap|up to)\s+\d+',
        r'(?:not\s+(?:more|less)\s+than|at\s+least)\s+\d+',

        # Important actions/processes
        r'\b(?:shall|must|required|mandatory|obligatory)\b',
        r'\b(?:prohibited|forbidden|not\s+allowed|illegal)\b',
        r'\b(?:applicable|valid|effective|in\s+force)\b'
    ]

    # Check for universal high-priority patterns
    high_priority_matches = 0
    for pattern in universal_high_priority:
        if re.search(pattern, text_lower):
            high_priority_matches += 1

    # Step 4: Set priority based on patterns and content
    if high_priority_matches >= 2:
        priority = 95
        chunk_type = 'critical'
    elif high_priority_matches >= 1:
        priority = 85
        chunk_type = 'important'
    elif len(detected_types) >= 2:  # Multi-domain content
        priority = 80
        chunk_type = 'multi_domain'
    elif detected_types:
        priority = 75
        chunk_type = detected_types[0]  # Use primary detected type

    # Step 5: Boost priority for specific content types
    content_boosters = {
        'numbers': (re.search(r'\d+', text), 10),
        'percentages': (re.search(r'\d+\s*%', text), 5),
        'definitions': (re.search(r'\b(?:means|defined|definition)\b', text_lower), 15),
        'negatives': (re.search(r'\b(?:not|no|never|except|exclude)\b', text_lower), 8),
        'procedures': (re.search(r'\b(?:procedure|process|method|steps)\b', text_lower), 6),
        'temporal': (re.search(r'\b(?:period|time|duration|deadline)\b', text_lower), 7)
    }

    for booster_name, (pattern_match, boost_value) in content_boosters.items():
        if pattern_match:
            priority += boost_value

    # Step 6: Dynamic keyword-based boosting
    important_keywords_in_chunk = 0
    for keyword in chunk_keywords:
        if len(keyword) > 4:  # Only consider substantial keywords
            important_keywords_in_chunk += 1

    if important_keywords_in_chunk >= 5:
        priority += 10
    elif important_keywords_in_chunk >= 3:
        priority += 5

    # Step 7: Length-based adjustment
    if len(text) > 800:  # Substantial content
        priority += 5
    elif len(text) < 100:  # Too short, might be incomplete
        priority -= 10

    return {
        'text': text,
        'metadata': {
            'type': chunk_type,
            'priority': min(priority, 100),  # Cap at 100
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_percentages': bool(re.search(r'\d+\s*%', text)),
            'coverage_score': len(text),
            'section_id': section_id,
            'is_complete_section': True,
            'detected_types': detected_types,
            'dynamic_keywords': chunk_keywords[:5],  # Store top 5 keywords
            'high_priority_patterns': high_priority_matches
        }
    }
def detect_document_domains(chunks, query_words):
    """Detect domains by analyzing what terms co-occur in document chunks"""
    domain_signals = {}

    for chunk in chunks[:30]:  # Sample chunks to learn domain signals
        text_lower = chunk['text'].lower()
        chunk_words = set(re.findall(r'\b\w{3,}\b', text_lower))

        # For each query word, see what other terms appear in same chunks
        for query_word in query_words:
            if query_word in text_lower:
                if query_word not in domain_signals:
                    domain_signals[query_word] = set()

                # Find positions of query word
                query_positions = [m.start() for m in re.finditer(re.escape(query_word), text_lower)]

                for pos in query_positions:
                    # Extract context around query word (50 chars each side)
                    start = max(0, pos - 50)
                    end = min(len(text_lower), pos + len(query_word) + 50)
                    context = text_lower[start:end]

                    # Extract meaningful terms from context (not all chunk words)
                    context_terms = set(re.findall(r'\b\w{4,}\b', context))  # Only words 4+ chars

                    # Filter out common words
                    meaningful_terms = {term for term in context_terms
                                        if term not in {'that', 'with', 'from', 'this', 'they', 'have', 'been', 'will',
                                                        'were', 'would', 'could', 'should'}}

                    domain_signals[query_word].update(meaningful_terms)

    return domain_signals


def extract_semantic_relationships(chunks, query):
    """Learn semantic relationships from document context"""
    relationships = set()
    query_lower = query.lower()

    for chunk in chunks[:25]:
        text_lower = chunk['text'].lower()

        # If chunk contains query terms, extract related semantic terms
        if any(word in text_lower for word in query.split()):
            # Extract temporal terms that appear with query
            temporal_terms = re.findall(
                r'\b(?:\d+\s*(?:day|month|year|hour|week)s?|period|time|duration|deadline|when)\b', text_lower)
            relationships.update(temporal_terms)

            # Extract quantitative terms
            quantitative_terms = re.findall(
                r'\b(?:amount|number|limit|maximum|minimum|rate|percentage|\d+|up\s+to|at\s+least)\b', text_lower)
            relationships.update(quantitative_terms)

            # Extract conditional terms
            conditional_terms = re.findall(
                r'\b(?:if|when|provided|subject\s+to|condition|requirement|criteria|must|shall)\b', text_lower)
            relationships.update(conditional_terms)

            # Extract structural terms
            structural_terms = re.findall(r'\b(?:section|clause|article|part|chapter|exclusion|benefit|coverage)\b',
                                          text_lower)
            relationships.update(structural_terms)

    return relationships


def extract_document_patterns(chunks):
    """Extract important patterns from document structure"""
    patterns = set()

    # Get patterns from high-priority chunks
    high_priority_chunks = [c for c in chunks if c['metadata'].get('priority', 0) > 70]

    for chunk in high_priority_chunks[:20]:
        text_lower = chunk['text'].lower()

        # Extract section/structural patterns that actually exist in document
        section_patterns = re.findall(
            r'\b(?:section|clause|article|exclusion|benefit|part|chapter)\s*[a-z0-9\-\.]*', text_lower)
        patterns.update(section_patterns)

        # Extract definition patterns
        definition_patterns = re.findall(r'\b\w+\s+(?:means|defined|refers\s+to)', text_lower)
        patterns.update(definition_patterns)

        # Extract procedural patterns
        procedural_patterns = re.findall(r'\b(?:requirement|condition|procedure|process|method|rule)\w*',
                                         text_lower)
        patterns.update(procedural_patterns)

        # Extract temporal patterns
        temporal_patterns = re.findall(r'\b(?:period|time|duration|days?|months?|years?)\b', text_lower)
        patterns.update(temporal_patterns)

    return list(patterns)


def multi_strategy_search(query: str, chunks: List[Dict], top_k: int = 15) -> List[Dict]:
    """Multi-strategy search for maximum accuracy across any document type"""

    query_lower = query.lower()
    query_words = [w.lower() for w in query.split() if len(w) > 2]

    # Extract key terms and patterns from query
    key_terms = set()

    # Learn domains dynamically from document
    learned_domains = detect_document_domains(chunks, query_words)
    detected_domains = list(learned_domains.keys())

    # Expand search terms based on learned associations
    for query_word, associated_terms in learned_domains.items():
        # Only add the most relevant associated terms (avoid noise)
        relevant_terms = [term for term in associated_terms
                          if len(term) > 3 and term not in query_words][:15]
        key_terms.update(relevant_terms)

        if relevant_terms:
            logger.debug(f"[DYNAMIC_DOMAIN] '{query_word}' -> learned {len(relevant_terms)} related terms")

    # If no associations learned, use basic structural terms from document
    if not learned_domains:
        basic_terms = set()
        for chunk in chunks[:10]:
            text_lower = chunk['text'].lower()
            # Extract structural terms that actually appear in this document
            structural_terms = re.findall(
                r'\b(?:section|article|definition|requirement|condition|process|method|rule|limit|amount|period|time|clause|exclusion|benefit)\b',
                text_lower)
            basic_terms.update(structural_terms)

        key_terms.update(basic_terms)
        logger.debug(f"[BASIC_TERMS] Used {len(basic_terms)} structural terms from document")

    # Learn semantic relationships from document
    semantic_relationships = extract_semantic_relationships(chunks, query)
    key_terms.update(semantic_relationships)

    if semantic_relationships:
        logger.debug(f"[SEMANTIC_LEARNING] Learned {len(semantic_relationships)} semantic relationships")

    # Add extracted keywords from top chunks
    chunk_keywords = set()
    for chunk in chunks[:20]:  # Sample top chunks
        dynamic_kw = chunk['metadata'].get('dynamic_keywords', [])
        chunk_keywords.update(dynamic_kw)

    # Add most relevant chunk keywords to search terms
    if chunk_keywords:
        key_terms.update(list(chunk_keywords)[:10])  # Add top 10 chunk keywords

    # Add query words and partial matching
    key_terms.update(query_words)

    # Add partial matching for compound terms
    for term in list(key_terms):
        if len(term) > 6:  # For longer terms, add partial matches
            key_terms.add(term[:4])  # First 4 characters

    # Extract patterns specific to this document
    important_patterns = extract_document_patterns(chunks)

    # Add learned domain-specific patterns
    for domain_term in detected_domains:
        if domain_term in learned_domains:
            # Add the most frequent terms associated with this domain
            domain_patterns = list(learned_domains[domain_term])[:10]
            important_patterns.extend(domain_patterns)

    logger.debug(f"[PATTERN_EXTRACTION] Found {len(important_patterns)} document-specific patterns")

    # Enhanced scoring for complete sections
    scored_chunks = []

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        metadata = chunk['metadata']
        score = 0

        # Start with ZERO base score - only relevant chunks get points

        # 1. EXACT PHRASE MATCHING (highest priority)
        if query_lower in text_lower:
            score += 500  # Much higher for exact matches
            logger.debug(f"[EXACT_MATCH] Found exact query phrase in chunk")

        # 2. QUERY-SPECIFIC TERM MATCHING (most important)
        query_term_score = 0
        total_query_words = len(query_words)
        matched_query_words = 0

        for query_word in query_words:
            if query_word in text_lower:
                matched_query_words += 1
                # Weight by word importance and frequency in chunk
                word_frequency = text_lower.count(query_word)
                word_score = len(query_word) * 10 + word_frequency * 5
                query_term_score += word_score
                logger.debug(f"[QUERY_TERM] '{query_word}' found {word_frequency} times: +{word_score}")

        # Only add score if significant portion of query is matched
        if matched_query_words >= max(1, total_query_words // 2):
            score += query_term_score
            # Bonus for matching most/all query terms
            completeness_bonus = (matched_query_words / total_query_words) * 200
            score += completeness_bonus
            logger.debug(
                f"[COMPLETENESS] Matched {matched_query_words}/{total_query_words} terms: +{completeness_bonus}")
        else:
            # Heavily penalize chunks that don't match main query terms
            score -= 100

        # 3. CONTEXTUAL PROXIMITY SCORING
        context_score = 0
        for i, query_word in enumerate(query_words):
            for j, other_word in enumerate(query_words[i + 1:], i + 1):
                if query_word in text_lower and other_word in text_lower:
                    # Find positions of both words
                    word1_positions = [m.start() for m in re.finditer(re.escape(query_word), text_lower)]
                    word2_positions = [m.start() for m in re.finditer(re.escape(other_word), text_lower)]

                    # Check if they appear close together (within 100 characters)
                    for pos1 in word1_positions:
                        for pos2 in word2_positions:
                            distance = abs(pos1 - pos2)
                            if distance <= 100:
                                proximity_score = max(0, 50 - distance // 2)
                                context_score += proximity_score
                                logger.debug(
                                    f"[PROXIMITY] '{query_word}' and '{other_word}' close together: +{proximity_score}")
                                break
        score += context_score

        # 4. SPECIFIC DETAIL EXTRACTION
        detail_score = 0

        # Look for specific numbers, amounts, dates mentioned with query terms
        for query_word in query_words:
            if query_word in text_lower:
                query_positions = [m.start() for m in re.finditer(re.escape(query_word), text_lower)]

                for pos in query_positions:
                    # Extract 50 characters around query word
                    start = max(0, pos - 50)
                    end = min(len(text_lower), pos + len(query_word) + 50)
                    context = text_lower[start:end]

                    # Look for specific details in context
                    numbers = re.findall(
                        r'(?:₹|rs\.?|usd|\$)?\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|thousands?|k|m|days?|months?|years?|hours?|%|percent))?\b',
                        context)
                    periods = re.findall(r'\d+\s*(?:days?|months?|years?|hours?)', context)
                    sections = re.findall(r'(?:section|clause|article|part)\s*[a-z0-9\-\.]+', context)

                    if numbers or periods or sections:
                        detail_boost = len(numbers) * 100 + len(periods) * 80 + len(sections) * 60
                        detail_score += detail_boost
                        logger.debug(
                            f"[SPECIFIC_DETAILS] Found {len(numbers)} numbers, {len(periods)} periods, {len(sections)} sections: +{detail_boost}")

        score += detail_score

        # 5. SECTION RELEVANCE (only if chunk actually contains relevant sections)
        section_score = 0
        relevant_sections = re.findall(r'(?:section|clause|article|part|chapter)\s*[a-z0-9\-\.]*', text_lower)
        if relevant_sections:
            # Check if section content is relevant to query
            for section in relevant_sections:
                section_pos = text_lower.find(section)
                if section_pos >= 0:
                    # Look for query terms near this section
                    section_start = max(0, section_pos - 100)
                    section_end = min(len(text_lower), section_pos + len(section) + 200)
                    section_context = text_lower[section_start:section_end]

                    relevant_terms_in_section = sum(1 for word in query_words if word in section_context)
                    if relevant_terms_in_section > 0:
                        section_boost = relevant_terms_in_section * 50
                        section_score += section_boost
                        logger.debug(
                            f"[SECTION_RELEVANCE] Section '{section}' contains {relevant_terms_in_section} query terms: +{section_boost}")

        score += section_score

        # 6. PRIORITY AND TYPE BONUSES (reduced weight)
        priority = metadata.get('priority', 0)
        score += priority // 2  # Reduced from full priority

        if metadata.get('type') == 'critical':
            score += 30  # Reduced from 60

        # 7. PENALIZE IRRELEVANT CHUNKS
        if score < 50:  # Very low relevance
            score = 0  # Don't include at all

        # Complete section bonus


        # PRECISION BOOST: Reward chunks with specific numbers, amounts, and exact details
        # 8. BASIC PRECISION BOOST - only if query-relevant
        if matched_query_words > 0:  # Only boost if chunk is relevant to query
            # Look for specific numbers/amounts near query terms
            for query_word in query_words:
                if query_word in text_lower:
                    positions = [m.start() for m in re.finditer(re.escape(query_word), text_lower)]
                    for pos in positions:
                        start = max(0, pos - 30)
                        end = min(len(text_lower), pos + len(query_word) + 30)
                        context = text_lower[start:end]

                        # Count specific details near query terms
                        numbers = len(re.findall(r'\d+', context))
                        if numbers > 0:
                            score += numbers * 20
                            logger.debug(f"[PRECISION] Found {numbers} numbers near '{query_word}': +{numbers * 20}")
        # UNIVERSAL PRECISION BOOST: Dynamic pattern detection


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

    # Extract top-K from original scoring
    preliminary_top_chunks = [chunk for chunk, _ in scored_chunks[:top_k * 2]]
    if scored_chunks:
        logger.info(f"[DEBUG] Search for '{query}' found {len(scored_chunks)} chunks")
        for i, (chunk, score) in enumerate(scored_chunks[:5]):
            preview = chunk['text'][:100].replace('\n', ' ')
            logger.info(f"[DEBUG] Rank {i + 1}: score={score}, preview='{preview}...'")
    else:
        logger.warning(f"[DEBUG] No chunks found for query: '{query}'")
        logger.info(f"[DEBUG] Available chunk types: {[c['metadata'].get('type', 'unknown') for c in chunks[:10]]}")

    # Enhanced debugging
    logger.info(f"[ADAPTIVE_SEARCH] Detected domains: {detected_domains}")
    logger.info(f"[ADAPTIVE_SEARCH] Key terms expanded to: {len(key_terms)} terms")
    logger.info(f"[ADAPTIVE_SEARCH] Dynamic patterns: {len(important_patterns)} patterns")

    # Rerank by metadata weight
    reranked_chunks = sorted(preliminary_top_chunks, key=metadata_weight_score, reverse=True)

    # Final top chunks
    final_chunks = reranked_chunks[:top_k]

    logger.info(f"[RERANK] Re-ranked top {top_k} chunks using metadata-weighted scoring")

    return final_chunks
class AdaptiveRetriever:
    def __init__(self):
        self.chunks = []
        self.is_indexed = False
        self.doc_hash = None
        # Performance tracking for hackathon evaluation
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.retrieval_metrics = []

    async def index_document(self, text: str):
        doc_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        if self.doc_hash == doc_hash and self.is_indexed:
            logger.info("[kg290] Document already indexed")
            return

        start = time.time()
        # Switch to comprehensive chunking for better accuracy
        self.chunks = smart_section_chunking(text)
        self.is_indexed = True
        self.doc_hash = doc_hash

        # Add indexing metrics for explainability
        chunk_types = {}
        for chunk in self.chunks:
            chunk_type = chunk['metadata'].get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        indexing_time = time.time() - start
        logger.info(f"[kg290] Indexed {len(self.chunks)} chunks in {indexing_time:.2f}s")
        logger.info(f"[kg290] Chunk distribution: {dict(chunk_types)}")

    async def get_adaptive_context(self, query: str, is_complex: bool = False) -> str:
        """Get context adapted to query complexity"""
        if not self.is_indexed:
            return "Document not indexed"

        # Check cache first for latency optimization
        cache_key = f"{hashlib.md5(query.encode()).hexdigest()[:8]}_{is_complex}"
        if cache_key in self.query_cache:
            self.cache_hits += 1
            logger.info(
                f"[kg290] Cache hit (rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%)")
            return self.query_cache[cache_key]

        self.cache_misses += 1
        start_time = time.time()

        # Adjust chunk count based on complexity
        chunk_count = 20 if is_complex else 18
        relevant_chunks = multi_strategy_search(query, self.chunks, top_k=chunk_count)

        # Soft fallback if too few relevant chunks
        if len(relevant_chunks) < 3:
            # 🔍 Include clause-bearing chunks (force add legal language chunks)
            clause_chunks = [
                c for c in self.chunks
                if re.search(r'\b(clause|section|article|exclusion|definition|limit)\b', c["text"].lower())
                   and c not in relevant_chunks
            ]

            # Add only if we're still under the desired chunk count
            needed = chunk_count - len(relevant_chunks)
            if needed > 0:
                relevant_chunks += clause_chunks[:needed]
                logger.info(
                    f"[kg290] Added {min(needed, len(clause_chunks))} clause-bearing chunks due to low relevant context")

            logger.info(f"[kg290] Only {len(relevant_chunks)} chunks found, augmenting with high-priority backups")

            # Select high-priority backup chunks
            high_priority_backups = sorted(
                [c for c in self.chunks if c['metadata'].get('priority', 0) > 50 and c not in relevant_chunks],
                key=lambda x: x['metadata']['priority'],
                reverse=True
            )[:(chunk_count - len(relevant_chunks))]

            # Append them to fill in
            relevant_chunks += high_priority_backups

        # Final hard fallback (should rarely happen)
        if not relevant_chunks:
            relevant_chunks = self.chunks[:chunk_count]
            logger.warning("[kg290] Using fallback chunks - may impact accuracy")

        # Build adaptive context with complexity-based limits
        context_parts = []
        total_chars = 0
        max_context = 9500 if is_complex else 8500  # More context for complex queries
        chunks_used = 0

        for i, chunk in enumerate(relevant_chunks):
            text = chunk['text']
            priority = chunk['metadata'].get('priority', 0)

            # Adaptive text inclusion based on complexity and priority
            # Adaptive text inclusion with precision-aware chunking
            chunk_metadata = chunk['metadata']
            has_numbers = chunk_metadata.get('has_numbers', False)
            chunk_type = chunk_metadata.get('type', 'general')

            if is_complex:
                # More generous limits for complex queries
                if priority > 80:
                    chunk_text = text[:1200] if len(text) > 1200 else text
                    label = f"[CRITICAL-{i + 1}]"
                elif priority > 50:
                    chunk_text = text[:800] if len(text) > 800 else text
                    label = f"[IMPORTANT-{i + 1}]"
                else:
                    chunk_text = text[:600] if len(text) > 600 else text
                    label = f"[{i + 1}]"
            else:
                # For simple queries, prioritize chunks with specific details
                if priority > 80 or has_numbers or chunk_type in ['critical', 'exclusion', 'benefit']:
                    chunk_text = text[:900] if len(text) > 900 else text  # More content for important chunks
                    label = f"[CRITICAL-{i + 1}]" if priority > 80 else f"[DETAILED-{i + 1}]"
                elif priority > 50:
                    chunk_text = text[:700] if len(text) > 700 else text
                    label = f"[IMPORTANT-{i + 1}]"
                else:
                    chunk_text = text[:500] if len(text) > 500 else text
                    label = f"[{i + 1}]"

            # Check if adding this chunk exceeds limit
            if total_chars + len(chunk_text) > max_context:
                # Truncate to fit but ensure minimum useful content
                remaining = max_context - total_chars
                if remaining > 150:  # Only add if meaningful amount left
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(f"{label} {chunk_text}")
                    chunks_used += 1
                break

            context_parts.append(f"{label} {chunk_text}")
            total_chars += len(chunk_text)
            chunks_used += 1

        context = "\n\n".join(context_parts)
        retrieval_time = time.time() - start_time

        # Cache the result for future queries (latency optimization)
        if len(self.query_cache) < 50:  # Limit cache size to prevent memory issues
            self.query_cache[cache_key] = context

        # Track metrics for explainability
        self.retrieval_metrics.append({
            'query_length': len(query),
            'is_complex': is_complex,
            'chunks_retrieved': chunks_used,
            'context_length': len(context),
            'retrieval_time': retrieval_time,
            'cache_hit': False
        })

        complexity_note = " (complex query)" if is_complex else ""
        logger.info(
            f"[kg290] Adaptive context: {len(context)} chars from {chunks_used}/{len(relevant_chunks)} chunks in {retrieval_time:.2f}s{complexity_note}")
        return context

    async def get_clause_priority_context(self, query: str) -> str:
        """Get context prioritizing legal/policy language for retry scenarios"""
        if not self.is_indexed:
            return "Document not indexed"

        start_time = time.time()

        # Prioritize chunks with legal/policy patterns
        legal_patterns = [
            r'\b(clause|section|article|provision|exclusion|definition|benefit|coverage|condition|limitation|term)\b',
            r'\b(waiting\s+period|grace\s+period|sum\s+insured|deductible|premium|eligibility)\b',
            r'\b(means|refers\s+to|defined\s+as|shall\s+mean|includes|excludes)\b'
        ]

        priority_chunks = []
        for chunk in self.chunks:
            text_lower = chunk['text'].lower()
            legal_score = sum(1 for pattern in legal_patterns if re.search(pattern, text_lower))

            if legal_score > 0:
                # Boost priority for legal language
                enhanced_priority = chunk['metadata'].get('priority', 0) + (legal_score * 25)
                chunk_copy = {
                    'text': chunk['text'],
                    'metadata': {**chunk['metadata'], 'priority': enhanced_priority, 'legal_score': legal_score}
                }
                priority_chunks.append(chunk_copy)

        # Fallback to all chunks if no legal chunks found
        if not priority_chunks:
            priority_chunks = self.chunks[:15]
            logger.warning("[kg290] No legal chunks found, using general chunks")

        # Sort by enhanced priority
        priority_chunks.sort(key=lambda x: x['metadata'].get('priority', 0), reverse=True)
        top_chunks = priority_chunks[:12]

        # Build context with legal emphasis
        context_parts = []
        total_chars = 0
        max_context = 8000

        for i, chunk in enumerate(top_chunks):
            text = chunk['text'][:600]  # Controlled chunk size
            legal_score = chunk['metadata'].get('legal_score', 0)

            if legal_score >= 2:
                label = f"[LEGAL-{i + 1}]"
            elif chunk['metadata'].get('priority', 0) > 80:
                label = f"[CRITICAL-{i + 1}]"
            else:
                label = f"[CLAUSE-{i + 1}]"

            if total_chars + len(text) > max_context:
                break

            context_parts.append(f"{label} {text}")
            total_chars += len(text)

        context = "\n\n".join(context_parts)
        retrieval_time = time.time() - start_time

        logger.info(
            f"[kg290] Clause priority context: {len(context)} chars from {len(top_chunks)} chunks in {retrieval_time:.2f}s")
        return context

    def get_performance_metrics(self) -> dict:
        """Get retrieval performance metrics for explainability"""
        if not self.retrieval_metrics:
            return {"status": "no_metrics_available"}

        total_retrievals = len(self.retrieval_metrics)
        total_cache_ops = self.cache_hits + self.cache_misses

        return {
            "retrieval_stats": {
                "total_retrievals": total_retrievals,
                "avg_retrieval_time": round(sum(m['retrieval_time'] for m in self.retrieval_metrics) / total_retrievals,
                                            3),
                "avg_chunks_per_query": round(
                    sum(m['chunks_retrieved'] for m in self.retrieval_metrics) / total_retrievals, 1),
                "avg_context_length": round(sum(m['context_length'] for m in self.retrieval_metrics) / total_retrievals,
                                            0),
                "complex_query_ratio": round(
                    sum(1 for m in self.retrieval_metrics if m['is_complex']) / total_retrievals, 2)
            },
            "cache_performance": {
                "cache_hit_rate": round((self.cache_hits / max(total_cache_ops, 1)) * 100, 1),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.query_cache)
            },
            "indexing_status": {
                "is_indexed": self.is_indexed,
                "total_chunks": len(self.chunks),
                "doc_hash": self.doc_hash
            }
        }

    def clear_cache(self):
        """Clear query cache for memory management"""
        self.query_cache.clear()
        logger.info("[kg290] Query cache cleared")

    def reset_metrics(self):
        """Reset performance metrics"""
        self.retrieval_metrics.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("[kg290] Performance metrics reset")

# Global retriever
retriever = AdaptiveRetriever()

# Concise adaptive system prompts
ADAPTIVE_PRIMARY_PROMPT = """

You are an expert document analyst specializing in policy documents. Provide accurate, concise answers based strictly on the document context.

RESPONSE FORMAT (JSON):
{
  "answer": "Clear, concise answer with essential facts, numbers, and conditions",
  "confidence": "high | medium | low",
  "needs_more_time": "true | false"
}

FORMATTING & ACCURACY RULES:
1. Keep answers concise (30–40 words) – ensure essential information is conveyed without verbosity
2. Include ALL exact numbers, percentages, time periods as stated (e.g., “thirty (30) days”, “5%”, “24 months”)
3. Faithfully reflect clause numbers or section names if present (e.g., “Clause 2.1 specifies…”)
4. For definitions, provide the exact wording as stated in the document wherever possible
5. If information exists in context, provide it – never say "not mentioned" without thorough analysis
6. Use clear, professional language – DO NOT use \\n or line break characters
7. Include key conditions using phrases like "provided that", "subject to", "limited to"
8. ALWAYS cite exact section numbers when available (e.g., "Exclusion 3.4 states...", "Benefit 2.1 covers...")
9. Quote exact phrases from document sections when defining terms or stating rules
10. Set confidence HIGH if answer is definitive, MEDIUM if partially found, LOW if uncertain
11. Set needs_more_time to TRUE only if question is very complex and you need more analysis time
12. Write in flowing prose without literal newlines or formatting characters
13. Avoid bullet points, excessive sub-sections, or overly detailed explanations

TRACEABILITY:
- Reflect all numbers, limits, time durations, and clause names exactly as in the document
- Use phrases like “Clause 1.3 defines…”, “Section B states…”, or “subject to a maximum of…”

CONCISENESS PRIORITY:
Provide complete but succinct answers within 30–40 words. Focus on clarity and completeness without adding extra detail.

COMPLEXITY ASSESSMENT:
- Set needs_more_time=true for: complex definitions, multi-part questions, comprehensive coverage requests
- Set needs_more_time=false for: simple yes/no, basic coverage, straightforward facts

BALANCE:
Provide complete information while maintaining readability and professional conciseness. Match the answer length to formal policy summaries and regulatory communication styles.

"""

ADAPTIVE_FALLBACK_PROMPT = """

You are a senior document expert handling a complex query that needs thorough analysis. Provide comprehensive but concise answers.

RESPONSE FORMAT (JSON):
{
  "answer": "Comprehensive answer with complete information, properly formatted and clearly structured"
}

EXPERT ANALYSIS FOR COMPLEX QUERIES:
1. Conduct thorough analysis of ALL context sections
2. Include ALL numerical details, percentages, time periods, conditions
3. Provide complete eligibility criteria and exceptions
4. Cross-reference multiple sections for comprehensive coverage
5. Use clear, professional language with logical structure – NO \n characters
6. Include all qualifying conditions and procedural requirements
7. Format for maximum clarity while being thorough but concise
8. Ensure no important detail is omitted
9. Provide complete benefit structures and limitations
10. Faithfully include **clause references, section names, or article numbers** when present in the context
11. Write in flowing, professional prose without special formatting characters
12. Avoid excessive elaboration – focus on essential information
13. Keep the answer within 30–40 words whenever possible, unless truly unavoidable due to query complexity

CONCISENESS: Even for complex queries, aim to deliver complete and structured answers in 30–40 words where feasible, while maintaining professional brevity and clarity.

THOROUGHNESS: This is a complex query requiring detailed analysis – be comprehensive while ensuring the response remains focused, concise, and free from formatting artifacts like literal newlines or unnecessary repetition. **Ensure any exact numbers, time periods, and policy conditions found in the document context are clearly mirrored in the answer.**

"""
SELF_EVAL_PROMPT = """
You are verifying whether the following answer to a document-based question is complete and well-supported.

QUESTION: {question}

ANSWER: {answer}

Check:
1. Does it cover all relevant clauses, conditions, and numbers?
2. Does it use traceable language (e.g., 'Clause 2.3', 'subject to a limit of...', 'waiting period of 30 days')?
3. Is the answer vague or generic?

Respond with one word: COMPLETE, INCOMPLETE, or AMBIGUOUS.
"""



async def analyze_with_adaptive_timing(query: str, context: str, is_complex: bool = False,
                                       complexity_reason: str = "") -> Dict[str, Any]:
    """Adaptive analysis that adjusts timing based on query complexity"""
    try:
        # Adaptive context limits
        context_limit = 10000 if is_complex else 8500
        if len(context) > context_limit:
            context = context[:context_limit] + "..."

        # Adaptive token estimation
        base_tokens = int(len(context + query) / 3.5) + 150
        estimated_tokens = int(base_tokens * 1.2) if is_complex else base_tokens

        # Step 1: Primary model with adaptive settings
        api_key = groq_pool.get_adaptive_key(estimated_tokens, use_fallback=False, is_complex=is_complex)

        # Adaptive timeout and max_tokens - reduced for conciseness
        timeout = 40 if is_complex else 30
        max_tokens = int(base_tokens * 1.3)   # Reduced for more concise answers

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": ADAPTIVE_PRIMARY_PROMPT},
                {"role": "user",
                 "content": f"QUESTION: {query}\n\nDOCUMENT CONTEXT:\n{context}\n\nProvide concise, accurate answer (complexity: {complexity_reason}):"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.01,
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
        fallback_trigger_keywords = [
            "not mentioned",
            "unable to determine",
            "information not available",
            "no mention",
            "not specified",
            "analysis failed",
            "context missing",
            "nothing found",
            "no clear answer",
            "insufficient context",
            "unknown",
            "unable to verify",  # This is what your system is returning
            "document does not specify",
            "not explicitly mentioned"
        ]

        should_use_fallback = (
                confidence == "low"
                or len(answer) < 50
                or len(answer.split()) < 12
                or any(phrase in answer.lower() for phrase in fallback_trigger_keywords)
                or needs_more_time
                or (is_complex and confidence == "medium")
        )

        if should_use_fallback:
            complexity_note = f" (complex: {complexity_reason})" if is_complex else ""
            logger.info(f"[Jay121305] Primary {confidence} confidence{complexity_note}, using adaptive fallback...")

            try:
                fallback_key = groq_pool.get_adaptive_key(estimated_tokens, use_fallback=True, is_complex=is_complex)

                fallback_timeout = 45 if is_complex else 35
                fallback_tokens = 1200 if is_complex else 900  # Reduced for conciseness

                fallback_payload = {
                    "model": "moonshotai/kimi-k2-instruct",
                    "messages": [
                        {"role": "system", "content": ADAPTIVE_FALLBACK_PROMPT},
                        {"role": "user",
                         "content": f"QUESTION: {query}\n\nDOCUMENT CONTEXT:\n{context}\n\nConcise analysis needed ({complexity_reason}). Primary had {confidence} confidence:"}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.01,
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

            # Adaptive delay
            delay = (i % 6) * (0.4 if is_complex else 0.2) + random.uniform(0.2, 0.5 if is_complex else 0.2)
            await asyncio.sleep(delay)

            # 🔹 STEP 1: First attempt
            context_1 = await retriever.get_adaptive_context(question, is_complex)
            result_1 = await analyze_with_adaptive_timing(question, context_1, is_complex, complexity_reason)
            answer = result_1.get("answer", "")
            retry_reason = None

            # 🔹 STEP 2: Trigger retry if vague or low confidence
            vague_phrases = [
                "not explicitly mentioned", "not mentioned", "not clearly stated", "provided context",
                "not available", "not described", "not stated", "not defined", "focuses on", "cannot be determined"
            ]
            if any(p in answer.lower() for p in vague_phrases):
                retry_reason = "Vague answer"
            elif result_1.get("confidence", "unknown") in {"low", "unknown"}:
                retry_reason = "Low confidence"

            # 🔹 STEP 3: Retry if needed
            if retry_reason:
                logger.warning(f"[RERUN] Triggered retry due to: {retry_reason}")
                context_2 = await retriever.get_clause_priority_context(question)
                result_2 = await analyze_with_adaptive_timing(question, context_2, True, "retry")

                fallback_answer = result_2.get("answer", "")
                if fallback_answer and fallback_answer != answer and "unable to verify" not in fallback_answer.lower():
                    answer = fallback_answer
                    logger.info(f"[RERUN] Used fallback answer")

            # 🔹 STEP 4: Final cleaning
            final_answer = clean_answer_formatting(answer)

            elapsed = time.time() - question_start
            tokens = result_1.get('tokens_used', 0)
            model_used = result_1.get('model_used', 'Unknown')
            confidence = result_1.get('confidence', 'unknown')
            complexity = result_1.get('complexity', 'simple')

            timing_emoji = "🕐" if elapsed > 2.0 else "⚡"
            logger.info(
                f"[Jay121305] Q{i + 1} ✓ {timing_emoji} | {elapsed:.1f}s | {tokens}t | {model_used} | {confidence} | {complexity}")

            return final_answer

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

                vague_phrases = [
                    "not explicitly mentioned",
                    "not mentioned",
                    "not clearly stated",
                    "provided context",
                    "not available in excerpt",
                    "not described",
                    "not stated",
                    "not defined",
                    "focuses on",
                    "cannot be determined"
                ]

                if any(phrase in cleaned_answer.lower() for phrase in vague_phrases):
                    cleaned_answer = "Unable to verify from available document context"

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