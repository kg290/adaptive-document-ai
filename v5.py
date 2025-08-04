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

    # REDUCED weights to prevent generic chunks from dominating
    return (
        0.2 * priority +           # Reduced from 0.4
        0.1 * (coverage / 1000) +  # Reduced from 0.3
        0.1 * has_numbers +        # Reduced from 0.2
        0.05 * has_percent +       # Reduced from 0.1
        0.15 * clause_like         # Reduced from 0.3
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
        """Reset counters every minute and handle cooldown properly"""
        now = time.time()
        stats = self.key_stats[key]

        # ALWAYS check cooldown first, regardless of minute reset
        if stats['is_cooling'] and now >= stats['cooldown_until']:
            stats['is_cooling'] = False
            stats['consecutive_fails'] = 0
            logger.info(f"[Jay121305] Key ...{key[-5:]} cooled down and ready")

        # Reset counters every minute
        if now - stats['last_reset'] >= 60:
            stats['primary_tokens_used'] = 0
            stats['fallback_tokens_used'] = 0
            stats['requests_made'] = 0
            stats['last_reset'] = now
            logger.info(f"[Jay121305] Key ...{key[-5:]} tokens reset for new minute")

    def get_adaptive_key(self, estimated_tokens=1000, use_fallback=False, is_complex=False):
        """Robust adaptive key selection with auto-unstick for submission"""
        with self.lock:
            tpm_limit = self.fallback_tpm if use_fallback else self.primary_tpm
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            # Adjust estimated token budget for complex queries
            if is_complex:
                estimated_tokens = int(estimated_tokens * 1.3)

            usage_limit = 0.85 if not is_complex else 0.75

            # 🔁 Reset counters for ALL keys
            for key in self.keys:
                self.reset_counters_if_needed(key)

            # 🚨 AUTO-UNSTICK: Force recovery if too many keys are stuck
            stuck_keys = sum(1 for key in self.keys if self.key_stats[key]['is_cooling'])
            if stuck_keys >= len(self.keys) - 1:  # If 5+ out of 6 keys stuck
                logger.warning(f"[Jay121305] AUTO-UNSTICK: {stuck_keys}/{len(self.keys)} keys stuck, forcing recovery")
                now = time.time()
                for key in self.keys:
                    stats = self.key_stats[key]
                    if stats['is_cooling']:
                        # Auto-unstick if stuck for more than 30 seconds
                        if now - (stats.get('cooldown_until', now) - 25) > 30:
                            stats['is_cooling'] = False
                            stats['consecutive_fails'] = 0
                            stats['cooldown_until'] = 0
                            logger.warning(f"[Jay121305] AUTO-UNSTUCK key ...{key[-5:]}")

            # ⚖️ Build list of usable keys
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

            # ✅ If usable keys exist, use least loaded
            if usable_keys:
                usable_keys.sort(key=lambda x: x[1])
                selected_key = usable_keys[0][0]
                return selected_key

            # 🔄 EMERGENCY: If still no keys, force unstick ALL and retry ONCE
            logger.error(f"[Jay121305] EMERGENCY: All keys unavailable, forcing system recovery")
            for key in self.keys:
                stats = self.key_stats[key]
                stats['is_cooling'] = False
                stats['consecutive_fails'] = 0
                stats['cooldown_until'] = 0
                # Reset usage to allow immediate use
                stats[token_field] = 0
                stats['requests_made'] = 0

            # Return first key after emergency recovery
            return self.keys[0]

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
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6")
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
    """Improved chunking that preserves complete policy sections"""
    chunks = []

    # Step 1: Split by major section headers first
    major_sections = re.split(r'\n\s*(?:SECTION|CLAUSE|ARTICLE|EXCLUSION|BENEFIT)\s*\d+', text, flags=re.IGNORECASE)

    if len(major_sections) < 5:  # If no major sections found, try numbered sections
        major_sections = re.split(r'\n\s*\d+\.\s+[A-Z]', text)

    if len(major_sections) < 5:  # Still not enough, split by paragraphs
        major_sections = text.split('\n\n')

    # Step 2: Create chunks ensuring important content stays together
    for i, section in enumerate(major_sections):
        section = section.strip()
        if len(section) < 100:  # Skip tiny sections
            continue

        # Keep complete sections under 1000 chars together
        if len(section) <= 1000:
            chunks.append(create_chunk_simple(section, i))
        else:
            # Split longer sections by sentences, keeping related content together
            sentences = sent_tokenize(section)
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) > 800 and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(create_chunk_simple(chunk_text, len(chunks)))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)

            # Don't forget the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(create_chunk_simple(chunk_text, len(chunks)))

    logger.info(f"[IMPROVED_CHUNKING] Created {len(chunks)} chunks preserving policy sections")
    return chunks[:100]  # Limit to reasonable number
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


def debug_search_terms(query: str, chunks: List[Dict]) -> None:
    """Debug why search might be failing"""
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    logger.info(f"[DEBUG_SEARCH] Query: '{query}'")
    logger.info(f"[DEBUG_SEARCH] Search words: {query_words}")

    # Check if any chunks contain the words
    word_counts = {}
    for word in query_words:
        count = sum(1 for chunk in chunks if word in chunk['text'].lower())
        word_counts[word] = count

    logger.info(f"[DEBUG_SEARCH] Word occurrences in {len(chunks)} chunks: {word_counts}")

    # Sample some chunk content
    if chunks:
        sample_text = chunks[0]['text'][:200].lower()
        logger.info(f"[DEBUG_SEARCH] Sample chunk content: '{sample_text}...'")


def multi_strategy_search(query: str, chunks: List[Dict], top_k: int = 15) -> List[Dict]:
    """Dynamic search that adapts to query intent and document structure"""

    query_lower = query.lower()
    query_words = [w.lower() for w in query.split() if len(w) > 2]

    # Dynamic query analysis
    query_intent = analyze_query_intent(query_lower)
    key_terms = extract_key_terms(query_lower)

    scored_chunks = []

    for chunk in chunks:
        text_lower = chunk['text'].lower()
        score = 0

        # 1. EXACT PHRASE MATCHING
        if query_lower in text_lower:
            score += 2000

        # 2. KEY TERM MATCHING with dynamic weighting
        matched_words = 0
        for word in query_words:
            if word in text_lower:
                matched_words += 1
                frequency = text_lower.count(word)

                # Dynamic scoring based on term importance
                term_weight = get_term_weight(word, query_intent)
                score += (150 * term_weight) + (frequency * 20)

        # 3. SEMANTIC PROXIMITY SCORING
        score += calculate_semantic_proximity(query_words, text_lower)

        # 4. INTENT-SPECIFIC SCORING
        score += apply_intent_scoring(query_intent, text_lower, matched_words)

        # 5. DOCUMENT STRUCTURE AWARENESS
        score += evaluate_document_context(chunk, query_intent, key_terms)

        # 6. CONTENT RELEVANCE FILTERING
        if is_relevant_content(text_lower, key_terms, query_intent):
            score += 200
        else:
            score = max(0, score - 300)  # Penalize irrelevant content

        # 7. ANSWER PATTERN DETECTION
        if contains_answer_pattern(text_lower, query_intent):
            score += 800

        if score > 0:
            scored_chunks.append((chunk, score, matched_words))

    # Dynamic sorting and filtering
    scored_chunks.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Remove duplicates and improve quality
    unique_chunks = remove_duplicates_and_filter(scored_chunks, query_intent)

    logger.info(f"[DYNAMIC_SEARCH] Intent: {query_intent}, Found {len(unique_chunks)} chunks")
    if unique_chunks:
        logger.info(f"[DYNAMIC_SEARCH] Top score: {unique_chunks[0][1]}")
        for i, (chunk, score, matches) in enumerate(unique_chunks[:3]):
            preview = chunk['text'][:80].replace('\n', ' ')
            logger.info(
                f"[DYNAMIC_DEBUG] Rank {i + 1}: score={score}, intent_match={query_intent}, text='{preview}...'")

    return [chunk for chunk, score, matches in unique_chunks[:top_k]]


def analyze_query_intent(query_lower: str) -> str:
    """Dynamically determine what the user is looking for"""
    intent_patterns = {
        'time_period': ['grace period', 'waiting period', 'how long', 'days', 'months', 'years'],
        'definition': ['what is', 'define', 'definition of', 'means', 'what does'],
        'coverage': ['covered', 'cover', 'benefits', 'include', 'expense'],
        'limits': ['limit', 'maximum', 'minimum', 'sub-limit', 'up to'],
        'conditions': ['conditions', 'criteria', 'requirements', 'eligible'],
        'exclusions': ['not covered', 'exclude', 'exception', 'restriction'],
        'discount': ['discount', 'ncd', 'bonus', 'claim-free'],
        'specific_benefit': ['maternity', 'cataract', 'donor', 'ayush', 'hospital']
    }

    for intent, patterns in intent_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            return intent

    return 'general'


def extract_key_terms(query_lower: str) -> List[str]:
    """Extract the most important terms from the query"""
    # Remove common words
    stop_words = {'is', 'the', 'for', 'and', 'or', 'in', 'on', 'at', 'to', 'a', 'an', 'this', 'that'}
    words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

    # Priority terms that are highly specific
    priority_terms = []
    important_terms = []

    for word in words:
        if any(key in word for key in
               ['grace', 'waiting', 'maternity', 'cataract', 'donor', 'ncd', 'ayush', 'hospital']):
            priority_terms.append(word)
        elif any(key in word for key in ['period', 'coverage', 'limit', 'benefit', 'condition']):
            important_terms.append(word)

    return priority_terms + important_terms + words[:5]  # Return most relevant terms


def get_term_weight(word: str, intent: str) -> float:
    """Dynamic weighting based on term importance and query intent"""

    # High-value terms for different intents
    intent_weights = {
        'time_period': {'grace': 3.0, 'waiting': 3.0, 'period': 2.5, 'days': 2.0, 'months': 2.0},
        'definition': {'hospital': 3.0, 'means': 2.5, 'defined': 2.5, 'definition': 2.0},
        'coverage': {'covered': 2.5, 'expenses': 2.0, 'benefit': 2.0, 'include': 1.5},
        'discount': {'ncd': 3.0, 'discount': 2.5, 'bonus': 2.5, 'claim': 2.0},
        'specific_benefit': {'maternity': 3.0, 'cataract': 3.0, 'donor': 3.0, 'ayush': 3.0}
    }

    if intent in intent_weights and word in intent_weights[intent]:
        return intent_weights[intent][word]

    return 1.0  # Default weight


def calculate_semantic_proximity(query_words: List[str], text: str) -> int:
    """Calculate how close query words appear together in text"""
    score = 0

    for i, word1 in enumerate(query_words):
        if word1 in text:
            for word2 in query_words[i + 1:]:
                if word2 in text:
                    pos1 = text.find(word1)
                    pos2 = text.find(word2)
                    distance = abs(pos1 - pos2)

                    if distance <= 50:
                        score += 150
                    elif distance <= 100:
                        score += 100
                    elif distance <= 200:
                        score += 50

    return score


def apply_intent_scoring(intent: str, text: str, matched_words: int) -> int:
    """Apply scoring based on detected intent"""
    if matched_words == 0:
        return 0

    score = 0

    if intent == 'time_period':
        # Look for time expressions
        if re.search(r'\d+\s*(?:days?|months?|years?)', text):
            score += 500
        if any(term in text for term in ['grace', 'waiting', 'period']):
            score += 300

    elif intent == 'definition':
        # Look for definition patterns
        if re.search(r'\b\w+\s+means\b', text):
            score += 600
        if any(pattern in text for pattern in ['defined as', 'refers to', 'shall mean']):
            score += 400

    elif intent == 'coverage':
        # Look for coverage information
        if any(term in text for term in ['covered', 'benefits', 'expenses', 'include']):
            score += 300
        if re.search(r'(?:shall be|will be|are)\s+covered', text):
            score += 400

    elif intent == 'limits':
        # Look for numerical limits
        if re.search(r'(?:maximum|minimum|limit|up to)\s+(?:₹|\d)', text):
            score += 500
        if re.search(r'\d+\s*%', text):
            score += 300

    elif intent == 'discount':
        # Look for discount/bonus information
        if any(term in text for term in ['discount', 'bonus', 'ncd', 'claim-free']):
            score += 400
        if re.search(r'\d+\s*%.*(?:discount|bonus)', text):
            score += 600

    return score


def evaluate_document_context(chunk: Dict, intent: str, key_terms: List[str]) -> int:
    """Evaluate chunk based on document structure and metadata"""
    metadata = chunk.get('metadata', {})
    text = chunk['text'].lower()
    score = 0

    # Boost based on chunk metadata
    if metadata.get('priority', 0) > 80:
        score += 100

    # Boost for structured content
    if any(term in text for term in ['section', 'clause', 'article']):
        score += 50

    # Boost for policy-specific terms
    policy_terms = ['policy', 'insured', 'premium', 'claim', 'benefit', 'coverage']
    policy_matches = sum(1 for term in policy_terms if term in text)
    score += policy_matches * 20

    return score


def is_relevant_content(text: str, key_terms: List[str], intent: str) -> bool:
    """Filter out obviously irrelevant content"""

    # Blacklist obviously irrelevant content
    irrelevant_phrases = [
        'trade logo', 'belongs to hdfc', 'ergo international',
        'natural parents', 'legally adopted'
    ]

    if any(phrase in text for phrase in irrelevant_phrases):
        return False

    # Must contain at least one key term for most intents
    if intent != 'general' and not any(term in text for term in key_terms):
        return False

    return True


def contains_answer_pattern(text: str, intent: str) -> bool:
    """Detect if text likely contains the answer"""

    if intent == 'time_period':
        return bool(re.search(r'\d+\s*(?:days?|months?|years?)', text))

    elif intent == 'definition':
        return bool(re.search(r'\w+\s+means|defined as|refers to', text))

    elif intent == 'coverage':
        return any(pattern in text for pattern in ['covered', 'shall be', 'include', 'benefit'])

    elif intent == 'discount':
        return bool(re.search(r'\d+\s*%|discount|bonus|ncd', text))

    elif intent == 'limits':
        return bool(re.search(r'(?:maximum|minimum|limit|up to).*\d', text))

    return True  # Default to true for general queries


def remove_duplicates_and_filter(scored_chunks: List, intent: str) -> List:
    """Remove duplicates and filter based on quality"""
    seen_texts = set()
    unique_chunks = []

    for chunk, score, matches in scored_chunks:
        # Create a more sophisticated deduplication key
        text = chunk['text']
        dedup_key = ' '.join(text.split()[:15])  # First 15 words

        if dedup_key not in seen_texts:
            seen_texts.add(dedup_key)
            unique_chunks.append((chunk, score, matches))

        # Limit based on intent
        max_chunks = 30 if intent in ['definition', 'time_period'] else 20
        if len(unique_chunks) >= max_chunks:
            break

    return unique_chunks
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

    async def get_clause_priority_context(self, query: str, max_chunks: int = 12) -> str:
        """Fallback method for clause priority context"""
        if not self.is_indexed:
            return "Document not indexed"

        # Use fixed search to find relevant chunks
        relevant_chunks = multi_strategy_search(query, self.chunks, top_k=max_chunks)

        # Build simple context
        context_parts = []
        total_chars = 0
        max_context = 7000

        for i, chunk in enumerate(relevant_chunks):
            text = chunk['text']
            chunk_text = text[:600] if len(text) > 600 else text

            if total_chars + len(chunk_text) > max_context:
                break

            context_parts.append(f"[CLAUSE-{i + 1}] {chunk_text}")
            total_chars += len(chunk_text)

        context = "\n\n".join(context_parts)
        logger.info(f"[CLAUSE_PRIORITY] Built context: {len(context)} chars from {len(context_parts)} clauses")
        return context

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
        """Simplified context building focused on relevant content"""
        if not self.is_indexed:
            return "Document not indexed"

        # Get relevant chunks using fixed search
        relevant_chunks = multi_strategy_search(query, self.chunks, top_k=20)

        # If very few relevant chunks, add high-priority fallbacks
        if len(relevant_chunks) < 5:
            logger.warning(f"[CONTEXT] Only {len(relevant_chunks)} relevant chunks, adding fallbacks")

            # Add chunks that contain any query words
            query_words = query.lower().split()
            fallback_chunks = []

            for chunk in self.chunks:
                if chunk in relevant_chunks:
                    continue
                chunk_text = chunk['text'].lower()
                if any(word in chunk_text for word in query_words if len(word) > 3):
                    fallback_chunks.append(chunk)

            relevant_chunks.extend(fallback_chunks[:12])
            logger.info(f"[CONTEXT] Added {len(fallback_chunks[:12])} fallback chunks")

        # Build context with clear labeling
        context_parts = []
        total_chars = 0
        max_context = 12000 if is_complex else 9750

        for i, chunk in enumerate(relevant_chunks):
            text = chunk['text']

            # Use more of each chunk to ensure complete information
            chunk_text = text[:1200] if len(text) > 1200 else text

            if total_chars + len(chunk_text) > max_context:
                remaining = max_context - total_chars
                if remaining > 300:  # Only add if meaningful content can fit
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(f"[SECTION-{i + 1}] {chunk_text}")
                break

            context_parts.append(f"[SECTION-{i + 1}] {chunk_text}")
            total_chars += len(chunk_text)

        context = "\n\n".join(context_parts)
        logger.info(f"[CONTEXT] Built context: {len(context)} chars from {len(context_parts)} sections")
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
You are a document analyst specializing in extracting specific information from policy documents.

RESPONSE FORMAT (JSON):
{
  "answer": "Direct, specific answer with exact numbers/periods from the document",
  "confidence": "high | medium | low"
}

CRITICAL RULES:
1. FIND EXACT INFORMATION: Look for specific numbers, periods, amounts, percentages
2. QUOTE DIRECTLY: Use exact wording from document sections
3. BE SPECIFIC: "30 days grace period" not "grace period exists"
4. CITE SECTIONS: Reference specific clauses/sections when found
5. ADMIT GAPS: If specific details aren't in provided sections, state clearly

EXAMPLES:
✅ GOOD: "30 days grace period as per Section 2.21"
✅ GOOD: "36 months waiting period for pre-existing diseases"
✅ GOOD: "5% NCD discount on base premium for claim-free policies"

❌ AVOID: "Grace period exists but duration not specified"
❌ AVOID: "Document does not explicitly state"
❌ AVOID: Generic statements without specific numbers

If the provided context sections do not contain the specific information requested, state: "The specific information is not available in the provided document sections."
"""

ADAPTIVE_FALLBACK_PROMPT = """
You are a senior document expert with expertise in extracting specific numerical details from complex policy documents.

RESPONSE FORMAT (JSON):
{
  "answer": "Comprehensive answer with all specific numbers, amounts, time periods, and exact conditions found in the document"
}

EXPERT NUMERICAL EXTRACTION:
1. Scan ALL context sections for specific numerical information
2. Extract EXACT amounts, percentages, time periods, and limits
3. Look for benefit tables, schedules, and coverage matrices
4. Find precise eligibility criteria and exclusion conditions
5. Identify specific premium rates, discounts, and charges
6. Extract exact waiting periods for different conditions
7. Locate precise age limits, tenure requirements, and policy terms

COMPREHENSIVE ANALYSIS PRIORITIES:
1. FINANCIAL DETAILS: All ₹ amounts, premium rates, benefit limits
2. TIME PERIODS: All waiting periods, grace periods, policy terms
3. PERCENTAGES: All discount rates, co-payment percentages, coverage ratios
4. CONDITIONS: All specific eligibility criteria and exclusion terms
5. LIMITS: All maximum/minimum amounts, age limits, coverage caps

THOROUGHNESS REQUIREMENTS:
- Cross-reference multiple sections for complete numerical picture
- Include ALL relevant numbers found, not just the first one
- Specify the exact context where each number applies
- Provide complete benefit structures with all amounts
- Include all qualifying conditions with their specific requirements

PRECISION FOCUS:
Even for complex queries, prioritize extracting and presenting ALL specific numerical information clearly. If the document contains exact numbers, amounts, or periods related to the query, these MUST be included in your response.

CONTEXT ANALYSIS:
When you see benefit tables, schedules, or numerical lists in the context, extract the specific details that answer the query rather than providing general descriptions of what these sections contain.
"""

SELF_EVAL_PROMPT = """
You are verifying whether the following answer contains specific numerical details as required.

QUESTION: {question}

ANSWER: {answer}

EVALUATION CRITERIA:
1. Does it include specific numbers, amounts, or time periods when the question asks for them?
2. Does it extract exact figures from benefit tables or schedules?
3. Does it provide precise waiting periods, coverage amounts, or discount rates?
4. Is it specific rather than generic (e.g., "24 months" vs "waiting period")?
5. Does it include exact clause/section references for traceability?

EVALUATION STANDARDS:
- COMPLETE: Contains specific numbers/amounts that directly answer the question
- INCOMPLETE: Missing specific numerical details that should be available
- AMBIGUOUS: Contains some numbers but lacks precision or completeness

Respond with one word: COMPLETE, INCOMPLETE, or AMBIGUOUS.
"""

# Add this to your system prompt for definition questions:
DEFINITION_PROMPT_ADDITION = """
For DEFINITION questions specifically:
- Look for exact text like "Hospital means..." or "AYUSH refers to..."
- If no exact definition found, check exclusions/inclusions for clues
- Report "Definition not provided in available context" if truly missing
- Do not say "Processing failed" - always provide some response
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
            confidence = parsed.get("confidence", "high").lower()
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
                and len(answer) < 40  # Changed OR to AND, reduced threshold
                or any(phrase in answer.lower() for phrase in fallback_trigger_keywords[:8])  # Fewer trigger phrases
                or (is_complex and confidence == "low")  # Only low confidence complex queries
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

            # 🔹 STEP 2: Enhanced retry trigger conditions
            vague_phrases = [
                "not explicitly mentioned", "not mentioned", "not clearly stated", "provided context",
                "not available", "not described", "not stated", "not defined", "focuses on",
                "cannot be determined", "unable to verify", "document does not specify",
                "information not available", "no mention", "not specified", "insufficient context"
            ]

            # Check for multiple retry conditions
            retry_triggers = []

            if any(p in answer.lower() for p in vague_phrases):
                retry_triggers.append("Vague answer")

            if result_1.get("confidence", "unknown") in {"low", "unknown"}:
                retry_triggers.append("Low confidence")

            if len(answer.strip()) < 30:  # Very short answers often incomplete
                retry_triggers.append("Too short")

            if answer.count('.') < 1 and '?' not in question.lower():  # Incomplete sentences
                retry_triggers.append("Incomplete sentence")

            # 🔹 STEP 3: Multi-level retry strategy
            if retry_triggers:
                retry_reason = " + ".join(retry_triggers)
                logger.warning(f"[ENHANCED_RETRY] Q{i + 1}: {retry_reason}")

                # RETRY LEVEL 1: Clause priority context
                context_2 = await retriever.get_clause_priority_context(question)
                result_2 = await analyze_with_adaptive_timing(question, context_2, True, "clause_retry")
                fallback_answer = result_2.get("answer", "")

                # Use Level 1 retry if it's better
                if (fallback_answer and
                        fallback_answer != answer and
                        not any(p in fallback_answer.lower() for p in vague_phrases) and
                        len(fallback_answer.strip()) > len(answer.strip())):
                    answer = fallback_answer
                    logger.info(f"[ENHANCED_RETRY] Q{i + 1}: Level 1 success (clause priority)")

                # RETRY LEVEL 2: If still problematic, try broader search
                elif any(p in answer.lower() for p in vague_phrases[:6]):  # Still vague
                    logger.warning(f"[ENHANCED_RETRY] Q{i + 1}: Level 1 failed, trying broader search")

                    # Create broader context with more chunks
                    broader_chunks = multi_strategy_search(question, retriever.chunks, top_k=25)
                    broader_context = ""
                    total_chars = 0

                    for j, chunk in enumerate(broader_chunks):
                        chunk_text = chunk['text'][:600]  # Smaller chunks but more of them
                        if total_chars + len(chunk_text) < 9500:  # Larger context limit
                            broader_context += f"[BROAD-{j + 1}] {chunk_text}\n\n"
                            total_chars += len(chunk_text)

                    # Try with broader context
                    result_3 = await analyze_with_adaptive_timing(question, broader_context, True, "broad_retry")
                    broad_answer = result_3.get("answer", "")

                    if (broad_answer and
                            not any(p in broad_answer.lower() for p in vague_phrases) and
                            len(broad_answer.strip()) > 25):
                        answer = broad_answer
                        logger.info(f"[ENHANCED_RETRY] Q{i + 1}: Level 2 success (broader search)")

                # RETRY LEVEL 3: Last resort - fuzzy word matching
                elif "unable to verify" in answer.lower() or len(answer.strip()) < 20:
                    logger.warning(f"[ENHANCED_RETRY] Q{i + 1}: Level 2 failed, trying fuzzy matching")

                    # Extract key terms from question
                    question_words = [w.lower() for w in question.split() if len(w) > 3]
                    fuzzy_chunks = []

                    for chunk in retriever.chunks:
                        chunk_text = chunk['text'].lower()
                        # Look for partial matches and related terms
                        matches = 0
                        for word in question_words:
                            if word in chunk_text:
                                matches += 2  # Exact match
                            elif word[:-1] in chunk_text or word + 's' in chunk_text:
                                matches += 1  # Fuzzy match

                        if matches > 0:
                            fuzzy_chunks.append((chunk, matches))

                    # Sort by relevance and take top chunks
                    fuzzy_chunks.sort(key=lambda x: x[1], reverse=True)
                    fuzzy_context = ""
                    total_chars = 0

                    for j, (chunk, score) in enumerate(fuzzy_chunks[:15]):
                        chunk_text = chunk['text'][:700]
                        if total_chars + len(chunk_text) < 8500:
                            fuzzy_context += f"[FUZZY-{j + 1}] {chunk_text}\n\n"
                            total_chars += len(chunk_text)

                    if fuzzy_context.strip():
                        result_4 = await analyze_with_adaptive_timing(question, fuzzy_context, True, "fuzzy_retry")
                        fuzzy_answer = result_4.get("answer", "")

                        if (fuzzy_answer and
                                "unable to verify" not in fuzzy_answer.lower() and
                                len(fuzzy_answer.strip()) > 20):
                            answer = fuzzy_answer
                            logger.info(f"[ENHANCED_RETRY] Q{i + 1}: Level 3 success (fuzzy matching)")

            # 🔹 STEP 4: Final answer validation and cleaning
            final_answer = clean_answer_formatting(answer)

            # Last resort cleanup - if still problematic, provide a more helpful response
            if any(p in final_answer.lower() for p in ["unable to verify", "not mentioned", "not available"]):
                # Try to extract at least partial information
                if len(context_1) > 1000:  # We had context, maybe partial info exists
                    final_answer = "The provided document sections do not contain specific information about this query, though related policy terms may exist in other sections."
                else:
                    final_answer = "Insufficient relevant content found in the available document sections to provide a specific answer."

            elapsed = time.time() - question_start
            tokens = result_1.get('tokens_used', 0)
            model_used = result_1.get('model_used', 'Unknown')
            confidence = result_1.get('confidence', 'unknown')
            complexity = result_1.get('complexity', 'simple')

            # Enhanced logging with retry info
            timing_emoji = "🕐" if elapsed > 2.0 else "⚡"
            retry_note = f" [RETRY: {retry_reason}]" if retry_triggers else ""
            logger.info(
                f"[Jay121305] Q{i + 1} ✓ {timing_emoji} | {elapsed:.1f}s | {tokens}t | {model_used} | {confidence} | {complexity}{retry_note}")

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
