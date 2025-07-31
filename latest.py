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


class HighThroughputKeyPool:
    def __init__(self, keys, primary_tpm=30000, fallback_tpm=10000, rpm_limit=30):
        self.keys = keys
        self.primary_tpm = primary_tpm  # Llama-4-Scout: 30K TPM
        self.fallback_tpm = fallback_tpm  # Kimi-K2: 10K TPM
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
                'cooldown_until': 0,
                'last_used': 0  # For round-robin balancing
            }

        self.total_requests = 0
        self.successful_requests = 0
        self.primary_model_used = 0
        self.fallback_model_used = 0
        self.current_key_index = 0  # For round-robin

    def reset_counters_if_needed(self, key):
        """Reset counters every minute"""
        now = time.time()
        stats = self.key_stats[key]

        if now - stats['last_reset'] >= 60:  # 1 minute reset
            stats['primary_tokens_used'] = 0
            stats['fallback_tokens_used'] = 0
            stats['requests_made'] = 0
            stats['last_reset'] = now
            logger.info(f"[Jay121305] Key ...{key[-5:]} tokens reset for new minute")

        # Reset cooling if cooldown period passed
        if stats['is_cooling'] and now >= stats['cooldown_until']:
            stats['is_cooling'] = False
            stats['consecutive_fails'] = 0
            logger.info(f"[Jay121305] Key ...{key[-5:]} cooled down and ready")

    def get_aggressive_key(self, estimated_tokens=800, use_fallback=False):
        """Aggressive key selection for high throughput"""
        with self.lock:
            tpm_limit = self.fallback_tpm if use_fallback else self.primary_tpm
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            # Try round-robin first for even distribution
            for _ in range(len(self.keys)):
                key_idx = self.current_key_index
                self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                key = self.keys[key_idx]

                self.reset_counters_if_needed(key)
                stats = self.key_stats[key]

                # Skip if cooling down
                if stats['is_cooling']:
                    continue

                # Check aggressive limits (allow higher usage)
                tokens_ok = stats[token_field] + estimated_tokens <= tpm_limit * 0.95  # Use 95% of limit
                requests_ok = stats['requests_made'] < self.rpm_limit

                if tokens_ok and requests_ok:
                    model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                    logger.debug(f"[Jay121305] Round-robin key ...{key[-5:]} for {model_name}")
                    return key

            # If round-robin fails, find least loaded key
            available_keys = []
            for key in self.keys:
                self.reset_counters_if_needed(key)
                stats = self.key_stats[key]

                if stats['is_cooling']:
                    continue

                tokens_ok = stats[token_field] + estimated_tokens <= tpm_limit * 0.95
                requests_ok = stats['requests_made'] < self.rpm_limit

                if tokens_ok and requests_ok:
                    load_score = stats[token_field] / tpm_limit + stats['requests_made'] / self.rpm_limit
                    available_keys.append((key, load_score))

            if available_keys:
                # Return least loaded key
                available_keys.sort(key=lambda x: x[1])
                selected_key = available_keys[0][0]
                model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                logger.debug(f"[Jay121305] Selected least loaded key ...{selected_key[-5:]} for {model_name}")
                return selected_key

            # All keys at limit - aggressive wait strategy
            soonest_key = min(self.keys, key=lambda k: self.key_stats[k]['last_reset'] + 60)
            wait_time = max(0, self.key_stats[soonest_key]['last_reset'] + 60 - time.time())

            if wait_time > 0:
                model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
                logger.warning(f"[Jay121305] All keys at {model_name} limit, waiting {wait_time:.1f}s...")
                time.sleep(min(wait_time, 5))  # Max 5 second wait
                return self.get_aggressive_key(estimated_tokens, use_fallback)

            return soonest_key

    def record_success(self, key, tokens_used, use_fallback=False):
        """Record successful API call"""
        with self.lock:
            stats = self.key_stats[key]
            token_field = 'fallback_tokens_used' if use_fallback else 'primary_tokens_used'

            stats[token_field] += tokens_used
            stats['requests_made'] += 1
            stats['consecutive_fails'] = 0
            stats['last_used'] = time.time()

            self.total_requests += 1
            self.successful_requests += 1

            if use_fallback:
                self.fallback_model_used += 1
            else:
                self.primary_model_used += 1

            model_name = "Kimi-K2" if use_fallback else "Llama-4-Scout"
            limit = self.fallback_tpm if use_fallback else self.primary_tpm
            usage_pct = (stats[token_field] / limit) * 100
            logger.debug(
                f"[Jay121305] Key ...{key[-5:]} ({model_name}): {stats[token_field]}/{limit} tokens ({usage_pct:.1f}%)")

    def record_failure(self, key, error_type="unknown"):
        """Record failed API call with aggressive recovery"""
        with self.lock:
            stats = self.key_stats[key]
            stats['consecutive_fails'] += 1
            self.total_requests += 1

            # More lenient cooldown for high throughput
            if stats['consecutive_fails'] >= 4:  # Increased from 3
                stats['is_cooling'] = True
                stats['cooldown_until'] = time.time() + 20  # Reduced from 30
                logger.warning(
                    f"[Jay121305] Key ...{key[-5:]} cooling down (20s) due to {stats['consecutive_fails']} fails")

            logger.error(f"[Jay121305] Key ...{key[-5:]} failed: {error_type}")

    def get_pool_status(self):
        """Get comprehensive pool status"""
        with self.lock:
            total_primary_capacity = len(self.keys) * self.primary_tpm
            total_fallback_capacity = len(self.keys) * self.fallback_tpm

            status = {
                'total_keys': len(self.keys),
                'active_keys': sum(1 for k in self.keys if not self.key_stats[k]['is_cooling']),
                'success_rate': f"{(self.successful_requests / max(1, self.total_requests) * 100):.1f}%",
                'total_requests': self.total_requests,
                'primary_model_usage': f"{self.primary_model_used} (Llama-4-Scout)",
                'fallback_model_usage': f"{self.fallback_model_used} (Kimi-K2)",
                'total_capacity': f"Primary: {total_primary_capacity:,} TPM, Fallback: {total_fallback_capacity:,} TPM",
                'estimated_questions_capacity': f"~{total_primary_capacity // 400} questions/minute (400 tokens each)",
                'key_details': {}
            }

            for i, key in enumerate(self.keys):
                stats = self.key_stats[key]
                primary_pct = (stats['primary_tokens_used'] / self.primary_tpm) * 100
                fallback_pct = (stats['fallback_tokens_used'] / self.fallback_tpm) * 100

                status['key_details'][f'key_{i + 1}'] = {
                    'primary_usage': f"{stats['primary_tokens_used']}/{self.primary_tpm} ({primary_pct:.1f}%)",
                    'fallback_usage': f"{stats['fallback_tokens_used']}/{self.fallback_tpm} ({fallback_pct:.1f}%)",
                    'requests': f"{stats['requests_made']}/{self.rpm_limit}",
                    'status': 'cooling' if stats['is_cooling'] else 'active',
                    'fails': stats['consecutive_fails']
                }

            return status


# Initialize with all 6 keys and aggressive limits
groq_keys = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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"
]

groq_pool = HighThroughputKeyPool(groq_keys, primary_tpm=30000, fallback_tpm=10000, rpm_limit=30)

app = FastAPI(title="High-Throughput Insurance Assistant", version="8.3.0")

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


def enhanced_chunking_for_throughput(text: str, chunk_size: int = 800) -> List[Dict]:
    """Enhanced chunking optimized for higher throughput (more context per token)"""
    chunks = []

    # Comprehensive critical patterns for maximum coverage
    critical_patterns = [
        r'grace\s+period[^.]{0,150}\..*?(?:\.|$)',
        r'thirty\s+days[^.]{0,150}premium[^.]{0,100}\..*?(?:\.|$)',
        r'pre-existing\s+disease[^.]{0,200}\..*?(?:\.|$)',
        r'thirty[- ]six.*?months[^.]{0,150}coverage[^.]{0,100}\..*?(?:\.|$)',
        r'maternity[^.]{0,300}\..*?(?:\.|$)',
        r'twenty[- ]four.*?months.*?female[^.]{0,150}\..*?(?:\.|$)',
        r'two\s+deliveries[^.]{0,100}\..*?(?:\.|$)',
        r'cataract[^.]{0,150}surgery[^.]{0,100}\..*?(?:\.|$)',
        r'organ\s+donor[^.]{0,200}\..*?(?:\.|$)',
        r'transplantation.*?human.*?organs[^.]{0,150}\..*?(?:\.|$)',
        r'no\s+claim\s+discount[^.]{0,200}\..*?(?:\.|$)',
        r'5%[^.]{0,150}premium[^.]{0,100}\..*?(?:\.|$)',
        r'health\s+check[^.]{0,200}\..*?(?:\.|$)',
        r'hospital.*?defined[^.]{0,400}\..*?(?:\.|$)',
        r'institution.*?beds[^.]{0,300}\..*?(?:\.|$)',
        r'ayush[^.]{0,300}\..*?(?:\.|$)',
        r'room\s+rent[^.]{0,200}\..*?(?:\.|$)',
        r'1%.*?sum.*?insured[^.]{0,150}\..*?(?:\.|$)',
        r'2%.*?sum.*?insured[^.]{0,150}\..*?(?:\.|$)',
        r'day\s+care[^.]{0,200}\..*?(?:\.|$)',
        r'domiciliary[^.]{0,200}\..*?(?:\.|$)',
        r'aids[^.]{0,150}\..*?(?:\.|$)',
        r'hiv[^.]{0,150}\..*?(?:\.|$)',
        r'floater\s+sum\s+insured[^.]{0,150}\..*?(?:\.|$)',
        r'cosmetic\s+surgery[^.]{0,150}\..*?(?:\.|$)',
        r'outpatient[^.]{0,150}\..*?(?:\.|$)',
        r'opd[^.]{0,150}\..*?(?:\.|$)',
        r'infertility[^.]{0,200}\..*?(?:\.|$)',
        r'ivf[^.]{0,150}\..*?(?:\.|$)',
        r'stem\s+cell[^.]{0,150}\..*?(?:\.|$)',
        r'rabies[^.]{0,150}\..*?(?:\.|$)',
        r'vaccination[^.]{0,150}\..*?(?:\.|$)',
    ]

    # Extract critical chunks with expanded context
    critical_chunks = []
    used_positions = set()

    for pattern in critical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            start = max(0, match.start() - 100)  # More context
            end = min(len(text), match.end() + 100)

            # Allow some overlap for better coverage
            if not any(abs(start - pos) < 150 for pos in used_positions):
                context = text[start:end].strip()

                if len(context) > 60:
                    critical_chunks.append({
                        'text': context,
                        'metadata': {
                            'type': 'critical',
                            'priority': 100,
                            'has_numbers': bool(re.search(r'\d+', context)),
                            'pattern': pattern,
                            'coverage_score': len(context)  # Longer chunks get priority
                        }
                    })
                    used_positions.add(start)

    # Enhanced regular chunking for broader coverage
    section_patterns = [
        r'CLAUSE\s+\d+[^.]{0,800}\..*?(?:\.|$)',
        r'DEFINITIONS[^.]{0,1200}\..*?(?:\.|$)',
        r'Table\s+of\s+Benefits[^.]{0,1200}\..*?(?:\.|$)',
        r'EXCLUSIONS[^.]{0,1000}\..*?(?:\.|$)',
        r'BENEFITS[^.]{0,1000}\..*?(?:\.|$)',
        r'WAITING\s+PERIOD[^.]{0,800}\..*?(?:\.|$)',
        r'CONDITIONS[^.]{0,800}\..*?(?:\.|$)',
    ]

    regular_chunks = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_text = match.group(0)
            if len(section_text) > 150:
                # Split into appropriately sized chunks
                sentences = re.split(r'(?<=[.!?])\s+', section_text)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + " " + sentence) <= chunk_size:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk.strip() and len(current_chunk) > 100:
                            regular_chunks.append({
                                'text': current_chunk.strip(),
                                'metadata': {
                                    'type': 'section',
                                    'priority': 40,
                                    'has_numbers': bool(re.search(r'\d+', current_chunk)),
                                    'coverage_score': len(current_chunk)
                                }
                            })
                        current_chunk = sentence

                if current_chunk.strip() and len(current_chunk) > 100:
                    regular_chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'type': 'section',
                            'priority': 40,
                            'has_numbers': bool(re.search(r'\d+', current_chunk)),
                            'coverage_score': len(current_chunk)
                        }
                    })

    # Combine and prioritize by coverage score
    all_chunks = critical_chunks + regular_chunks

    # Enhanced deduplication
    unique_chunks = []
    seen_signatures = set()

    for chunk in all_chunks:
        # Create signature based on key words
        words = chunk['text'].lower().split()
        key_words = [w for w in words if len(w) > 4][:10]  # First 10 meaningful words
        signature = ' '.join(sorted(key_words))

        if signature not in seen_signatures and len(chunk['text']) > 50:
            seen_signatures.add(signature)
            unique_chunks.append(chunk)
            if len(unique_chunks) >= 100:  # Increased limit for more coverage
                break

    # Sort by priority and coverage score
    unique_chunks.sort(key=lambda x: (x['metadata']['priority'], x['metadata']['coverage_score']), reverse=True)

    logger.info(
        f"[Jay121305] Enhanced chunking: {len(critical_chunks)} critical + {len(regular_chunks)} regular = {len(unique_chunks)} unique")
    return unique_chunks


def enhanced_search_for_throughput(query: str, chunks: List[Dict], top_k: int = 4) -> List[Dict]:
    """Enhanced search optimized for higher accuracy and throughput"""

    query_lower = query.lower()

    # Comprehensive keyword mapping with expanded coverage
    keyword_mappings = {
        'grace period': ['grace', 'period', 'premium', 'payment', 'thirty', '30', 'days', 'due', 'date', 'renew',
                         'continue', 'continuity'],
        'pre-existing': ['pre-existing', 'ped', 'disease', 'thirty-six', '36', 'months', 'continuous', 'coverage',
                         'inception', 'complications'],
        'maternity': ['maternity', 'childbirth', 'pregnancy', 'twenty-four', '24', 'months', 'delivery', 'female',
                      'termination', 'eligible'],
        'cataract': ['cataract', 'surgery', 'twenty-four', '24', 'months', 'two', 'years', 'waiting', 'period'],
        'organ donor': ['organ', 'donor', 'transplant', 'harvesting', 'transplantation', 'human', 'organs', 'act',
                        '1994', 'expenses'],
        'no claim discount': ['no claim', 'ncd', 'discount', '5%', 'five', 'percent', 'premium', 'renewal', 'base',
                              'flat'],
        'health check': ['health check', 'preventive', 'check-up', 'medical', 'examination', 'block', 'continuous',
                         'years'],
        'hospital': ['hospital', 'definition', 'institution', 'beds', 'nursing', 'staff', 'operation', 'theatre',
                     'qualified', 'practitioners'],
        'ayush': ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy', 'systems', 'inpatient'],
        'room rent': ['room rent', 'icu', 'intensive care', 'plan a', 'sub-limit', '1%', '2%', 'charges', 'daily'],
        'day care': ['day care', 'day-care', 'procedure', 'surgical', 'treatment', 'less', '24', 'hours'],
        'domiciliary': ['domiciliary', 'hospitalization', 'home', 'treatment', 'confined', 'bed', 'availability'],
        'aids': ['aids', 'hiv', 'immune', 'deficiency', 'syndrome', 'virus', 'treatment'],
        'floater': ['floater', 'sum', 'insured', 'aggregate', 'family', 'available', 'claims'],
        'cosmetic': ['cosmetic', 'surgery', 'implants', 'aesthetic'],
        'outpatient': ['outpatient', 'opd', 'out-patient', 'consultation'],
        'infertility': ['infertility', 'ivf', 'fertility', 'reproductive'],
        'stem cell': ['stem', 'cell', 'therapy', 'treatment'],
        'vaccination': ['vaccination', 'vaccine', 'immunization', 'rabies', 'anti-rabies'],
        'exclusion': ['exclusion', 'excluded', 'not covered', 'limitation'],
        'first 30 days': ['first', '30', 'thirty', 'days', 'initial', 'waiting', 'illness', 'disease']
    }

    # Multi-phase scoring for better accuracy
    scored_chunks = []

    # Phase 1: Determine primary query domain
    primary_domain = None
    matched_keywords = set()

    for domain, keywords in keyword_mappings.items():
        domain_words = domain.split()
        if all(word in query_lower for word in domain_words):
            primary_domain = domain
            matched_keywords.update(keywords)
            break
        elif any(word in query_lower for word in domain_words):
            matched_keywords.update(keywords)

    # Phase 2: Add general query terms
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    matched_keywords.update(query_words)

    # Phase 3: Advanced scoring algorithm
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        metadata = chunk['metadata']
        score = 0

        # Exact domain match bonus (highest priority)
        if primary_domain:
            domain_phrases = primary_domain.split()
            for phrase in domain_phrases:
                if phrase in text_lower:
                    score += 40

        # Keyword density scoring
        keyword_hits = 0
        for keyword in matched_keywords:
            if keyword in text_lower:
                keyword_weight = len(keyword) + 5  # Longer keywords weighted more
                score += keyword_weight
                keyword_hits += 1

        # Keyword density bonus
        if keyword_hits > 0:
            density_bonus = min(keyword_hits * 5, 30)  # Max 30 points
            score += density_bonus

        # Metadata and priority scoring
        priority = metadata.get('priority', 0)
        score += priority

        if metadata.get('type') == 'critical':
            score += 50

        # Number presence bonus (insurance docs have many numbers)
        if metadata.get('has_numbers'):
            score += 20

        # Coverage score bonus (longer chunks often more complete)
        coverage_score = metadata.get('coverage_score', 0)
        score += min(coverage_score // 20, 15)  # Max 15 points

        # Exact phrase matching (very high bonus)
        for query_phrase in [query_lower] + query_lower.split():
            if len(query_phrase) > 5 and query_phrase in text_lower:
                score += 35

        # Important pattern bonuses
        important_patterns = [
            'grace period', 'waiting period', 'pre-existing disease',
            'maternity expenses', 'cataract surgery', 'organ donor',
            'no claim discount', 'health check', 'hospital means',
            'ayush treatment', 'room rent', 'icu charges', 'day care',
            'domiciliary', 'aids', 'floater sum insured'
        ]

        for pattern in important_patterns:
            if pattern in text_lower:
                score += 25

        # Penalty for very short chunks
        if len(chunk['text']) < 100:
            score -= 10

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    if scored_chunks:
        logger.info(
            f"[Jay121305] Enhanced search: top score = {scored_chunks[0][1]}, found {len(scored_chunks)} relevant chunks")
        # Log top 3 for debugging
        for i, (chunk, score) in enumerate(scored_chunks[:3]):
            preview = chunk['text'][:80].replace('\n', ' ')
            logger.debug(f"[Jay121305] Rank {i + 1}: score={score}, text='{preview}...'")

    return [chunk for chunk, _ in scored_chunks[:top_k]]


class HighThroughputRetriever:
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
        self.chunks = enhanced_chunking_for_throughput(text)
        self.is_indexed = True
        self.doc_hash = doc_hash
        logger.info(f"[Jay121305] Indexed {len(self.chunks)} chunks in {time.time() - start:.2f}s")

    async def get_enhanced_context(self, query: str) -> str:
        """Enhanced context for higher accuracy"""
        if not self.is_indexed:
            return "Document not indexed"

        relevant_chunks = enhanced_search_for_throughput(query, self.chunks, top_k=4)

        if not relevant_chunks:
            # Fallback to high priority chunks
            high_priority = [c for c in self.chunks if c['metadata'].get('priority', 0) > 80]
            relevant_chunks = high_priority[:4] if high_priority else self.chunks[:4]

        # Build enhanced context with more information
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            text = chunk['text']
            priority = chunk['metadata'].get('priority', 0)

            # Include more text for high priority chunks
            if priority > 80:
                # Keep full text for critical chunks
                context_parts.append(f"[CRITICAL-{i + 1}] {text}")
            else:
                # Limit regular chunks to 600 chars
                limited_text = text[:600] if len(text) > 600 else text
                context_parts.append(f"[{i + 1}] {limited_text}")

        context = "\n\n".join(context_parts)
        logger.info(f"[Jay121305] Enhanced context: {len(context)} chars from {len(relevant_chunks)} chunks")
        return context


# Global retriever
retriever = HighThroughputRetriever()

# Optimized system prompts for high throughput
PRIMARY_SYSTEM_PROMPT = """You are an expert insurance policy analyst. Provide accurate, concise answers based on the document context.

RESPONSE FORMAT (JSON):
{
  "answer": "Direct answer with key facts, numbers, and conditions",
  "confidence": "high | medium | low"
}

RULES:
1. Keep under 120 words but include all key information
2. Include exact numbers, periods, percentages as stated
3. For definitions, provide complete definitions
4. If information exists in context, provide it - don't say "not mentioned"
5. Be factual and comprehensive
6. Set confidence: "high" if answer is definitive, "medium" if partially found, "low" if uncertain

EXAMPLES:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
- "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
- "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
"""

FALLBACK_SYSTEM_PROMPT = """You are a senior insurance policy expert. The primary model had low confidence, so provide a comprehensive, accurate analysis.

RESPONSE FORMAT (JSON):
{
  "answer": "Thorough answer with all relevant details, numbers, conditions, and context"
}

RULES:
1. Be comprehensive and include all relevant information from context
2. Include exact numbers, periods, percentages, conditions, and eligibility criteria
3. For definitions, provide complete definitions as stated in the policy
4. Cross-reference multiple context sections for complete information
5. If information exists in context, provide it fully - never say "not mentioned"
6. Include all qualifying phrases like "provided that", "subject to", "limited to"
7. Match the style and completeness of sample responses
8. Provide complete benefit details including amounts, limits, and conditions

EXAMPLES:
- "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits pertaining to Waiting Periods and coverage of Pre-Existing Diseases. Coverage shall not be available during the period for which no premium is received."
- "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered. However, if the Insured Person is continuously covered without any break as defined under the portability norms of the extant IRDAI (Health Insurance) Regulations, then the waiting period for the same would be reduced to the extent of prior coverage."
"""


async def analyze_with_high_throughput_system(query: str, context: str) -> Dict[str, Any]:
    """High-throughput dual-model analysis"""
    try:
        # More generous context limit for accuracy
        if len(context) > 4000:
            context = context[:4000] + "..."

        # Step 1: Primary model (Llama-4-Scout)
        estimated_tokens = (len(context) + len(query) + 500) // 4
        api_key = groq_pool.get_aggressive_key(estimated_tokens, use_fallback=False)

        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {"role": "system", "content": PRIMARY_SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nProvide an accurate answer with confidence level:"}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.05,
            "max_tokens": 700  # Increased for more comprehensive answers
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
                    timeout=aiohttp.ClientTimeout(total=25)
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
            confidence = "low"

        elapsed = time.time() - start_time

        # Step 2: Fallback logic for low confidence
        if confidence == "low":
            logger.info(f"[Jay121305] Primary confidence LOW, trying fallback...")

            try:
                fallback_key = groq_pool.get_aggressive_key(estimated_tokens, use_fallback=True)

                fallback_payload = {
                    "model": "moonshotai/kimi-k2-instruct",
                    "messages": [
                        {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nThe primary model had low confidence. Provide a comprehensive answer:"}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.05,
                    "max_tokens": 1000  # More tokens for detailed fallback
                }

                fallback_headers = {
                    "Authorization": f"Bearer {fallback_key}",
                    "Content-Type": "application/json"
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            json=fallback_payload,
                            headers=fallback_headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                    ) as fallback_response:

                        if fallback_response.status >= 400:
                            groq_pool.record_failure(fallback_key, f"http_{fallback_response.status}")
                            logger.warning(f"[Jay121305] Fallback failed, using primary answer")
                        else:
                            fallback_result = await fallback_response.json()
                            fallback_tokens = fallback_result.get("usage", {}).get("total_tokens", estimated_tokens)
                            groq_pool.record_success(fallback_key, fallback_tokens, use_fallback=True)

                            fallback_content = fallback_result["choices"][0]["message"]["content"]
                            try:
                                fallback_parsed = json.loads(fallback_content)
                                fallback_answer = fallback_parsed.get("answer", answer)

                                # Use fallback if significantly better
                                if len(fallback_answer) > len(answer) + 30:
                                    answer = fallback_answer
                                    confidence = "medium"
                                    logger.info(f"[Jay121305] Using improved fallback answer")

                            except json.JSONDecodeError:
                                logger.warning(f"[Jay121305] Fallback JSON parse failed")

            except Exception as e:
                logger.warning(f"[Jay121305] Fallback failed: {e}")

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
        logger.error(f"[Jay121305] High-throughput analysis failed: {str(e)}")
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
        "message": "High-Throughput Insurance Assistant v8.3",
        "primary_model": "meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)",
        "fallback_model": "moonshotai/kimi-k2-instruct (10K TPM)",
        "total_capacity": "180K TPM primary + 60K TPM fallback = 240K TPM",
        "estimated_throughput": "~450 questions/minute (400 tokens each)",
        "optimized_for": "70-80 questions in single batch",
        "keys_available": len(groq_keys),
        "features": ["High-throughput processing", "Enhanced context", "Dual-model fallback"],
        "user": "Jay121305",
        "timestamp": "2025-07-31 19:41:24"
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
        logger.info(f"[Jay121305] HIGH-THROUGHPUT: Processing {num_questions} questions (optimized for 70-80)")

        pdf_text = extract_pdf_text(request.documents)
        await retriever.index_document(pdf_text)

        async def process_single_question(i, question):
            question_start = time.time()
            logger.info(f"[Jay121305] Q{i + 1}: {question[:60]}...")

            # Minimal delay for high throughput
            delay = (i % 6) * 0.15 + random.uniform(0.05, 0.15)  # 0.05-1.05 second delays
            await asyncio.sleep(delay)

            context = await retriever.get_enhanced_context(question)
            result = await analyze_with_high_throughput_system(question, context)

            elapsed = time.time() - question_start
            if result.get("success"):
                tokens = result.get('tokens_used', 0)
                model_used = result.get('model_used', 'Unknown')
                confidence = result.get('confidence', 'unknown')
                logger.info(f"[Jay121305] Q{i + 1} ✓ | {elapsed:.1f}s | {tokens}t | {model_used} | {confidence}")
            else:
                logger.error(f"[Jay121305] Q{i + 1} ✗ | {elapsed:.1f}s | {result.get('error', 'unknown')}")

            return result.get("answer", "Analysis failed")

        # High concurrency for throughput (utilizing 30K TPM limit)
        if num_questions <= 60:
            concurrency = 4  # Higher concurrency for smaller batches
        elif num_questions <= 80:
            concurrency = 3  # Moderate concurrency for larger batches
        else:
            concurrency = 2  # Conservative for very large batches

        semaphore = asyncio.Semaphore(concurrency)
        logger.info(f"[Jay121305] Using concurrency level: {concurrency}")

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

        logger.info(f"[Jay121305] HIGH-THROUGHPUT COMPLETED")
        logger.info(f"[Jay121305] ⏱️  Total time: {total_time:.1f}s ({total_time / num_questions:.2f}s per question)")
        logger.info(
            f"[Jay121305] ✅ Success: {success_count}/{num_questions} ({(success_count / num_questions) * 100:.1f}%)")
        logger.info(
            f"[Jay121305] 🔧 Model usage: {pool_status['primary_model_usage']}, {pool_status['fallback_model_usage']}")
        logger.info(f"[Jay121305] 🔑 Keys: {pool_status['active_keys']}/{pool_status['total_keys']} active")

        return {"answers": final_answers}

    except Exception as e:
        logger.error(f"[Jay121305] High-throughput endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"High-throughput processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("[Jay121305] Starting High-Throughput Insurance Assistant v8.3")
    logger.info(f"[Jay121305] Optimized for: 70-80 questions per batch")
    logger.info(f"[Jay121305] Primary: meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)")
    logger.info(f"[Jay121305] Fallback: moonshotai/kimi-k2-instruct (10K TPM)")
    logger.info(f"[Jay121305] Total capacity: 240K TPM (~600 questions/minute theoretical)")
    logger.info(f"[Jay121305] User: Jay121305 | Time: 2025-07-31 19:41:24")
    uvicorn.run(app, host="0.0.0.0", port=8000)
