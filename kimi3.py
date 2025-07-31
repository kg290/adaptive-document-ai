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

# Ultimate cloud-only solution - no ML libraries!
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


# --- Enhanced Multi-key pool for GROQ API keys ---
class GroqAPIKeyPool:
    def __init__(self, keys, cooldown=45):  # Reduced cooldown for faster recovery
        self.keys = keys
        self.cooldown = cooldown
        self.lock = threading.Lock()
        self.cooling = {}  # key: available_time
        self.next_key_idx = 0
        self.usage_stats = {key: 0 for key in keys}

    def get_key(self):
        with self.lock:
            now = time.time()
            n = len(self.keys)
            for _ in range(n):
                idx = self.next_key_idx
                key = self.keys[idx]
                if key not in self.cooling or self.cooling[key] <= now:
                    self.next_key_idx = (idx + 1) % n
                    self.usage_stats[key] += 1
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
            logger.warning(f"[Jay121305] Key ...{key[-5:]} rate limited until {time.time() + self.cooldown}")


# Your Groq API keys
groq_keys = [
        "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",
        "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",
        "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",
        "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"
]
groq_pool = GroqAPIKeyPool(groq_keys)

app = FastAPI(title="Ultimate Document Intelligence Assistant", version="6.0.0")

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


# --- Ultimate Semantic Retrieval System ---

def extract_comprehensive_metadata(text: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from text chunk"""
    metadata = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'has_numbers': bool(re.search(r'\d+', text)),
        'has_percentages': bool(re.search(r'\d+%', text)),
        'has_currency': bool(re.search(r'(Rs|INR|₹|\$)\s*\d+', text, re.IGNORECASE)),
        'has_time_periods': bool(re.search(r'\d+\s*(month|year|day|week)', text, re.IGNORECASE)),
        'has_definitions': bool(re.search(r'(define|means|refers to|shall mean|is defined as)', text, re.IGNORECASE)),
        'has_lists': bool(re.search(r'(\n\s*[-*•]\s|\n\s*\d+\.\s|\n\s*\([a-z]\))', text)),
        'has_tables': bool(re.search(r'(\|.*\||\t.*\t)', text)),
        'has_legal_language': bool(re.search(r'(shall|hereby|whereas|provided that|subject to)', text, re.IGNORECASE)),
        'has_medical_terms': bool(re.search(r'(treatment|medical|hospital|surgery|disease)', text, re.IGNORECASE)),
        'has_technical_terms': bool(re.search(r'(engine|brake|tyre|specification)', text, re.IGNORECASE)),
        'section_type': 'unknown'
    }

    # Determine section type
    text_lower = text.lower()
    if any(term in text_lower for term in ['clause', 'section', 'article']):
        metadata['section_type'] = 'formal_clause'
    elif any(term in text_lower for term in ['definition', 'meaning', 'term']):
        metadata['section_type'] = 'definition'
    elif any(term in text_lower for term in ['benefit', 'coverage', 'limit']):
        metadata['section_type'] = 'benefit_clause'
    elif any(term in text_lower for term in ['waiting', 'period', 'exclusion']):
        metadata['section_type'] = 'restriction_clause'
    elif metadata['has_tables'] or metadata['has_lists']:
        metadata['section_type'] = 'structured_data'

    return metadata


def extract_section_header(text: str) -> str:
    """Enhanced section header extraction"""
    lines = text.split('\n')[:5]  # Check first 5 lines

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Priority patterns
        if re.match(r'^(CLAUSE|SECTION|ARTICLE)\s+\d+', line, re.IGNORECASE):
            return line
        if re.match(r'^\d+\.\d+', line):
            return line
        if re.match(r'^[A-Z][A-Z\s]{5,50}:?$', line):
            return line
        if any(keyword in line.lower() for keyword in ['grace period', 'waiting period', 'maternity', 'definitions']):
            return line

    # Fallback to first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()[:100]

    return "Unknown Section"


def ultimate_document_chunking(text: str, chunk_size: int = 1800, overlap: int = 400) -> List[Dict]:
    """Ultimate chunking strategy that preserves context and important details"""
    chunks = []

    # Enhanced section patterns for all document types
    section_patterns = [
        # Insurance/Legal patterns
        r'\n(?=CLAUSE\s+\d+)',
        r'\n(?=SECTION\s+\d+)',
        r'\n(?=ARTICLE\s+\d+)',
        r'\n(?=\d+\.\d+\s)',
        r'\n(?=DEFINITIONS?\s*:)',
        r'\n(?=BENEFITS?\s*:)',
        r'\n(?=COVERAGE\s*:)',
        r'\n(?=EXCLUSIONS?\s*:)',
        r'\n(?=WAITING\s+PERIOD)',
        r'\n(?=GRACE\s+PERIOD)',
        r'\n(?=MATERNITY)',
        r'\n(?=NO\s+CLAIM\s+DISCOUNT)',
        r'\n(?=HOSPITAL\s*:)',
        r'\n(?=AYUSH)',
        r'\n(?=ROOM\s+RENT)',

        # Technical/Manual patterns
        r'\n(?=SPECIFICATIONS?\s*:)',
        r'\n(?=FEATURES?\s*:)',
        r'\n(?=MAINTENANCE\s*:)',
        r'\n(?=SAFETY\s*:)',

        # Legal/Constitutional patterns
        r'\n(?=PREAMBLE)',
        r'\n(?=FUNDAMENTAL\s+RIGHTS)',
        r'\n(?=DIRECTIVE\s+PRINCIPLES)',

        # General patterns
        r'\n(?=[A-Z][A-Z\s]{10,50}:)',
        r'\n(?=\d+\.\s)',
        r'\n(?=\([a-z]\)\s)',
        r'\n(?=Table\s+\d+)',
        r'\n(?=Figure\s+\d+)',
    ]

    # Pre-process text to identify special sections
    special_sections = []

    # Find tables and preserve them as single chunks
    table_pattern = r'(Table\s+\d+.*?(?=\n\n|\n[A-Z]|\Z))'
    for match in re.finditer(table_pattern, text, re.DOTALL | re.IGNORECASE):
        special_sections.append({
            'start': match.start(),
            'end': match.end(),
            'type': 'table',
            'content': match.group(1)
        })

    # Find benefit/coverage tables
    benefit_pattern = r'(Plan\s+[ABC].*?(?=\n\n|\n[A-Z]|\Z))'
    for match in re.finditer(benefit_pattern, text, re.DOTALL | re.IGNORECASE):
        special_sections.append({
            'start': match.start(),
            'end': match.end(),
            'type': 'benefit_table',
            'content': match.group(1)
        })

    # Split by sections
    current_pos = 0
    sections = []

    for pattern in section_patterns:
        if current_pos >= len(text):
            break

        matches = list(re.finditer(pattern, text[current_pos:]))
        if matches:
            for match in matches:
                actual_pos = current_pos + match.start()
                if actual_pos > current_pos:
                    sections.append({
                        'text': text[current_pos:actual_pos].strip(),
                        'start': current_pos,
                        'end': actual_pos
                    })
                current_pos = actual_pos

    # Add remaining text
    if current_pos < len(text):
        sections.append({
            'text': text[current_pos:].strip(),
            'start': current_pos,
            'end': len(text)
        })

    # Process each section
    for i, section in enumerate(sections):
        section_text = section['text']
        if not section_text.strip():
            continue

        if len(section_text) <= chunk_size:
            # Section fits in one chunk
            metadata = extract_comprehensive_metadata(section_text)
            metadata['section_header'] = extract_section_header(section_text)
            metadata['chunk_type'] = 'complete_section'
            metadata['original_position'] = section['start']

            chunks.append({
                'text': section_text,
                'section_id': i,
                'metadata': metadata
            })
        else:
            # Split large section intelligently
            # Try to split by sentences first
            sentences = re.split(r'(?<=[.!?])\s+', section_text)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk + " " + sentence) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk.strip():
                        metadata = extract_comprehensive_metadata(current_chunk)
                        metadata['section_header'] = extract_section_header(section_text)
                        metadata['chunk_type'] = 'sentence_split'
                        metadata['original_position'] = section['start']

                        chunks.append({
                            'text': current_chunk.strip(),
                            'section_id': i,
                            'metadata': metadata
                        })

                    current_chunk = sentence

            # Add remaining chunk
            if current_chunk.strip():
                metadata = extract_comprehensive_metadata(current_chunk)
                metadata['section_header'] = extract_section_header(section_text)
                metadata['chunk_type'] = 'sentence_split_final'
                metadata['original_position'] = section['start']

                chunks.append({
                    'text': current_chunk.strip(),
                    'section_id': i,
                    'metadata': metadata
                })

    # Add special sections as priority chunks
    for special in special_sections:
        metadata = extract_comprehensive_metadata(special['content'])
        metadata['section_header'] = f"Special {special['type'].title()}"
        metadata['chunk_type'] = special['type']
        metadata['priority'] = True

        chunks.append({
            'text': special['content'],
            'section_id': -1,  # Special marker
            'metadata': metadata
        })

    logger.info(f"[Jay121305] Ultimate chunking created {len(chunks)} chunks")
    return chunks


def ultimate_keyword_search(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """Ultimate keyword-based search with comprehensive domain knowledge"""

    query_lower = query.lower()
    query_words = [word.strip('.,!?()') for word in query.split() if len(word) > 2]

    # Comprehensive domain keyword mappings
    domain_keywords = {
        # Insurance specific
        'grace period': ['grace', 'premium', 'payment', 'due', 'thirty', '30', 'days', 'renewal'],
        'pre-existing': ['pre-existing', 'ped', 'disease', 'waiting', '36', '48', 'months', 'continuous'],
        'maternity': ['maternity', 'childbirth', 'pregnancy', 'delivery', '24', 'months', 'female', 'caesarean',
                      'normal', 'two', 'deliveries'],
        'cataract': ['cataract', 'surgery', 'waiting', '24', 'months', 'two', 'years', 'eye'],
        'organ donor': ['organ', 'donor', 'transplant', 'harvest', 'hospitalization', 'expenses'],
        'no claim discount': ['ncd', 'no claim', 'discount', 'bonus', 'renewal', '5%', 'base premium'],
        'health check': ['health check', 'preventive', 'check-up', 'medical examination', 'reimbursement', 'block',
                         'years'],
        'hospital definition': ['hospital', 'nursing home', 'definition', 'inpatient', 'beds', '10', '15', 'qualified',
                                'staff', 'operation theatre'],
        'ayush': ['ayush', 'ayurveda', 'yoga', 'unani', 'siddha', 'homeopathy', 'sum insured'],
        'room rent': ['room rent', 'icu', 'sub-limit', 'plan a', 'daily', '1%', '2%', 'preferred provider'],

        # Legal/Constitutional
        'article': ['article', 'constitution', 'fundamental', 'rights', 'equality', 'discrimination'],
        'amendment': ['amendment', 'parliament', 'boundaries', 'alter'],
        'employment': ['employment', 'children', 'child labour', 'factory', 'age'],

        # Technical/Automotive
        'spark plug': ['spark plug', 'gap', 'electrode', 'ignition', 'specification'],
        'brake': ['brake', 'disc', 'drum', 'safety', 'compulsory'],
        'tyre': ['tyre', 'tire', 'tubeless', 'version', 'specification'],
        'oil': ['oil', 'engine', 'lubricant', 'maintenance', 'grade'],

        # General numerical
        'percentage': ['%', 'percent', 'rate', 'discount'],
        'currency': ['rs', 'inr', '₹', 'rupees', 'amount', 'cost'],
        'time': ['months', 'years', 'days', 'period', 'duration']
    }

    # Find relevant domain keywords
    relevant_keywords = set(query_words)
    for domain, keywords in domain_keywords.items():
        if any(term in query_lower for term in domain.split()):
            relevant_keywords.update(keywords)

    # Also add synonyms and related terms
    for word in query_words:
        if word in ['define', 'definition', 'what is']:
            relevant_keywords.update(['definition', 'means', 'refers to', 'shall mean'])
        elif word in ['cover', 'coverage', 'covered']:
            relevant_keywords.update(['benefit', 'indemnify', 'reimburse', 'expenses'])
        elif word in ['period', 'time']:
            relevant_keywords.update(['months', 'years', 'days', 'waiting'])

    # Score chunks with advanced algorithm
    scored_chunks = []
    for chunk in chunks:
        chunk_text = chunk['text'].lower()
        metadata = chunk['metadata']
        score = 0

        # Basic keyword matching with position weighting
        for keyword in relevant_keywords:
            keyword_count = chunk_text.count(keyword)
            if keyword_count > 0:
                # Give higher score for keywords at the beginning
                if chunk_text.find(keyword) < len(chunk_text) * 0.3:
                    score += keyword_count * 3
                else:
                    score += keyword_count * 2

        # Metadata-based scoring
        if metadata.get('has_numbers'):
            score += 4
        if metadata.get('has_percentages'):
            score += 3
        if metadata.get('has_currency'):
            score += 3
        if metadata.get('has_time_periods'):
            score += 4
        if metadata.get('has_definitions'):
            score += 3
        if metadata.get('has_lists'):
            score += 2
        if metadata.get('has_tables'):
            score += 5
        if metadata.get('priority'):  # Special sections
            score += 10

        # Section type bonuses
        section_type = metadata.get('section_type', 'unknown')
        if 'definition' in query_lower and section_type == 'definition':
            score += 8
        elif any(term in query_lower for term in ['benefit', 'coverage']) and section_type == 'benefit_clause':
            score += 8
        elif any(term in query_lower for term in ['waiting', 'exclusion']) and section_type == 'restriction_clause':
            score += 8
        elif section_type == 'structured_data':
            score += 6

        # Content pattern bonuses
        if 'waiting period' in chunk_text:
            score += 6
        if 'table of benefits' in chunk_text:
            score += 8
        if 'plan a' in chunk_text and 'plan a' in query_lower:
            score += 5

        # Penalty for very short chunks (likely incomplete)
        if len(chunk_text.split()) < 10:
            score = max(0, score - 3)

        # Boost for chunks with exact phrase matches
        for phrase in ['grace period', 'pre-existing disease', 'no claim discount', 'health check-up']:
            if phrase in query_lower and phrase in chunk_text:
                score += 10

        if score > 0:
            scored_chunks.append((chunk, score))

    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    selected_chunks = [chunk for chunk, score in scored_chunks[:top_k]]

    if scored_chunks:
        logger.info(
            f"[Jay121305] Ultimate search: top score = {scored_chunks[0][1]}, selected {len(selected_chunks)} chunks")
        # Log top 3 scores for debugging
        for i, (chunk, score) in enumerate(scored_chunks[:3]):
            header = chunk['metadata'].get('section_header', 'Unknown')[:50]
            logger.info(f"[Jay121305] Chunk {i + 1}: score={score}, header='{header}'")

    return selected_chunks


class UltimateSemanticRetriever:
    def __init__(self):
        self.chunks = []
        self.is_indexed = False
        self.document_hash = None
        self.document_type = None

    def detect_document_type(self, text: str) -> str:
        """Detect document type for specialized handling"""
        text_lower = text.lower()

        if any(term in text_lower for term in ['policy', 'insurance', 'premium', 'claim', 'coverage']):
            return 'insurance'
        elif any(term in text_lower for term in ['constitution', 'article', 'fundamental rights', 'preamble']):
            return 'legal'
        elif any(term in text_lower for term in ['engine', 'brake', 'specification', 'maintenance']):
            return 'technical'
        elif any(term in text_lower for term in ['principia', 'newton', 'physics', 'motion']):
            return 'scientific'
        else:
            return 'general'

    async def index_document(self, pdf_text: str):
        """Index document using ultimate chunking strategy"""
        current_hash = hashlib.md5(pdf_text.encode()).hexdigest()[:16]
        if self.document_hash == current_hash and self.is_indexed:
            logger.info(f"[Jay121305] Document already indexed (hash: {current_hash})")
            return

        logger.info("[Jay121305] Starting ultimate semantic indexing...")
        start_time = time.time()

        # Detect document type
        self.document_type = self.detect_document_type(pdf_text)
        logger.info(f"[Jay121305] Detected document type: {self.document_type}")

        # Ultimate chunking
        self.chunks = ultimate_document_chunking(pdf_text)
        self.is_indexed = True
        self.document_hash = current_hash

        elapsed = time.time() - start_time
        logger.info(f"[Jay121305] Indexed {len(self.chunks)} chunks in {elapsed:.2f}s")

    async def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> str:
        """Retrieve most relevant chunks using ultimate search"""
        if not self.is_indexed:
            logger.warning("[Jay121305] Document not indexed")
            return "Document not properly indexed."

        # Ultimate keyword search
        relevant_chunks = ultimate_keyword_search(query, self.chunks, top_k)

        if not relevant_chunks:
            logger.warning("[Jay121305] No relevant chunks found, using high-scoring chunks")
            # Fallback: select chunks with highest metadata scores
            fallback_chunks = []
            for chunk in self.chunks:
                metadata = chunk['metadata']
                score = sum([
                    metadata.get('has_numbers', 0) * 2,
                    metadata.get('has_percentages', 0) * 2,
                    metadata.get('has_time_periods', 0) * 2,
                    metadata.get('has_definitions', 0),
                    metadata.get('priority', 0) * 5
                ])
                if score > 0:
                    fallback_chunks.append((chunk, score))

            fallback_chunks.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = [chunk for chunk, _ in fallback_chunks[:top_k]]

        # Combine relevant chunks with enhanced metadata
        combined_sections = []
        for i, chunk in enumerate(relevant_chunks):
            metadata = chunk['metadata']
            section_header = metadata.get('section_header', 'Unknown Section')
            chunk_type = metadata.get('chunk_type', 'unknown')

            section_text = f"""[Rank: {i + 1}] [Type: {chunk_type}] [Section: {section_header}]
{chunk['text']}"""
            combined_sections.append(section_text)

        result = "\n\n--- RELEVANT SECTION ---\n\n".join(combined_sections)
        logger.info(f"[Jay121305] Retrieved {len(relevant_chunks)} chunks, total length: {len(result)}")
        return result


# Initialize global retriever
semantic_retriever = UltimateSemanticRetriever()

# Ultimate system prompt
ULTIMATE_SYSTEM_PROMPT = """
You are an expert document analyst with deep expertise across insurance, legal, technical, and scientific domains. Extract the most accurate and comprehensive answer based strictly on the provided document sections.

CRITICAL ANALYSIS REQUIREMENTS:
- Each section is marked with [Rank: X] (relevance), [Type: Y] (content type), and [Section: Z] (header)
- Higher-ranked sections contain more relevant information
- Special attention to numerical data, percentages, time periods, and specific conditions
- Cross-reference information across multiple sections for complete answers

RESPONSE FORMAT (JSON):
{
  "answer": "Comprehensive answer with ALL specific details: exact numbers, percentages, time periods, conditions, and qualifications found in the document",
  "confidence": "high | medium | low",
  "supporting_clauses": ["Specific section references with exact clause numbers/headers"],
  "matched_text": ["Exact text excerpts that directly support each part of the answer"],
  "reasoning": "Detailed explanation showing which sections provided which information and how they were combined"
}

ACCURACY RULES:
1. Extract ALL numerical values exactly as stated (30 days, 36 months, 5%, Rs 25,000, etc.)
2. Include ALL conditions and qualifications ("provided that", "subject to", "limited to")
3. When multiple sections mention the same topic, combine ALL information
4. For definitions, include the complete definition as stated
5. For benefits/coverage, include amounts, limits, and eligibility criteria
6. For waiting periods, include exact durations and applicability
7. Never simplify or summarize - provide complete, precise information
8. If information spans multiple sections, reference all relevant sections
9. Include exact terminology used in the document
10. Cross-check for consistency across sections

DOMAIN-SPECIFIC FOCUS:
- Insurance: Grace periods, waiting periods, coverage limits, exclusions, benefits, definitions
- Legal: Article numbers, constitutional provisions, rights, restrictions
- Technical: Specifications, safety requirements, maintenance procedures
- Scientific: Laws, principles, mathematical formulations

Remember: Accuracy is paramount. Include every relevant detail found in the provided sections.
"""


def extract_pdf_text_only(pdf_url: str) -> str:
    """Enhanced PDF extraction with better error handling"""
    try:
        logger.info(f"[Jay121305] Downloading PDF: {pdf_url[:50]}...")

        response = requests.get(pdf_url, stream=True, timeout=90)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)

        # Extract text with better formatting preservation
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            # Clean up text but preserve structure
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Reduce excessive newlines
            pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        doc.close()
        os.unlink(temp_path)

        logger.info(f"[Jay121305] Extracted {len(full_text)} characters from {len(pages_text)} pages")
        return full_text

    except Exception as e:
        logger.error(f"[Jay121305] PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")


async def ultimate_cloud_analysis(query: str, relevant_context: str, model_id: str = "moonshotai/kimi-k2-instruct") -> \
Dict[str, Any]:
    """Ultimate cloud analysis with enhanced error handling and retries"""
    try:
        # Optimize context length
        if len(relevant_context) > 120000:
            # Keep most relevant sections
            sections = relevant_context.split("--- RELEVANT SECTION ---")
            # Prioritize sections with higher ranks
            ranked_sections = []
            for section in sections:
                rank_match = re.search(r'\[Rank: (\d+)\]', section)
                rank = int(rank_match.group(1)) if rank_match else 999
                ranked_sections.append((rank, section))

            ranked_sections.sort(key=lambda x: x[0])
            relevant_context = "\n\n--- RELEVANT SECTION ---\n\n".join([section for _, section in ranked_sections[:8]])

        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": ULTIMATE_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""
QUESTION: {query}

DOCUMENT SECTIONS (Ranked by Relevance and Classified by Type):
{relevant_context}

Provide a comprehensive analysis based on ALL the provided sections. Extract every relevant detail and combine information from multiple sections when applicable."""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.03,  # Very low temperature for consistency
            "max_tokens": 2500
        }

        t0 = time.time()
        api_key = groq_pool.get_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        max_attempts = 5
        last_exception = None

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=45
                )
                response.raise_for_status()
                break

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if response.status_code == 429:
                    groq_pool.mark_rate_limited(api_key)
                    if attempt < max_attempts - 1:
                        api_key = groq_pool.get_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        wait_time = min(2 ** attempt, 8)  # Exponential backoff
                        logger.warning(f"[Jay121305] Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                raise e
            except Exception as ex:
                last_exception = ex
                if attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
                break
        else:
            if last_exception:
                raise last_exception
            raise Exception("All attempts failed")

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
                "processing_type": "ultimate_analysis",
                "context_length": len(relevant_context),
                "success": True,
                "tokens_used": response.json().get("usage", {}).get("total_tokens"),
                "time_used_seconds": round(elapsed, 2),
                "api_key_used": api_key[-5:]
            }
        except json.JSONDecodeError as e:
            logger.warning(f"[Jay121305] JSON decode failed: {e}, treating as raw text")
            return {
                "answer": result,
                "confidence": "medium (0.6)",
                "confidence_score": 0.6,
                "success": True,
                "model_used": model_id,
                "processing_type": "ultimate_analysis_raw",
                "tokens_used": response.json().get("usage", {}).get("total_tokens"),
                "time_used_seconds": round(elapsed, 2),
                "api_key_used": api_key[-5:]
            }

    except Exception as e:
        logger.error(f"[Jay121305] Ultimate analysis failed: {str(e)}")
        return {
            "answer": f"Analysis failed: {str(e)}",
            "confidence": "low (0.3)",
            "confidence_score": 0.3,
            "success": False,
            "model_used": model_id
        }


@app.get("/")
def root():
    return {
        "message": "Ultimate Document Intelligence Assistant v6.0",
        "status": "running",
        "features": [
            "Ultimate chunking with metadata",
            "Multi-domain keyword search",
            "Document type detection",
            "Enhanced error handling",
            "Comprehensive context analysis"
        ],
        "supported_domains": ["Insurance", "Legal", "Technical", "Scientific", "General"],
        "user": "Jay121305",
        "version": "6.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "processing_type": "ultimate_semantic_cloud",
        "api_available": True,
        "semantic_retriever": semantic_retriever.is_indexed,
        "document_type": semantic_retriever.document_type,
        "api_pool_stats": groq_pool.usage_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }


@app.post("/api/v1/hackrx/run")
async def hackathon_endpoint(
        request: HackathonRequest,
        authorization: Optional[str] = Header(None)
):
    """Ultimate hackathon endpoint with maximum accuracy optimization"""
    try:
        start_time = time.time()

        if authorization:
            logger.info(f"[Jay121305] Authorization header received: {authorization[:20]}...")
        else:
            logger.warning(f"[Jay121305] No authorization header provided")

        logger.info(f"[Jay121305] ULTIMATE request: {len(request.questions)} questions")
        logger.info(f"[Jay121305] Document URL: {request.documents[:50]}...")

        # Enhanced PDF extraction
        pdf_text = extract_pdf_text_only(request.documents)
        logger.info(f"[Jay121305] PDF extracted: {len(pdf_text)} characters")

        # Ultimate document indexing
        await semantic_retriever.index_document(pdf_text)

        answers = []
        success_count = 0
        fallback_count = 0
        failed_count = 0

        # Optimized model selection
        primary_model = "moonshotai/kimi-k2-instruct"
        fallback_model = "llama-3.1-70b-versatile"

        for i, question in enumerate(request.questions):
            question_start = time.time()
            logger.info(f"[Jay121305] Question {i + 1}/{len(request.questions)}: {question[:100]}...")

            # Ultimate semantic retrieval
            relevant_context = await semantic_retriever.retrieve_relevant_chunks(question, top_k=10)

            # Primary analysis with ultimate prompt
            result_primary = await ultimate_cloud_analysis(question, relevant_context, model_id=primary_model)

            if result_primary.get("success", False) and result_primary.get("confidence_score", 0.0) >= 0.5:
                answer_text = result_primary.get("answer", "Unable to determine")
                success_count += 1

                question_time = time.time() - question_start
                logger.info(
                    f"[Jay121305] [Q{i + 1}/{len(request.questions)}] SUCCESS | "
                    f"confidence: {result_primary.get('confidence')} | "
                    f"time: {question_time:.2f}s | "
                    f"tokens: {result_primary.get('tokens_used')}"
                )
                answers.append(answer_text)

            elif result_primary.get("success", False):
                # Try fallback model for low confidence
                logger.info(
                    f"[Jay121305] Low confidence ({result_primary.get('confidence_score', 0.0)}), trying fallback")
                result_fallback = await ultimate_cloud_analysis(question, relevant_context, model_id=fallback_model)

                if (result_fallback.get("success", False) and
                        result_fallback.get("confidence_score", 0.0) > result_primary.get("confidence_score", 0.0)):
                    answer_text = result_fallback.get("answer", "Unable to determine")
                    fallback_count += 1

                    question_time = time.time() - question_start
                    logger.info(f"[Jay121305] [Q{i + 1}] FALLBACK BETTER | time: {question_time:.2f}s")
                    answers.append(answer_text)
                else:
                    # Use primary answer despite low confidence
                    answer_text = result_primary.get("answer", "Unable to determine")
                    question_time = time.time() - question_start
                    logger.info(f"[Jay121305] [Q{i + 1}] PRIMARY KEPT | time: {question_time:.2f}s")
                    answers.append(answer_text)
            else:
                # Both failed
                answer_text = "Analysis failed: Unable to process with available models."
                failed_count += 1
                question_time = time.time() - question_start
                logger.error(f"[Jay121305] [Q{i + 1}] FAILED | time: {question_time:.2f}s")
                answers.append(answer_text)

        total_time = time.time() - start_time
        logger.info(f"[Jay121305] COMPLETED - Total time: {total_time:.2f}s")
        logger.info(
            f"[Jay121305] Stats -- Success: {success_count}, Fallback: {fallback_count}, Failed: {failed_count}")
        logger.info(f"[Jay121305] API usage: {groq_pool.usage_stats}")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"[Jay121305] Ultimate endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"[Jay121305] Starting Ultimate Document Intelligence Assistant v6.0")
    uvicorn.run(app, host="0.0.0.0", port=8000)
