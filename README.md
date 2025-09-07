# Concise Adaptive Document Assistant v9.3

A FastAPI-based document analysis system that uses adaptive AI models to provide accurate, concise answers from PDF documents. Features smart complexity detection, dynamic keyword extraction, and multi-strategy document retrieval.

## 🚀 Features

- **Adaptive AI Processing**: Automatically adjusts response timing based on query complexity
- **Dynamic Keyword Extraction**: Uses YAKE algorithm for intelligent document analysis
- **Multi-Strategy Search**: Advanced chunk scoring with metadata-weighted reranking
- **Dual Model System**: Primary Llama-4-Scout with Kimi-K2 fallback
- **PDF Processing**: Direct PDF URL ingestion and text extraction
- **Concise Responses**: Professional, clean formatting (30-40 words typically)
- **Smart Rate Limiting**: Adaptive key pool management with 6 API keys
- **Performance Optimized**: ~1s for simple queries, extends only when needed

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kg290/adaptive-document-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```env
# Optional: Add any additional configuration
LOG_LEVEL=INFO
```

## 🔧 Configuration

The system comes pre-configured with:
- **6 Groq API keys** (built-in)
- **Primary Model**: meta-llama/llama-4-scout-17b-16e-instruct (30K TPM)
- **Fallback Model**: moonshotai/kimi-k2-instruct (10K TPM)
- **Adaptive concurrency**: 2-3 concurrent requests based on complexity

## 🚦 Usage

### Start the Server
```bash
python legal_document_analyser.py
```

The server will start on `http://127.0.0.1:8000/docs`

### API Endpoints

#### 1. Health Check
```bash
GET /
```

#### 2. Pool Status
```bash
GET /pool-status
```

#### 3. Process Documents (Main Endpoint)
```bash
POST /api/v1/hackrx/run
```

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payments?",
    "What are the maternity benefits covered?",
    "Define the term 'Hospital' as mentioned in the policy?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Grace period is thirty (30) days from the due date for premium payment, subject to policy terms.",
    "Maternity expenses covered after 9 months waiting period, limited to sum insured.",
    "Hospital means any institution with at least 10 beds for indoor treatment under qualified medical supervision."
  ]
}
```

## 🧠 How It Works

### 1. Document Processing
- Downloads PDF from provided URL
- Extracts text using PyMuPDF
- Creates comprehensive chunks with metadata scoring

### 2. Query Analysis
- **Complexity Detection**: Analyzes query for complexity indicators
- **Dynamic Keywords**: Uses YAKE to extract relevant terms
- **Adaptive Context**: Adjusts chunk count (7-8) based on complexity

### 3. AI Processing
- **Primary**: Llama-4-Scout for initial analysis
- **Fallback**: Kimi-K2 for complex queries or low confidence
- **Smart Triggers**: Automatic fallback based on answer quality

### 4. Response Optimization
- Clean formatting (removes \\n characters)
- Professional tone
- Essential information preservation
- 30-40 word target length

## 🎯 Query Complexity Examples

### Simple Queries (⚡)
- "What is the waiting period?"
- "Is maternity covered?"
- "What is the room rent limit?"

### Complex Queries (🔍)
- "What are all the exclusions and their conditions?"
- "Define comprehensive coverage including all sub-limits?"
- "Explain the complete claim process with requirements?"

## 📊 Performance Metrics

- **Simple Queries**: ~1.0s average response time
- **Complex Queries**: ~2.5s average response time
- **Success Rate**: >95% with dual model system
- **Concurrency**: 2-3 requests (adaptive based on complexity)
- **Rate Limits**: 30K TPM primary, 10K TPM fallback

## 🔧 Advanced Configuration

### Modify Complexity Detection
Edit `analyze_query_complexity()` function to add custom indicators:

```python
complexity_indicators = {
    'your_category': ['custom', 'keywords', 'here']
}
```

### Adjust Response Length
Modify prompts to change target word count:

```python
# In ADAPTIVE_PRIMARY_PROMPT
"Keep answers concise (X–Y words)"
```

### Custom Chunking
Modify `comprehensive_document_chunking()` for domain-specific patterns:

```python
critical_patterns = [
    r'your_custom_pattern',
    # Add patterns specific to your document type
]
```

## 📈 Monitoring

### Check Pool Status
```bash
curl http://127.0.0.1:8000/docs/pool-status
```

**Response includes:**
- Active/total keys
- Success rate
- Average response time
- Model usage statistics
- Complex queries handled

### Logs
The system provides detailed logging:
- Query complexity analysis
- Model selection decisions
- Performance metrics
- Error tracking

## 🛡️ Error Handling

- **Rate Limiting**: Automatic key rotation and cooling
- **Fallback System**: Secondary model for failed requests
- **Retry Logic**: Built-in retry with exponential backoff
- **Graceful Degradation**: Returns best available answer

## 🔍 Technical Details

### Key Components

1. **AdaptiveKeyPool**: Manages 6 API keys with rate limiting
2. **AdaptiveRetriever**: Smart document chunking and retrieval
3. **Dynamic Keyword Extraction**: YAKE-based relevance scoring
4. **Multi-Strategy Search**: Combines exact matching, semantic similarity, and metadata scoring

### Architecture
```
PDF URL → Text Extraction → Dynamic Chunking → Keyword Analysis
                                    ↓
Query → Complexity Analysis → Context Retrieval → AI Processing → Clean Response
```

## 📝 Example Use Cases

- **Insurance Policy Analysis**: Benefit explanations, coverage details
- **Legal Document Review**: Contract terms, compliance requirements
- **Technical Documentation**: Feature explanations, troubleshooting
- **Academic Papers**: Research summaries, methodology explanations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

*Built with FastAPI, Groq API, and advanced NLP techniques for professional document analysis.*
