# VoiceAgentRAG

Traditional RAG (Retrieval-Augmented Generation) kills real-time voice conversations. A 200ms+ vector database lookup blows the latency budget, making conversations feel unnatural. **VoiceAgentRAG** solves this with a **dual-agent architecture**: a background *Slow Thinker* continuously pre-fetches context into a fast cache, while a foreground *Fast Talker* reads only from this instant-access cache.

**Key Results** (200 queries, 10 scenarios, Qdrant Cloud):
- **75% cache hit rate** (86% on warm turns 5-9)
- **316x retrieval speedup** (110ms → 0.35ms)
- **16.5 seconds** of retrieval latency saved across 150 cache hits

---


```
### Option C: Fully local with Ollama (no API keys)

```bash
# 1. Install Ollama and pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Clone and install
git clone https://github.com/SalesforceAIResearch/VoiceAgentRAG
cd VoiceAgentRAG
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Run fully local
python examples/cli_demo.py --provider ollama --model llama3.2 --docs knowledge_base/
```

Type questions and see the dual-agent system in action — watch the cache hit rate climb as you ask follow-up questions.

## Architecture

```
    User Speech (Audio Stream)
            |
            v
    +-----------------------------------------------+
    |              Memory Router                     |
    |                                                |
    |   +-----------------+   +------------------+   |
    |   |  Slow Thinker   |   |   Fast Talker    |   |
    |   |  (Background)   |   |  (Foreground)    |   |
    |   |                 |   |                  |   |
    |   | - Listens to    |   | - Reads cache    |   |
    |   |   stream        |   | - Generates      |   |
    |   | - Predicts next |   |   response       |   |
    |   |   topics        |   | - <200ms budget  |   |
    |   | - Searches FAISS|   |                  |   |
    |   | - Fills cache   |   |                  |   |
    |   +--------+--------+   +--------+---------+   |
    |            |                     |              |
    |            v                     v              |
    |   +----------------------------------------+   |
    |   |          Semantic Cache                 |   |
    |   |   (In-memory FAISS, sub-ms access)     |   |
    |   +----------------------------------------+   |
    |            |                                    |
    |            v                                    |
    |   +----------------------------------------+   |
    |   |    Vector Store (FAISS / Qdrant)        |   |
    |   +----------------------------------------+   |
    +------------------------------------------------+
```

**How it works:**
1. The **Slow Thinker** runs as a background async task, subscribing to the conversation stream
2. On each user utterance, it uses an LLM to predict 3-5 likely follow-up topics
3. It retrieves relevant documents for those predictions and pre-fills the semantic cache
4. The **Fast Talker** checks the cache first (sub-ms), only falling back to the vector DB on miss
5. On miss, retrieved results are cached — so the next similar query hits
6. As the conversation progresses, the cache warms up and responses get faster

## Installation

```bash
# From source (recommended)
git clone https://github.com/SalesforceAIResearch/VoiceAgentRAG
cd VoiceAgentRAG
uv venv .venv && source .venv/bin/activate

# Core + OpenAI
uv pip install -e ".[openai,dev]"

# With Qdrant Cloud support
uv pip install -e ".[openai,qdrant,dev]"

# With voice demo (STT/TTS)
uv pip install -e ".[all,dev]"
```

### Supported Providers

| Component | Providers |
|-----------|-----------|
| **LLM** | OpenAI, Anthropic, Gemini/Vertex AI, Ollama (local) |
| **Embeddings** | OpenAI, Ollama |
| **Vector Store** | FAISS (local), Qdrant Cloud |
| **STT** | Whisper (local), OpenAI |
| **TTS** | Edge TTS (free), OpenAI |

## Usage

### Python API

```python
import asyncio
from pathlib import Path
from voice_optimized_rag import MemoryRouter, VORConfig

async def main():
    config = VORConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
    )
    router = MemoryRouter(config)
    await router.start()

    # Ingest your documents
    await router.ingest_directory(Path("docs/"))
    router.save_index()

    # Query — dual-agent caching is automatic
    response = await router.query("What are your pricing plans?")
    print(response)

    # Stream responses
    async for chunk in router.query_stream("Tell me about enterprise features"):
        print(chunk, end="", flush=True)

    # Check metrics
    print(f"\nCache hit rate: {router.metrics.cache_hit_rate:.0%}")

    await router.stop()

asyncio.run(main())
```

### CLI Demo

```bash
# With OpenAI
export OPENAI_API_KEY="sk-..."
python examples/cli_demo.py --docs knowledge_base/

# With Ollama (fully local, no API key)
python examples/cli_demo.py --provider ollama --model llama3.2 --docs knowledge_base/
```

### Run Benchmarks

```bash
# 20-turn voice call simulation (Qdrant Cloud)
python test_voice_call.py \
  --vector-store qdrant \
  --qdrant-url "https://your-cluster.cloud.qdrant.io" \
  --qdrant-key "your-key"

# Full 200-query benchmark (10 scenarios)
python test_200.py \
  --vector-store qdrant \
  --qdrant-url "https://your-cluster.cloud.qdrant.io" \
  --qdrant-key "your-key"
```

## Configuration

### Environment Variables (simplest)

```bash
# Option 1: Standard OpenAI (default)
export OPENAI_API_KEY="sk-..."

# Option 2: Gemini
export GEMINI_API_KEY="AIza..."

# Option 3: Salesforce Research Gateway (auto-detected from URL)
export OPENAI_API_KEY="your-gateway-key"
export OPENAI_BASE_URL="https://gateway.salesforceresearch.ai/openai/process/v1"

# Option 4: Gemini via Vertex AI (auto-detected from VERTEX_PROJECT)
export VERTEX_PROJECT="your-gcp-project"
```

### Python Config

All settings via `VORConfig` or env vars (prefix `VOR_`):

```python
config = VORConfig(
    # LLM
    llm_provider="openai",          # "openai", "anthropic", "ollama", "gemini"
    llm_model="gpt-4o-mini",
    llm_api_key="sk-...",           # or set OPENAI_API_KEY env

    # Embeddings
    embedding_provider="openai",    # "openai", "ollama"
    embedding_model="text-embedding-3-small",

    # Vector Store
    vector_store_provider="faiss",  # "faiss" or "qdrant"

    # Cache tuning
    cache_max_size=2000,
    cache_ttl_seconds=300,
    cache_similarity_threshold=0.40,

    # Slow Thinker
    prediction_strategy="llm",      # "llm" or "keyword"
    max_predictions=5,
    prefetch_top_k=10,
)
```

## Benchmark Results

**200 queries across 10 diverse conversation scenarios on Qdrant Cloud (real network latency):**

| Metric | Traditional RAG | VoiceAgentRAG |
|--------|----------------|---------------|
| Retrieval latency | 110.4 ms | 0.35 ms |
| Retrieval speedup | 1x | **316x** |
| Cache hit rate | — | **75%** (150/200) |
| Warm-cache hit rate | — | **79%** (150/190) |

**Per-scenario hit rates:**

| Scenario | Hit Rate | Pattern |
|----------|----------|---------|
| Feature comparison | 95% | Broad tour |
| Pricing / API / Onboarding / Integrations | 85% | Focused topics |
| Troubleshooting | 80% | Problem-solving |
| Security & compliance | 70% | Single topic |
| New customer overview | 65% | Exploration |
| Mixed rapid-fire | 55% | Random |

## Tests

```bash
# Unit + integration tests (32 tests)
python -m pytest tests/ -v
```

## Project Structure

```
voice_optimized_rag/
  core/
    memory_router.py        # Central orchestrator
    slow_thinker.py         # Background prefetch agent
    fast_talker.py          # Foreground response agent
    semantic_cache.py       # FAISS-backed semantic cache
    conversation_stream.py  # Async event bus
  retrieval/
    vector_store.py         # FAISS wrapper
    qdrant_store.py         # Qdrant Cloud adapter
    embeddings.py           # Embedding providers
    document_loader.py      # Document chunking
  llm/
    base.py                 # Abstract LLM interface
    openai_provider.py      # OpenAI + Salesforce gateway
    anthropic_provider.py
    gemini_provider.py      # Gemini + Vertex AI
    ollama_provider.py      # Local models
  voice/
    stt.py                  # Speech-to-text
    tts.py                  # Text-to-speech
    audio_stream.py         # Mic/speaker with VAD
  config.py                 # Pydantic settings
  utils/
    metrics.py              # Latency instrumentation
    logging.py

examples/
  cli_demo.py               # Interactive text demo
  voice_demo.py             # Full voice conversation loop
  benchmark.py              # Latency comparison
  ingest_documents.py       # Document ingestion CLI

knowledge_base/              # Sample enterprise KB (NovaCRM)
tests/                       # 32 unit + integration tests
```

