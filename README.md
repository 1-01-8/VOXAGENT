# VoiceAgentRAG

Traditional RAG (Retrieval-Augmented Generation) kills real-time voice conversations. A 200ms+ vector database lookup blows the latency budget, making conversations feel unnatural. **VoiceAgentRAG** solves this with a **dual-agent architecture**: a background *Slow Thinker* continuously pre-fetches context into a fast cache, while a foreground *Fast Talker* reads only from this instant-access cache.

### CLI Demo

```bash

# With Ollama (fully local, no API key)
python examples/cli_demo.py --provider ollama --model llama3.2 --docs knowledge_base/
```

## Tests

```bash
# Unit + integration tests (32 tests)
python -m pytest tests/ -v
```

