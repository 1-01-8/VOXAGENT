#!/usr/bin/env python3
"""Full voice demo: speak into the mic, get voice responses.

This demo creates a complete voice conversation loop:
    1. Listen via microphone → STT → text
    2. Feed to Memory Router → get streaming response
    3. Response → TTS → speaker
    4. Real-time metrics display

Usage:
    VOR_LLM_API_KEY=sk-... python examples/voice_demo.py --docs path/to/docs/

    # With Ollama (local, no API key):
    python examples/voice_demo.py --provider ollama --docs path/to/docs/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig
from voice_optimized_rag.voice.audio_stream import AudioStream
from voice_optimized_rag.voice.stt import create_stt
from voice_optimized_rag.voice.tts import create_tts


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-Optimized RAG Voice Demo")
    parser.add_argument("--docs", type=Path, help="Directory of documents to ingest")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "ollama", "gemini"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--stt", default="whisper", choices=["whisper", "openai"])
    parser.add_argument("--tts", default="edge", choices=["edge", "openai"])
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    # Config
    config_kwargs = {"llm_provider": args.provider, "stt_provider": args.stt, "tts_provider": args.tts}
    if args.model:
        config_kwargs["llm_model"] = args.model
    if args.api_key:
        config_kwargs["llm_api_key"] = args.api_key
    if args.provider == "ollama":
        config_kwargs.setdefault("llm_model", "llama3.2")
        config_kwargs["embedding_provider"] = "ollama"
        config_kwargs["embedding_model"] = "nomic-embed-text"
        config_kwargs["embedding_dimension"] = 768
    elif args.provider == "gemini":
        config_kwargs.setdefault("llm_model", "gemini-2.5-flash")

    config = VORConfig(**config_kwargs)

    # Initialize components
    router = MemoryRouter(config)
    audio = AudioStream(sample_rate=config.sample_rate, vad_aggressiveness=config.vad_aggressiveness)

    stt_kwargs = {"model_size": args.whisper_model}
    if args.stt == "openai":
        stt_kwargs = {"api_key": config.llm_api_key}
    stt = create_stt(args.stt, **stt_kwargs)

    tts_kwargs = {}
    if args.tts == "openai":
        tts_kwargs = {"api_key": config.llm_api_key}
    tts = create_tts(args.tts, **tts_kwargs)

    print("=" * 60)
    print("  Voice-Optimized RAG — Voice Demo")
    print("  Speak into your microphone to start a conversation")
    print("=" * 60)

    await router.start(log_level=args.log_level)

    # Ingest documents
    if args.docs and args.docs.is_dir():
        print(f"\nIngesting documents from {args.docs}...")
        count = await router.ingest_directory(args.docs)
        print(f"Ingested {count} chunks.")
        router.save_index()

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        async for utterance in audio.listen():
            # STT
            stt_start = time.perf_counter()
            text = await stt.transcribe(utterance, config.sample_rate)
            stt_ms = (time.perf_counter() - stt_start) * 1000

            if not text.strip():
                continue

            print(f"You: {text}  [STT: {stt_ms:.0f}ms]")

            # Query the memory router
            query_start = time.perf_counter()
            response_parts: list[str] = []
            async for chunk in router.query_stream(text):
                response_parts.append(chunk)
            response = "".join(response_parts)
            query_ms = (time.perf_counter() - query_start) * 1000

            print(f"Assistant: {response}")

            # TTS
            tts_start = time.perf_counter()
            audio_data = await tts.synthesize(response)
            tts_ms = (time.perf_counter() - tts_start) * 1000

            await audio.play(audio_data)

            hit_rate = router.metrics.cache_hit_rate
            print(f"  [STT: {stt_ms:.0f}ms | RAG: {query_ms:.0f}ms | TTS: {tts_ms:.0f}ms | Cache: {hit_rate:.0%}]\n")

    except KeyboardInterrupt:
        print("\n")
    finally:
        await router.stop()
        print("Final metrics:")
        summary = router.metrics.summary()
        print(f"  Cache hit rate: {summary.get('cache_hit_rate', 'N/A')}")
        latency = summary.get("latency", {}).get("fast_talker", {})
        if "total_response" in latency:
            print(f"  Avg response: {latency['total_response']['avg_ms']:.1f}ms")
        print()


if __name__ == "__main__":
    asyncio.run(main())
