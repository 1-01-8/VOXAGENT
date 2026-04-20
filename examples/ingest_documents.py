#!/usr/bin/env python3
"""Ingest documents into the FAISS vector store.

Usage:
    VOR_LLM_API_KEY=sk-... python examples/ingest_documents.py path/to/docs/

    # With Ollama embeddings:
    python examples/ingest_documents.py path/to/docs/ --embedding-provider ollama
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_optimized_rag import MemoryRouter, VORConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS index")
    parser.add_argument("directory", type=Path, nargs="?", default=None, help="Directory of documents to ingest")
    parser.add_argument("--input-dir", type=Path, dest="input_dir", default=None, help="Alias for positional directory argument")
    parser.add_argument("--output", type=Path, default=Path("data/faiss_index"), help="Output index path")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--extensions", nargs="+", default=None, help="File extensions (e.g., .txt .md)")
    args = parser.parse_args()

    directory = args.input_dir or directory
    if directory is None:
        parser.error("directory is required (provide as positional argument or via --input-dir)")

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    config_kwargs = {
        "llm_provider": args.provider,
        "faiss_index_path": args.output,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    if args.api_key:
        config_kwargs["llm_api_key"] = args.api_key
    if args.embedding_provider:
        config_kwargs["embedding_provider"] = args.embedding_provider
    if args.provider == "ollama":
        config_kwargs["embedding_provider"] = args.embedding_provider or "ollama"
        config_kwargs["embedding_model"] = "nomic-embed-text"
        config_kwargs["embedding_dimension"] = 768

    config = VORConfig(**config_kwargs)
    router = MemoryRouter(config)

    extensions = set(args.extensions) if args.extensions else None

    print(f"Ingesting documents from: {directory}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")

    count = await router.ingest_directory(directory, extensions=extensions)
    print(f"\nIngested {count} chunks")

    router.save_index(args.output)
    print(f"Index saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
