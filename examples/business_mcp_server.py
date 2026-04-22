#!/usr/bin/env python3
"""Start the VoxCare business MCP server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.mcp_server import build_mcp_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the VoxCare business MCP server")
    parser.add_argument("--docs", type=str, default="knowledge_base/", help="知识库目录路径")
    parser.add_argument("--provider", type=str, default="ollama", help="LLM provider")
    parser.add_argument("--model", type=str, default="llama3.2", help="LLM model")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="LLM base URL")
    args = parser.parse_args()

    embedding_provider = "ollama" if args.provider == "ollama" else "openai"
    embedding_model = "nomic-embed-text" if args.provider == "ollama" else "text-embedding-3-small"
    embedding_dimension = 768 if args.provider == "ollama" else 1536

    config = VORConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        llm_base_url=args.base_url,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        stt_provider="none",
        tts_provider="none",
    )
    server = build_mcp_server(config=config, docs_dir=args.docs)
    server.run()


if __name__ == "__main__":
    main()