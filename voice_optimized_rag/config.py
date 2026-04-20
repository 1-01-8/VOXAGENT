"""Configuration management for VoiceAgentRAG."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class VORConfig(BaseSettings):
    """Central configuration for the VoiceAgentRAG system."""

    model_config = {"env_prefix": "VOR_", "env_file": ".env", "extra": "ignore"}

    # LLM settings
    llm_provider: Literal["openai", "anthropic", "ollama", "gemini", "siliconflow"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_temperature: float = 0.3

    # Embedding settings
    embedding_provider: Literal["openai", "ollama", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Vector store settings
    vector_store_provider: Literal["faiss", "qdrant"] = "faiss"
    faiss_index_path: Path = Path("data/faiss_index")
    retrieval_latency_ms: float = 0  # Simulated retrieval latency for benchmarking

    # Qdrant settings (when vector_store_provider="qdrant")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "voice_rag"

    # Semantic cache settings
    cache_max_size: int = 2000
    cache_ttl_seconds: float = 300.0
    cache_similarity_threshold: float = 0.40  # Tuned for query-to-document cosine similarity

    # Slow thinker settings
    prediction_strategy: Literal["keyword", "llm"] = "llm"
    max_predictions: int = 5
    prefetch_top_k: int = 10
    slow_thinker_rate_limit: float = 0.5  # min seconds between predictions

    # Fast talker settings
    fast_talker_max_context_chunks: int = 10
    fast_talker_fallback_to_retrieval: bool = True

    # Conversation stream settings
    conversation_window_size: int = 10

    # Document chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Voice settings
    stt_provider: Literal["whisper", "deepgram", "openai", "sensevoice", "siliconflow", "none", ""] = "sensevoice"
    tts_provider: Literal["openai", "elevenlabs", "edge", "cosyvoice", "siliconflow", "none", ""] = "cosyvoice"
    whisper_model: str = "base.en"
    sample_rate: int = 16000
    vad_aggressiveness: int = 2

    # SenseVoice settings (STT + emotion)
    sensevoice_model: str = "iic/SenseVoiceSmall"
    sensevoice_device: str = "cuda:0"

    # CosyVoice settings (TTS + clone)
    cosyvoice_model: str = "iic/CosyVoice2-0.5B"
    cosyvoice_device: str = "cuda:0"
    cosyvoice_default_speaker: str = "中文女"
    cosyvoice_reference_audio: str = ""

    # SiliconFlow Cloud Voice API (FunAudioLLM/SenseVoiceSmall + CosyVoice2)
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_stt_model: str = "FunAudioLLM/SenseVoiceSmall"
    siliconflow_tts_model: str = "FunAudioLLM/CosyVoice2-0.5B"
    siliconflow_tts_voice: str = "alex"       # 预置音色后缀：alex/anna/bella/...
    siliconflow_tts_sample_rate: int = 24000
    siliconflow_tts_format: str = "pcm"
    siliconflow_tts_speed: float = 1.0

    # Dialogue settings
    intent_model: str = ""  # empty = use main LLM
    emotion_angry_threshold: int = 2  # consecutive angry turns before transfer
    memory_short_term_turns: int = 10
    memory_compress_model: str = ""  # empty = use main LLM
    transfer_max_failures: int = 3

    # Agent settings
    agent_max_iterations: int = 10
    agent_tool_timeout: float = 3.0  # seconds
    agent_tool_retry: int = 1

    # Knowledge base settings
    kb_confidence_threshold: float = 0.75
    kb_categories: list[str] = ["product", "policy", "faq", "competitor"]

    # Session logging
    session_log_dir: str = "logs/sessions"

    # Gemini / Vertex AI settings
    gemini_api_key: str = ""
    vertex_project: str = ""
    vertex_location: str = "us-central1"
