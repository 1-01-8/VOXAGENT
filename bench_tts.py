"""[DEPRECATED · v1.0.0+] 独立测 **本地** CosyVoice2 推理速度。

⚠️ 本脚本仅适用于 v0.x 本地 TTS 链路。项目已于 v1.0.0 迁移至 SiliconFlow 云端 TTS
(voice_optimized_rag/voice/siliconflow_tts.py)，本脚本不再是日常基准工具。

仅在需要回滚本地 CosyVoice2 或对比本地 vs 云端 RTF 时才运行。
若要基准当前链路，请用:
    python tests/bench_rag_match.py        # RAG 匹配耗时
    python examples/benchmark.py           # 端到端延迟

─────────────────────────────────────────────────────────────
原说明：跑两次；第一次触发 JIT，第二次才是稳态 RTF。
"""
import os, sys, time, logging
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/home/xxm/VoxCareAgent/CosyVoice")
sys.path.insert(0, "/home/xxm/VoxCareAgent/CosyVoice/third_party/Matcha-TTS")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("bench")

import torch
from cosyvoice.cli.cosyvoice import CosyVoice2

MODEL_ID = "CosyVoice/pretrained_models/CosyVoice2-0.5B"
REF = "/home/xxm/VoxCareAgent/CosyVoice/asset/zero_shot_prompt.wav"
PROMPT = "希望你以后能够做的比我还好呦。"
TEXT = "你好！很高兴遇到你。今天天气不错，是个聊天的好日子呢！你在做什么呢？"

log.info("Loading model...")
t = time.perf_counter()
with torch.cuda.device(0):
    m = CosyVoice2(MODEL_ID, load_jit=True, load_trt=False, fp16=True)
log.info(f"load_ms={(time.perf_counter()-t)*1000:.0f}")
log.info(f"sr={m.sample_rate}")

# 打印 frontend 类型
try:
    log.info(f"text_frontend={m.frontend.text_frontend}")
except Exception:
    pass

# cache speaker
try:
    m.add_zero_shot_spk(PROMPT, REF, "bench")
    log.info("spk cached")
except Exception as e:
    log.warning(f"add_zero_shot_spk failed: {e}")

def run(label: str, stream: bool):
    t0 = time.perf_counter()
    t_first = None
    audio_total = 0
    with torch.cuda.device(0):
        gen = m.inference_zero_shot(TEXT, "", "", zero_shot_spk_id="bench", stream=stream)
        chunks = []
        for i, c in enumerate(gen):
            if t_first is None:
                t_first = time.perf_counter()
                log.info(f"[{label}] first_chunk_ms={(t_first-t0)*1000:.0f}")
            n = c["tts_speech"].shape[-1]
            audio_total += n
            log.info(f"[{label}] chunk#{i} samples={n} audio_ms={n*1000/m.sample_rate:.0f} dt_ms={(time.perf_counter()-t0)*1000:.0f}")
            chunks.append(c)
    total = (time.perf_counter()-t0)*1000
    audio_ms = audio_total*1000/m.sample_rate
    log.info(f"[{label}] TOTAL wall_ms={total:.0f} audio_ms={audio_ms:.0f} RTF={total/audio_ms:.2f}")

log.info("=== run 1 (cold/JIT warmup) ===")
run("cold", stream=False)
log.info("=== run 2 (stream, warm) ===")
run("warm-stream", stream=True)
log.info("=== run 3 (non-stream, warm) ===")
run("warm-flat", stream=False)
