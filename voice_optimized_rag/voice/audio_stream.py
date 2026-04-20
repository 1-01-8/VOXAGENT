"""Audio stream handling: microphone input with VAD, speaker output."""

from __future__ import annotations

import asyncio
import struct
from typing import AsyncIterator

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("audio_stream")

# VAD frame duration in ms (must be 10, 20, or 30 for webrtcvad)
VAD_FRAME_MS = 30


class AudioStream:
    """Handles microphone input with Voice Activity Detection and speaker output.

    Uses sounddevice for audio I/O and webrtcvad for speech detection.
    Yields complete utterances (speech segments) as byte buffers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        silence_threshold_ms: int = 800,
    ) -> None:
        self._sample_rate = sample_rate
        self._vad_aggressiveness = vad_aggressiveness
        self._silence_threshold_ms = silence_threshold_ms
        self._frame_size = int(sample_rate * VAD_FRAME_MS / 1000)  # samples per frame

    async def listen(self) -> AsyncIterator[bytes]:
        """Listen to the microphone and yield complete utterances.

        Each yielded bytes object is a complete speech segment (from speech
        onset to silence), as 16-bit PCM audio.
        """
        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError:
            raise ImportError("Install voice deps: pip install voice-optimized-rag[voice]")

        vad = webrtcvad.Vad(self._vad_aggressiveness)
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            audio_queue.put_nowait(bytes(indata))

        # Frame size in bytes (16-bit mono)
        frame_bytes = self._frame_size * 2
        frames_for_silence = int(self._silence_threshold_ms / VAD_FRAME_MS)

        stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_size,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )

        with stream:
            logger.info("Listening... (speak to start)")
            buffer = bytearray()
            silent_frames = 0
            is_speaking = False

            while True:
                frame = await audio_queue.get()

                # Ensure frame is the right size for VAD
                if len(frame) != frame_bytes:
                    continue

                is_speech = vad.is_speech(frame, self._sample_rate)

                if is_speech:
                    if not is_speaking:
                        is_speaking = True
                        logger.debug("Speech detected")
                    buffer.extend(frame)
                    silent_frames = 0
                elif is_speaking:
                    buffer.extend(frame)
                    silent_frames += 1
                    if silent_frames >= frames_for_silence:
                        # End of utterance
                        is_speaking = False
                        silent_frames = 0
                        utterance = bytes(buffer)
                        buffer = bytearray()
                        logger.debug(f"Utterance complete: {len(utterance)} bytes")
                        yield utterance

    async def play(self, audio_data: bytes, sample_rate: int | None = None) -> None:
        """Play audio data through the speaker.

        Args:
            audio_data: Raw audio bytes. For PCM: 16-bit mono.
                       For MP3/other formats from TTS: will attempt playback directly.
            sample_rate: Override sample rate (default: self._sample_rate).
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError("Install sounddevice: pip install voice-optimized-rag[voice]")

        rate = sample_rate or self._sample_rate

        # Try to interpret as 16-bit PCM
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            logger.warning("Could not interpret audio as PCM, skipping playback")
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: sd.play(audio_np, rate))
        await loop.run_in_executor(None, sd.wait)
