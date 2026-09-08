"""Kokoro-only speech, validated from actual PCM sample counts."""

import io
import wave

import httpx
import numpy as np

from talotrace.artifacts.store import sha256


def inspect_wav(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
            raise ValueError("Expected mono 24kHz PCM16 Kokoro audio")
        count = audio.getnframes()
        samples = audio.readframes(count)
        if not count or len(samples) != count * 2:
            raise ValueError("Narration is empty or truncated")
    values = np.frombuffer(samples, dtype="<i2").astype(np.float64) / 32768
    rms = float(np.sqrt(np.mean(values**2)))
    if not np.isfinite(values).all() or rms < 0.0001:
        raise ValueError("Narration contains no usable speech signal")
    return {
        "sha256": sha256(data),
        "bytes": len(data),
        "sample_rate": 24000,
        "samples": count,
        "duration_seconds": count / 24000,
        "rms": rms,
        "pcm_sha256": sha256(samples),
    }


async def speak(base_url: str, text: str, voice: str = "af_heart") -> tuple[bytes, dict]:
    if voice != "af_heart" or not text.strip() or len(text) > 10000:
        raise ValueError("Narration requires the complete script and selected af_heart voice")
    async with httpx.AsyncClient(base_url=base_url, timeout=600, trust_env=False) as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "speed": 1.0,
            },
        )
        response.raise_for_status()
    data = response.content
    return data, {
        "provider": "Kokoro-82M",
        "voice": voice,
        "speed": 1.0,
        "text": text,
        "text_sha256": sha256(text.encode("utf-8")),
        "audio": inspect_wav(data),
    }
