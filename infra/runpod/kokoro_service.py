"""Private Kokoro adapter; exposed only through the SSH tunnel."""

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from kokoro import KModel, KPipeline
from pydantic import BaseModel, Field

ROOT = Path(os.getenv("TALOTRACE_ROOT", "/workspace/talotrace"))


@asynccontextmanager
async def lifespan(app):
    model_root = ROOT / "models/kokoro"
    # Keep this small model on CPU to reserve A100 memory for LTX.
    model = KModel(config=str(model_root / "config.json"),
                   model=str(model_root / "kokoro-v1_0.pth")).to("cpu").eval()
    app.state.pipeline = KPipeline(lang_code="a", model=model)
    app.state.voice = torch.load(model_root / "voices/af_heart.pt",
                                 map_location="cpu", weights_only=True)
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(title="Private Kokoro TTS", lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=10000)
    voice: str = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.get("/health")
async def health():
    return {"status": "ready", "model": "hexgrad/Kokoro-82M",
            "voice": "af_heart", "device": "cpu", "sample_rate": 24000}


@app.post("/v1/audio/speech")
async def speech(body: SpeechRequest):
    if body.voice != "af_heart":
        raise HTTPException(422, "Only the configured af_heart voice is supported")

    def synthesize():
        chunks = []
        for text, _, audio in app.state.pipeline(
            body.input, voice=app.state.voice, speed=body.speed
        ):
            if audio is None:
                if text.strip():
                    raise ValueError("A narration segment has no audio")
                continue
            chunks.append(audio.detach().cpu().numpy())
        if not chunks:
            raise ValueError("No audio produced")
        audio = np.concatenate(chunks)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("Invalid audio produced")
        result = io.BytesIO()
        sf.write(result, audio, 24000, format="WAV", subtype="PCM_16")
        return result.getvalue()

    async with app.state.lock:
        try:
            data = await asyncio.to_thread(synthesize)
        except Exception:
            raise HTTPException(502, "Kokoro synthesis failed") from None
    return Response(content=data, media_type="audio/wav")
