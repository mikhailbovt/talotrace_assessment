"""Run the isolated text-only LTX test through the existing private Comfy tunnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:18188"


def request(path: str, data: dict | None = None):
    req = urllib.request.Request(
        BASE + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def workflow(run_id: str) -> dict:
    graph = {}

    def add(key, kind, **inputs):
        graph[str(key)] = {"class_type": kind, "inputs": inputs}
        return [str(key), 0]

    model = add(
        1,
        "UNETLoader",
        unet_name="ltx-2.5-22b-dev-transformer-bf16.safetensors",
        weight_dtype="default",
    )
    clip = add(
        2,
        "CLIPLoader",
        clip_name="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        type="ltxv",
        device="default",
    )
    video_vae = add(3, "VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors")
    audio_vae = add(4, "VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors")
    positive = add(
        5,
        "CLIPTextEncode",
        clip=clip,
        text=(ROOT / "workflows/ltx-text/prompt.txt").read_text(encoding="utf-8-sig"),
    )
    negative = add(
        6,
        "CLIPTextEncode",
        clip=clip,
        text=(
            "blurry, distorted letters, extra letters, extra dots, disappearing dots, "
            "unreadable text, busy background, flickering, camera movement, hands, people, "
            "photorealism, 3D rendering, subtitles, speech, music"
        ),
    )
    cond = add(7, "LTXVConditioning", positive=positive, negative=negative, frame_rate=24.0)
    vlatent = add(8, "EmptyLTXVLatentVideo", width=960, height=544, length=121, batch_size=1)
    alatent = add(
        9,
        "LTXVEmptyLatentAudio",
        frames_number=121,
        frame_rate=24.0,
        batch_size=1,
        audio_vae=audio_vae,
    )
    av = add(10, "LTXVConcatAVLatent", video_latent=vlatent, audio_latent=alatent)
    stg = add(
        11,
        "LTXVSpatioTemporalGuidance",
        model=model,
        scale=1.0,
        blocks="28",
        start_percent=0.0,
        end_percent=1.0,
    )
    mod = add(
        12,
        "LTXVModalityGuidance",
        model=stg,
        modality_scale=3.0,
        start_percent=0.0,
        end_percent=1.0,
    )
    guide = add(
        13,
        "LTXVDualCFGGuider",
        model=mod,
        positive=cond,
        negative=["7", 1],
        video_cfg=3.0,
        audio_cfg=7.0,
    )
    noise = add(14, "RandomNoise", noise_seed=20260908)
    sampler = add(15, "KSamplerSelect", sampler_name="euler")
    sigmas = add(
        16,
        "LTXVScheduler",
        steps=30,
        max_shift=2.05,
        base_shift=0.95,
        stretch=True,
        terminal=0.1,
        latent=vlatent,
    )
    samples = add(
        17,
        "SamplerCustomAdvanced",
        noise=noise,
        guider=guide,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=av,
    )
    split = add(18, "LTXVSeparateAVLatent", av_latent=samples)
    up = add(
        19,
        "LatentUpscaleModelLoader",
        model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    )
    high = add(20, "LTXVLatentUpsampler", samples=split, upscale_model=up, vae=video_vae)
    high_av = add(21, "LTXVConcatAVLatent", video_latent=high, audio_latent=["18", 1])
    refined_model = add(
        22,
        "LoraLoaderModelOnly",
        model=model,
        lora_name="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        strength_model=1.0,
    )
    refined_guide = add(
        23,
        "LTXVDualCFGGuider",
        model=refined_model,
        positive=cond,
        negative=["7", 1],
        video_cfg=1.0,
        audio_cfg=1.0,
    )
    refined_noise = add(24, "RandomNoise", noise_seed=20260909)
    refined_sampler = add(25, "KSamplerSelect", sampler_name="euler_ancestral")
    refined_sigmas = add(26, "ManualSigmas", sigmas="0.85, 0.725, 0.421875, 0.0")
    refined = add(
        27,
        "SamplerCustomAdvanced",
        noise=refined_noise,
        guider=refined_guide,
        sampler=refined_sampler,
        sigmas=refined_sigmas,
        latent_image=high_av,
    )
    final = add(28, "LTXVSeparateAVLatent", av_latent=refined)
    images = add(
        29,
        "VAEDecodeTiled",
        samples=final,
        vae=video_vae,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=16,
    )
    audio = add(30, "LTXVAudioVAEDecode", samples=["28", 1], audio_vae=audio_vae)
    video = add(
        31, "CreateVideo", images=images, fps=24.0, audio=audio, bit_depth=8, color_space="sRGB"
    )
    add(
        32,
        "SaveVideo",
        video=video,
        filename_prefix=f"tests/ltx-text/{run_id}/hydrogen",
        **{
            "format": "mp4",
            "format.codec": "h264",
            "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": 18,
        },
    )
    return graph


def collect(dest: Path, record: dict):
    """Copy native outputs through the SSH tunnel and record media integrity."""
    status = record.get("status", {})
    if not status.get("completed") or status.get("status_str") != "success":
        raise RuntimeError("The generation did not complete successfully.")
    found = []

    def visit(value):
        if isinstance(value, dict):
            if "filename" in value and value.get("type") == "output":
                found.append(value)
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(record.get("outputs", {}))
    if not found:
        raise RuntimeError("The completed workflow has no downloadable output.")
    reports = []
    for item in found:
        filename = item["filename"]
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Comfy returned an unsafe output filename.")
        target = dest / filename
        params = {key: item[key] for key in ("filename", "subfolder", "type") if key in item}
        with urllib.request.urlopen(
            BASE + "/view?" + urllib.parse.urlencode(params), timeout=120
        ) as response:
            with target.open("wb") as output:
                shutil.copyfileobj(response, output)
        with target.open("rb") as saved:
            checksum = hashlib.file_digest(saved, "sha256").hexdigest()
        report = {
            "file": filename,
            "bytes": target.stat().st_size,
            "sha256": checksum,
        }
        if target.suffix.lower() == ".mp4":
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(target),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            report["ffprobe"] = json.loads(probe.stdout)
            decode = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"],
                capture_output=True,
                text=True,
            )
            report["decode_returncode"] = decode.returncode
            report["decode_errors"] = decode.stderr
            (dest / "media-integrity.json").write_text(json.dumps(report, indent=2))
            if decode.returncode or decode.stderr.strip():
                raise RuntimeError("Video failed full decoding; inspect media-integrity.json.")
            for second in (0, 1, 2, 3, 4, 5):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-ss",
                        str(second),
                        "-i",
                        str(target),
                        "-frames:v",
                        "1",
                        str(dest / f"frame-{second:02d}s.png"),
                    ],
                    check=True,
                )
        reports.append(report)
    (dest / "collected.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps({"collected": [r["file"] for r in reports]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        raise ValueError("run-id must be a plain directory name")
    dest = ROOT / "data/tests/ltx-text" / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    if args.collect:
        prompt_id = json.loads((dest / "submission.json").read_text())["prompt_id"]
        history = request("/history/" + prompt_id)
        if prompt_id not in history:
            raise RuntimeError("The prompt has not finished.")
        (dest / "history.json").write_text(json.dumps(history[prompt_id], indent=2))
        collect(dest, history[prompt_id])
        return
    if (dest / "submission.json").exists():
        raise RuntimeError(
            "This run was already submitted; use a new run ID for an intentional retry."
        )
    graph = workflow(args.run_id)
    (dest / "workflow-api.json").write_text(json.dumps(graph, indent=2))
    if args.build_only:
        return
    started = time.monotonic()
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "text_only": True,
        "image_inputs": [],
        "external_audio_inputs": [],
        "native_generated_audio": True,
        "model_manifest": json.loads((ROOT / "infra/runpod/models.lock.json").read_text()),
        "used_model_files": sorted(
            {
                v
                for n in graph.values()
                for v in n["inputs"].values()
                if isinstance(v, str) and v.endswith(".safetensors")
            }
        ),
        "status": "submitting",
    }
    try:
        submitted = request("/prompt", {"prompt": graph, "client_id": "talotrace-ltx-text-test"})
    except urllib.error.HTTPError as exc:
        (dest / "validation-error.json").write_bytes(exc.read())
        raise
    (dest / "submission.json").write_text(json.dumps(submitted, indent=2))
    prompt_id = submitted["prompt_id"]
    print(json.dumps({"prompt_id": prompt_id, "output_directory": str(dest)}), flush=True)
    metadata["prompt_id"] = prompt_id
    metadata["status"] = "running"
    (dest / "run.json").write_text(json.dumps(metadata, indent=2))
    while True:
        history = request("/history/" + prompt_id)
        if prompt_id in history:
            record = history[prompt_id]
            (dest / "history.json").write_text(json.dumps(record, indent=2))
            metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
            metadata["finished_at"] = datetime.now(UTC).isoformat()
            metadata["status"] = record["status"]
            (dest / "run.json").write_text(json.dumps(metadata, indent=2))
            print(
                json.dumps(
                    {
                        "elapsed_seconds": metadata["elapsed_seconds"],
                        "status": record["status"],
                        "outputs": record.get("outputs"),
                    }
                ),
                flush=True,
            )
            collect(dest, record)
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
