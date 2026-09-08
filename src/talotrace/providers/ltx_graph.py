"""Pinned native LTX/MSR two-stage graphs, with explicitly distinct temporal guides."""

import json
from pathlib import Path

from talotrace.artifacts.store import json_bytes, sha256

PROFILE_PATH = Path("workflows/video/render-profile.json")
MODEL_LOCK_PATH = Path("infra/runpod/models.lock.json")


def render_identity_settings(settings: dict) -> dict:
    """Assembly-only timing must change the job identity without changing LTX samples."""
    return {key: value for key, value in settings.items() if key != "timing_policy"}


def render_settings() -> dict:
    profile = json.loads(PROFILE_PATH.read_text("utf-8"))
    lock = json.loads(MODEL_LOCK_PATH.read_text("utf-8"))
    return {
        "profile": profile,
        "profile_sha256": sha256(json_bytes(profile)),
        "models_sha256": sha256(json_bytes(lock)),
        "models": lock["models"],
    }


def build_graph(
    *,
    mode: str,
    phase: str,
    scene: dict,
    images: dict[str, str],
    profile: dict,
    seed: int,
    output_prefix: str,
    prompt_override: str | None = None,
) -> tuple[dict, dict]:
    """Images are uploaded Comfy input names; no filesystem paths from API clients."""
    if mode not in {"references", "whiteboard_loop"}:
        raise ValueError("Unknown generation mode")
    if phase not in {"draw", "erase"} or (mode == "references" and phase != "draw"):
        raise ValueError("Invalid clip phase")
    if not {"keyframe", "hand_reference"} <= images.keys():
        raise ValueError("Both the scene keyframe and actual hand reference are required")
    frames = profile["erase_frames" if phase == "erase" else "draw_frames"]
    graph: dict = {}

    def node(name, kind, **inputs):
        graph[name] = {"class_type": kind, "inputs": inputs}
        return [name, 0]

    model = node(
        "model",
        "UNETLoader",
        unet_name="ltx-2.5-22b-dev-transformer-bf16.safetensors",
        weight_dtype="default",
    )
    clip = node(
        "text_encoder",
        "CLIPLoader",
        clip_name="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        type="ltxv",
        device="default",
    )
    vae = node("video_vae", "VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors")
    audio_vae = node("audio_vae", "VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors")
    msr = node(
        "msr_loader",
        "ComfyUILTX25MSRICLoRALoader",
        model=model,
        lora_name="LTX-2.5-Licon-MSR-V1.safetensors",
        strength_model=1.0,
    )
    prompt = prompt_override or scene_prompt(mode, phase, scene, frames / profile["fps"])
    positive = node("positive", "CLIPTextEncode", clip=clip, text=prompt)
    negative = node(
        "negative",
        "CLIPTextEncode",
        clip=clip,
        text=(
            "music, soundtrack, narration, speech, camera movement, zoom, "
            "photorealistic classroom, 3D objects, extra hands, distorted fingers, "
            "extra symbols, altered chemistry, illegible "
            "labels, flickering ink, clutter, shadows covering labels, gradients, textured board"
        ),
    )
    node(
        "conditioning",
        "LTXVConditioning",
        positive=positive,
        negative=negative,
        frame_rate=float(profile["fps"]),
    )
    keyframe = node("keyframe", "LoadImage", image=images["keyframe"])
    hand = node("hand", "LoadImage", image=images["hand_reference"])
    if "blank" in images:
        blank = node("blank", "LoadImage", image=images["blank"])
    else:
        blank = node(
            "blank",
            "EmptyImage",
            width=profile["width"] * 2,
            height=profile["height"] * 2,
            batch_size=1,
            color=16777215,
        )
    empty = node(
        "empty_video",
        "EmptyLTXVLatentVideo",
        width=profile["width"],
        height=profile["height"],
        length=frames,
        batch_size=1,
    )
    audio = node(
        "empty_audio",
        "LTXVEmptyLatentAudio",
        frames_number=frames,
        frame_rate=float(profile["fps"]),
        batch_size=1,
        audio_vae=audio_vae,
    )
    temporal_guides = []
    if mode == "whiteboard_loop":
        temporal_guides = [
            ("finished_end", keyframe, frames - 1)
            if phase == "draw"
            else ("blank_end", blank, frames - 1)
        ]

    def condition(stage, latent, pos, neg):
        if mode == "whiteboard_loop":
            # Pin the actual first latent, including a zero noise mask. Appended
            # temporal tokens alone did not produce a blank first output frame.
            latent = node(
                f"{stage}_start_pin",
                "LTXVImgToVideoInplace",
                vae=vae,
                image=blank if phase == "draw" else keyframe,
                latent=latent,
                strength=1.0,
                bypass=False,
            )
        # Native guides and MSR both append guide tokens to video-only latents.
        # Crop all of them after sampling, then rebuild at the refinement resolution.
        for label, image, index in temporal_guides:
            name = f"{stage}_{label}"
            node(
                name,
                "LTXVAddGuide",
                positive=pos,
                negative=neg,
                vae=vae,
                latent=latent,
                image=image,
                frame_idx=index,
                strength=profile["guide_strength"],
            )
            pos, neg, latent = [name, 0], [name, 1], [name, 2]
        name = f"{stage}_msr"
        node(
            name,
            "ComfyUILTX25MSRMultiReferenceGuide",
            positive=pos,
            negative=neg,
            vae=vae,
            latent=latent,
            **({"pic1": hand} if mode == "whiteboard_loop" else {"pic1": keyframe, "pic2": hand}),
            msr_parameters=["msr_loader", 1],
            strength=profile["msr_strength"],
            reference_frames=profile["reference_frames"],
            use_tiled_encode=True,
            tile_size=256,
            tile_overlap=64,
        )
        return [name, 0], [name, 1], [name, 2]

    pos1, neg1, guided1 = condition("base", empty, ["conditioning", 0], ["conditioning", 1])
    av1 = node("base_av", "LTXVConcatAVLatent", video_latent=guided1, audio_latent=audio)
    stg = node(
        "stg",
        "LTXVSpatioTemporalGuidance",
        model=msr,
        scale=1.0,
        blocks="28",
        start_percent=0.0,
        end_percent=1.0,
    )
    modality = node(
        "modality",
        "LTXVModalityGuidance",
        model=stg,
        modality_scale=3.0,
        start_percent=0.0,
        end_percent=1.0,
    )
    guider = node(
        "base_guider",
        "LTXVDualCFGGuider",
        model=modality,
        positive=pos1,
        negative=neg1,
        video_cfg=profile["video_cfg"],
        audio_cfg=profile["audio_cfg"],
    )
    noise = node("base_noise", "RandomNoise", noise_seed=seed)
    sampler = node("base_sampler", "KSamplerSelect", sampler_name="euler")
    sigmas = node(
        "base_sigmas",
        "LTXVScheduler",
        steps=profile["steps"],
        max_shift=2.05,
        base_shift=0.95,
        stretch=True,
        terminal=0.1,
        latent=empty,
    )
    sample = node(
        "base_sample",
        "SamplerCustomAdvanced",
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=av1,
    )
    node("base_separate", "LTXVSeparateAVLatent", av_latent=sample)
    node("base_crop", "LTXVCropGuides", positive=pos1, negative=neg1, latent=["base_separate", 0])
    upscaler = node(
        "upscaler",
        "LatentUpscaleModelLoader",
        model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    )
    upscale = node(
        "upscale", "LTXVLatentUpsampler", samples=["base_crop", 2], upscale_model=upscaler, vae=vae
    )
    # CropGuides clears its metadata to None. The pinned MSR extension expects an
    # iterable when that key exists, so rebuild from pristine text conditioning.
    # Only the cropped/upscaled generated video latent carries into refinement.
    pos2, neg2, guided2 = condition("refine", upscale, ["conditioning", 0], ["conditioning", 1])
    av2 = node(
        "refine_av", "LTXVConcatAVLatent", video_latent=guided2, audio_latent=["base_separate", 1]
    )
    distilled = node(
        "refine_lora",
        "LoraLoaderModelOnly",
        model=msr,
        lora_name="ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        strength_model=1.0,
    )
    guider2 = node(
        "refine_guider",
        "LTXVDualCFGGuider",
        model=distilled,
        positive=pos2,
        negative=neg2,
        video_cfg=1.0,
        audio_cfg=1.0,
    )
    noise2 = node("refine_noise", "RandomNoise", noise_seed=seed + 1)
    sampler2 = node("refine_sampler", "KSamplerSelect", sampler_name="euler_ancestral")
    sigmas2 = node("refine_sigmas", "ManualSigmas", sigmas=profile["refinement_sigmas"])
    sample2 = node(
        "refine_sample",
        "SamplerCustomAdvanced",
        noise=noise2,
        guider=guider2,
        sampler=sampler2,
        sigmas=sigmas2,
        latent_image=av2,
    )
    node("refine_separate", "LTXVSeparateAVLatent", av_latent=sample2)
    node(
        "refine_crop", "LTXVCropGuides", positive=pos2, negative=neg2, latent=["refine_separate", 0]
    )
    decoded = node(
        "decode",
        "VAEDecodeTiled",
        samples=["refine_crop", 2],
        vae=vae,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=16,
    )
    # Audio latents support the native AV model, but are deliberately NEVER decoded or saved.
    video = node(
        "video",
        "CreateVideo",
        images=decoded,
        fps=float(profile["fps"]),
        bit_depth=8,
        color_space="sRGB",
    )
    node(
        "save",
        "SaveVideo",
        video=video,
        filename_prefix=output_prefix,
        **{
            "format": "mp4",
            "format.codec": "h264",
            "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": 18,
        },
    )
    return graph, {
        "mode": mode,
        "phase": phase,
        "frames": frames,
        "fps": profile["fps"],
        "seed": seed,
        "prompt": prompt,
        "temporal_guides": [
            {"name": label, "frame_index": index} for label, _, index in temporal_guides
        ],
        "references": (
            {"pic1": "hand_reference"}
            if mode == "whiteboard_loop"
            else {"pic1": "keyframe", "pic2": "hand_reference"}
        ),
        "first_frame_conditioning": (
            {
                "method": "LTXVImgToVideoInplace",
                "image": "blank" if phase == "draw" else "keyframe",
                "strength": 1.0,
                "stages": ["base_start_pin", "refine_start_pin"],
            }
            if mode == "whiteboard_loop"
            else None
        ),
        "reference_conditioning_stages": ["base_msr", "refine_msr"],
        "audio_decoded": False,
    }


def scene_prompt(mode: str, phase: str, scene: dict, duration_seconds: float) -> str:
    if mode == "whiteboard_loop":
        common = (
            "One continuous educational chemistry whiteboard shot, locked frontal camera, "
            "fixed composition, even light, white matte board, black marker doodles with "
            "restrained blue and yellow accents. Reference pic1 supplies ONLY the physical "
            "hand identity: matching skin, fingers and forearm. One hand enters the board. "
            "No camera cuts, zooms, dissolves, morphs, jump cuts, music or soundtrack. "
        )
        if phase == "erase":
            return common + (
                "The first frame shows the complete chemistry diagram. The matching hand "
                "holds a broad whiteboard eraser instead of a marker. Press the eraser "
                "against the board and physically wipe the ink away in overlapping strokes. "
                "Each mark disappears only when the eraser crosses that exact location; "
                "all untouched marks remain visible until wiped. Clear the diagram and all "
                "lettering progressively, then move the hand out of view. The final frame "
                f"at {duration_seconds:.3f} seconds is the completely clean white board. "
                "No marks fade away by themselves, no reverse drawing, no sudden blank cut."
            )
        return common + (
            "The first frame is completely blank white with no ink, diagram, lettering or "
            "hand. The reference hand enters with a marker and constructs the completed "
            "diagram supplied ONLY by the later temporal guide. New ink must appear directly "
            "under the moving marker tip, following its exact path. Draw each outline, ion, "
            "symbol, letter and bracket through visible strokes, and use appropriate colored "
            "marker strokes for the accents. Previously drawn marks stay fixed. Keep "
            "drawing through the shot, finish the entire guided diagram only near the final "
            f"frame at {duration_seconds:.3f} seconds, then withdraw the hand. "
            "Never cut, dissolve or reveal the completed image; no diagram parts or labels "
            "appear without being physically drawn. The completed chemical symbols must be "
            f"legible and exact. Scene objective: {scene['educational_objective']}. "
            f"Completed labels: {', '.join(scene['visible_labels'])}. "
            f"Drawing sequence: {scene['scene_direction']}"
        )
    common = (
        "Educational chemistry doodle on a completely white, matte whiteboard, locked frontal "
        "camera, fixed framing and even lighting. Match reference pic1 exactly for composition, "
        "thin black marker outlines, handwritten labels and restrained colored accents. "
        "Reference pic2 supplies the single drawing hand and marker identity. Only the hand "
        "and forearm may enter the board. Keep all chemical symbols and labels legible and "
        "unchanged. No background music, speech, sound effects or soundtrack. "
        f"Scene objective: {scene['educational_objective']}. "
        f"Visible labels: {', '.join(scene['visible_labels'])}. "
    )
    if phase == "erase":
        return common + (
            "Start on the completed diagram shown in pic1. The same hand uses a whiteboard "
            "eraser to wipe away the marker lines progressively from left to right. By the "
            "last frame the board is completely blank white and the hand has left the frame. "
            "One continuous physical erasing action, no dissolve or sudden disappearance."
        )
    return common + (
        "Show the scene diagram using pic1 and the actual marker hand from pic2. The hand "
        "draws or points to the important relationship with restrained purposeful motion. "
        "Preserve the diagram structure and exact existing labels throughout. End with the "
        "hand out of the way and a clear completed diagram for reading. "
        f"Scene action: {scene['scene_direction']}"
    )
