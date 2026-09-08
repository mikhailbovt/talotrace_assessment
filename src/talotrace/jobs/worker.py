"""One bounded worker across API processes, with explicit recovery and artifact reuse."""

import asyncio
import logging

from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.config import Settings
from talotrace.jobs.preconditions import require_corrected_corpus, validate_extracted_sources
from talotrace.jobs.schemas import Direction, validate_grounding
from talotrace.jobs.store import JobStore
from talotrace.providers.openai_media import (
    direct,
    generate_frame,
    make_client,
    reference_metadata,
    safe_error,
)
from talotrace.retrieval import HybridRetriever

log = logging.getLogger(__name__)
WORKER_LOCK = 472930402
PIPELINE_VERSION = "keyframes-v1"


def pipeline_settings(settings: Settings) -> dict:
    if (
        settings.openai_director_model != "gpt-5.6-luna"
        or settings.openai_image_model != "gpt-image-2"
        or settings.openai_embedding_model != "text-embedding-3-large"
    ):
        raise ValueError("This assessment slice requires the explicitly selected OpenAI models")
    return {
        "pipeline_version": PIPELINE_VERSION,
        "director_model": settings.openai_director_model,
        "image_model": settings.openai_image_model,
        "embedding_model": settings.openai_embedding_model,
        "embedding_dimensions": 3072,
        "style_anchor": reference_metadata(settings.style_anchor_path, "style_anchor"),
        "hand_reference": reference_metadata(settings.hand_reference_path, "hand_reference"),
        "image_size": "2048x1152",
        "image_quality": "high",
    }


class KeyframePipeline:
    def __init__(self, settings: Settings, store: JobStore, artifacts: ArtifactStore):
        self.settings = settings
        self.store = store
        self.artifacts = artifacts

    async def run(self, job: dict) -> None:
        job_id = str(job["id"])
        stage = "preflight"
        scene_order = None
        try:
            current = pipeline_settings(self.settings)
            if current != job["settings"]:
                raise ValueError("Job settings or reference files changed; create a new job")
            self.artifacts.write_json(
                job_id,
                "request.json",
                {
                    "job_id": job_id,
                    "request": job["request"],
                    "settings": job["settings"],
                },
            )
            async with make_client(self.settings) as client:
                context = job["context"] or self.artifacts.read_json(job_id, "context.json")
                if context is None:
                    stage = "retrieving"
                    await self.store.update(job_id, state="retrieving")
                    async with await self.store.connect() as conn:
                        active = await (
                            await conn.execute(
                                "SELECT summary FROM rag.corpora WHERE active AND state='ready'"
                            )
                        ).fetchone()
                        require_corrected_corpus(active["summary"] if active else None)
                        query_hash = sha256(job["request"]["question"].encode())
                        cached = await (
                            await conn.execute(
                                """SELECT 1 FROM rag.embeddings WHERE input_sha256=%s
                            AND model=%s AND dimensions=3072""",
                                (query_hash, current["embedding_model"]),
                            )
                        ).fetchone()
                        retriever = HybridRetriever(conn, client)
                        context = await retriever.search(job["request"]["question"], top_k=8)
                        corpus = await (
                            await conn.execute(
                                "SELECT summary FROM rag.corpora WHERE id=%s",
                                (context["corpus_id"],),
                            )
                        ).fetchone()
                        context["corpus_summary"] = corpus["summary"]
                        context["question_embedding"] = {
                            "input_sha256": query_hash,
                            "model": current["embedding_model"],
                            "dimensions": 3072,
                            "cache_hit": bool(cached),
                            "storage": "rag.embeddings",
                        }
                    for hit in context["results"]:
                        hit["content_sha256"] = sha256(hit["content"].encode())
                    context["embedding_dimensions"] = 3072
                    if not context["results"]:
                        raise ValueError("Retrieval produced no grounding sources")
                    self.artifacts.write_json(job_id, "context.json", context)
                stage = "retrieval_validation"
                validate_extracted_sources(context)
                await self.store.update(job_id, context=context)

                direction = job["direction"] or self.artifacts.read_json(job_id, "direction.json")
                if direction is None:
                    stage = "directing"
                    await self.store.update(job_id, state="directing")
                    direction = await direct(
                        client, current["director_model"], context, job["request"]["keyframe_count"]
                    )
                    self.artifacts.write_json(job_id, "direction.json", direction)
                stage = "direction_validation"
                plan = Direction.model_validate(direction["plan"])
                validate_grounding(plan, context, job["request"]["keyframe_count"])
                await self.store.update(job_id, direction=direction, state="generating_keyframes")

                frames = []
                for scene in plan.scenes:
                    stage = "generating_keyframes"
                    scene_order = scene.order
                    identity = sha256(
                        json_bytes({"scene": scene.model_dump(), "settings": current})
                    )
                    receipt = self.artifacts.frame_receipt(job_id, scene.order, identity)
                    if receipt is None:
                        data, receipt = await generate_frame(
                            client,
                            current["image_model"],
                            scene,
                            self.settings.style_anchor_path,
                            self.settings.hand_reference_path,
                        )
                        receipt["input_sha256"] = identity
                        receipt["source_ids"] = scene.source_ids
                        receipt["artifact_url"] = f"/v1/keyframe-jobs/{job_id}/images/{scene.order}"
                        self.artifacts.write(job_id, f"scene-{scene.order:02d}.png", data)
                        self.artifacts.write_json(job_id, f"scene-{scene.order:02d}.json", receipt)
                    frames.append(receipt)
                    await self.store.update(job_id, frames=frames)
                stage = "persisting_result"
                scene_order = None
                manifest = {
                    "job_id": job_id,
                    "state": "keyframes_completed",
                    "video_generated": False,
                    "timing_basis": "estimated_until_tts",
                    "request": job["request"],
                    "settings": current,
                    "corpus_id": context["corpus_id"],
                    "context_sha256": sha256(json_bytes(context)),
                    "direction_sha256": sha256(json_bytes(direction)),
                    "frames": frames,
                    "visual_review": "pending_human_or_agent_inspection",
                }
                self.artifacts.write_json(job_id, "manifest.json", manifest)
                await self.store.update(job_id, state="keyframes_completed", error=None)
        except asyncio.CancelledError:
            await self.store.update(
                job_id,
                state="interrupted",
                error={
                    "category": "WorkerInterrupted",
                    "failed_stage": stage,
                    "scene_order": scene_order,
                    "detail": "Saved outputs are retained; unfinished calls may bill on resume",
                },
            )
            raise
        except Exception as error:
            details = safe_error(error)
            details["failed_stage"] = stage
            if scene_order is not None:
                details["scene_order"] = scene_order
            self.artifacts.write_json(job_id, "last-error.json", details)
            await self.store.update(job_id, state="failed", error=details)
            log.warning("Keyframe job %s failed: %s", job_id, details["category"])


class JobWorker:
    def __init__(self, pipeline: KeyframePipeline):
        self.pipeline = pipeline
        self.store = pipeline.store
        self.task = None
        self.active = False
        self.stopping = asyncio.Event()

    def start(self):
        self.task = asyncio.create_task(self.run(), name="keyframe-job-worker")

    async def stop(self):
        self.stopping.set()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def run(self):
        while not self.stopping.is_set():
            try:
                async with await self.store.connect() as conn:
                    row = await (
                        await conn.execute(
                            "SELECT pg_try_advisory_lock(%s) AS acquired", (WORKER_LOCK,)
                        )
                    ).fetchone()
                    if not row["acquired"]:
                        await asyncio.sleep(3)
                        continue
                    self.active = True
                    await self.store.mark_interrupted(conn)
                    while not self.stopping.is_set():
                        job = await self.store.claim(conn)
                        if job:
                            # Losing the lock connection must cancel provider work before takeover.
                            execution = asyncio.create_task(self.pipeline.run(job))
                            try:
                                while not execution.done():
                                    done, _ = await asyncio.wait({execution}, timeout=5)
                                    if not done:
                                        await conn.execute("SELECT 1")
                                await execution
                            finally:
                                if not execution.done():
                                    execution.cancel()
                                    try:
                                        await execution
                                    except asyncio.CancelledError:
                                        pass
                        else:
                            await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                log.warning("Job worker unavailable: %s", type(error).__name__)
                await asyncio.sleep(3)
            finally:
                self.active = False
