"""Publish a hash-bound reviewed image revision; never call a provider directly."""

import argparse
import copy
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256
from talotrace.config import Settings
from talotrace.jobs.preconditions import require_corrected_corpus, validate_extracted_sources
from talotrace.jobs.schemas import Direction, validate_grounding
from talotrace.jobs.worker import pipeline_settings


class RevisionRefused(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RevisionRefused(message)


def digest(value):
    return sha256(json_bytes(value))


def frame_identity(scene, settings):
    return digest({"scene": scene, "settings": settings})


def validate_revision(original, proposal, approved_hash, settings, artifacts, active, sources):
    """Validate every source and original receipt before preparing any new files."""
    require(digest(proposal) == approved_hash, "Patch differs from the approved proposal hash")
    require(
        proposal.get("schema_version") == "image-correction-proposal-v1", "Unknown patch schema"
    )
    original_id = str(original["id"])
    require(proposal.get("keyframe_job_id") == original_id, "Patch names another source job")
    require(original["state"] == "keyframes_completed", "Source job is not completed")
    require(original["settings"] == settings, "Source settings or reference hashes changed")
    require(proposal.get("preserve_originals") is True, "Revision must preserve originals")
    require(
        proposal.get("generation_requirements")
        == {
            "model": settings["image_model"],
            "quality": settings["image_quality"],
            "size": settings["image_size"],
            "attach_original_style_anchor": True,
        },
        "Patch generation requirements differ from the worker",
    )
    context, direction = original["context"], original["direction"]
    require(active is not None, "No active corpus")
    require_corrected_corpus(active["summary"])
    validate_extracted_sources(context)
    require(context["corpus_id"] == active["id"], "Source corpus is no longer active")
    for name, value in (("context.json", context), ("direction.json", direction)):
        require(artifacts.read_json(original_id, name) == value, f"Source {name} differs from DB")
    direction_hash = digest(direction)
    require(
        direction_hash == proposal["original_direction_sha256"], "Source direction hash changed"
    )
    source_map = {row["id"]: row for row in sources}
    for hit in context["results"]:
        current = source_map.get(hit["id"])
        require(current is not None, "A cited chunk is absent from the active corpus")
        for field in (
            "content",
            "file_sha256",
            "path",
            "page_number",
            "extractor",
            "quality_flags",
        ):
            require(hit.get(field) == current.get(field), f"Source chunk {field} changed")
        require(hit.get("content_sha256") == sha256(hit["content"].encode()), "Chunk hash changed")
    plan = Direction.model_validate(direction["plan"])
    validate_grounding(plan, context, original["request"]["keyframe_count"])
    receipts = {}
    require(len(original["frames"]) == len(plan.scenes), "Source does not have all frame receipts")
    for scene, saved in zip(plan.scenes, original["frames"], strict=True):
        receipt = artifacts.frame_receipt(
            original_id, scene.order, frame_identity(scene.model_dump(), settings)
        )
        require(receipt is not None and receipt == saved, "Original frame receipt differs from DB")
        require(receipt["source_ids"] == scene.source_ids, "Original frame citations differ")
        require(
            settings["style_anchor"] in receipt["references"], "Original style reference changed"
        )
        receipts[scene.order] = receipt
    effective = copy.deepcopy(plan.model_dump())
    changes = proposal.get("corrections", [])
    orders = [change["scene_order"] for change in changes]
    require(bool(orders) and len(set(orders)) == len(orders), "Duplicate or empty scene changes")
    require(set(orders) < set(receipts), "Revision must retain at least one original frame")
    allowed = {
        "scene_order",
        "original_direction_sha256",
        "original_image_sha256",
        "original_scene",
        "original_image_prompt",
        "reason",
        "narration_affected",
        "narration_review",
        "source_ids",
        "replacement_scene_direction",
        "replacement_image_prompt",
        "replacement_visible_labels",
    }
    for change in changes:
        require(set(change) == allowed, "Patch contains unsupported fields")
        order = change["scene_order"]
        scene = effective["scenes"][order - 1]
        require(change["original_scene"] == scene, "Patch was reviewed against a different scene")
        require(
            change["original_direction_sha256"] == direction_hash, "Patch direction hash changed"
        )
        require(
            change["original_image_sha256"] == receipts[order]["image"]["sha256"],
            "Patch image hash changed",
        )
        require(
            change["original_image_prompt"] == receipts[order]["prompt"], "Original prompt changed"
        )
        require(change["source_ids"] == scene["source_ids"], "Patch citations changed")
        require(change["narration_affected"] is False, "Narration changes need a separate review")
        for field in ("scene_direction", "image_prompt", "visible_labels"):
            scene[field] = change[f"replacement_{field}"]
    revised_plan = Direction.model_validate(effective)
    validate_grounding(revised_plan, context, len(plan.scenes))
    lineage = {
        "schema_version": "keyframe-revision-v1",
        "source_job_id": original_id,
        "original_direction_sha256": direction_hash,
        "context_sha256": digest(context),
        "approved_proposal_sha256": approved_hash,
        "approved_by": "root",
        "generated_scene_orders": sorted(orders),
        "reused_scene_orders": sorted(set(receipts) - set(orders)),
        "original_frame_receipt_sha256": {str(k): digest(v) for k, v in receipts.items()},
        "narration_and_source_ids_unchanged": True,
    }
    revised_direction = {
        "plan": revised_plan.model_dump(),
        "direction_origin": "reviewed_image_revision",
        "original_provider_output": copy.deepcopy(direction),
        "reviewed_revision": lineage,
        "model": direction.get("model"),
        "request_id": None,
        "usage": None,
    }
    return revised_direction, lineage, receipts


def seed_artifacts(artifacts, new_id, original, proposal, direction, lineage, receipts):
    """Only uncommitted/new job files are written. Retrying a failed transaction is safe."""
    directory = artifacts.directory(new_id)
    existing = artifacts.read_json(new_id, "revision.json")
    if any(directory.iterdir()):
        require(existing == lineage, "Revision directory has a different or missing lineage")
    artifacts.write_json(new_id, "revision.json", lineage)
    artifacts.write_json(new_id, "approved-image-corrections.json", proposal)
    artifacts.write_json(new_id, "context.json", original["context"])
    artifacts.write_json(new_id, "direction.json", direction)
    artifacts.write_json(
        new_id,
        "request.json",
        {
            "job_id": new_id,
            "request": original["request"],
            "settings": original["settings"],
        },
    )
    for order in lineage["reused_scene_orders"]:
        receipt = copy.deepcopy(receipts[order])
        receipt["artifact_url"] = f"/v1/keyframe-jobs/{new_id}/images/{order}"
        receipt["reused_from"] = {
            "job_id": str(original["id"]),
            "scene_order": order,
            "receipt_sha256": digest(receipts[order]),
            "provider_usage_is_original": True,
        }
        data = artifacts.path(str(original["id"]), f"scene-{order:02d}.png").read_bytes()
        artifacts.write(new_id, f"scene-{order:02d}.png", data)
        artifacts.write_json(new_id, f"scene-{order:02d}.json", receipt)
        artifacts.frame_receipt(new_id, order, receipt["input_sha256"])
    for order in lineage["generated_scene_orders"]:
        require(
            not artifacts.path(new_id, f"scene-{order:02d}.json").exists(),
            "An unpublished revision unexpectedly has a changed-scene receipt",
        )


def revise(settings, source_id, proposal, approved_hash, key, queue, after_jobs):
    require(8 <= len(key) <= 128, "Idempotency key must have 8 to 128 characters")
    require(digest(proposal) == approved_hash, "Patch differs from the approved proposal hash")
    new_id = str(uuid5(NAMESPACE_URL, "talotrace:keyframe-revision:" + key))
    require(new_id != source_id, "Revision must have a new job ID")
    artifacts = ArtifactStore(settings.artifacts_dir)
    with psycopg.connect(settings.database_url.get_secret_value(), row_factory=dict_row) as conn:
        # Hold publication until files are complete; the worker cannot see an uncommitted row.
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))
        existing = conn.execute(
            "SELECT * FROM pipeline.jobs WHERE idempotency_key=%s", (key,)
        ).fetchone()
        if existing:
            lineage = (existing["direction"] or {}).get("reviewed_revision", {})
            require(
                str(existing["id"]) == new_id
                and lineage.get("approved_proposal_sha256") == approved_hash
                and lineage.get("source_job_id") == source_id,
                "Idempotency key is already bound to another revision",
            )
            return {"job_id": new_id, "state": existing["state"], "replayed": True, **lineage}
        original = conn.execute(
            "SELECT * FROM pipeline.jobs WHERE id=%s FOR SHARE", (source_id,)
        ).fetchone()
        require(original is not None, "Source job does not exist")
        active = conn.execute(
            "SELECT id,summary FROM rag.corpora WHERE active AND state='ready' FOR SHARE"
        ).fetchone()
        sources = conn.execute(
            """SELECT c.id,c.content,d.path,d.file_sha256,c.page_number,p.quality_flags,
            p.extraction_metadata->>'extractor' AS extractor FROM rag.chunks c
            JOIN rag.corpus_chunks cc ON cc.chunk_id=c.id
            JOIN rag.documents d ON d.id=c.document_id
            JOIN rag.pages p ON p.document_id=c.document_id AND p.page_number=c.page_number
            WHERE cc.corpus_id=%s AND c.id=ANY(%s)""",
            (active["id"] if active else "", [hit["id"] for hit in original["context"]["results"]]),
        ).fetchall()
        question_hash = sha256(original["request"]["question"].encode())
        embedding = conn.execute(
            """SELECT 1 FROM rag.embeddings WHERE input_sha256=%s
            AND model='text-embedding-3-large' AND dimensions=3072
            AND vector_dims(embedding)=3072""",
            (question_hash,),
        ).fetchone()
        require(embedding is not None, "Original question embedding is unavailable")
        direction, lineage, receipts = validate_revision(
            original,
            proposal,
            approved_hash,
            pipeline_settings(settings),
            artifacts,
            active,
            sources,
        )
        for prerequisite in after_jobs:
            row = conn.execute(
                "SELECT state FROM pipeline.jobs WHERE id=%s FOR SHARE", (prerequisite,)
            ).fetchone()
            require(
                row and row["state"] == "keyframes_completed", "Prerequisite job is not complete"
            )
        result = {
            "job_id": new_id,
            "state": "queued" if queue else "validated_dry_run",
            "replayed": False,
            **lineage,
        }
        if queue:
            seed_artifacts(artifacts, new_id, original, proposal, direction, lineage, receipts)
            conn.execute(
                """INSERT INTO pipeline.jobs
                (id,idempotency_key,request_sha256,request,settings,context,direction)
                VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (
                    new_id,
                    key,
                    digest({"request": original["request"], "settings": original["settings"]}),
                    Jsonb(original["request"]),
                    Jsonb(original["settings"]),
                    Jsonb(original["context"]),
                    Jsonb(direction),
                ),
            )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-job", required=True, type=UUID)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--approved-proposal-sha256", required=True)
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--after-job", action="append", default=[], type=UUID)
    parser.add_argument("--queue", action="store_true", help="Publish for paid worker generation")
    args = parser.parse_args()
    try:
        result = revise(
            Settings(),
            str(args.source_job),
            json.loads(args.proposal.read_text("utf-8")),
            args.approved_proposal_sha256,
            args.idempotency_key,
            args.queue,
            args.after_job,
        )
    except RevisionRefused as error:
        parser.exit(2, f"Revision refused: {error}\n")
    except Exception as error:
        # Connection/provider details may contain credentials; report only the category.
        parser.exit(2, f"Revision failed: {type(error).__name__}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
