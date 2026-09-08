"""Recoverable private Comfy submissions. Ambiguous submissions never silently duplicate."""

import asyncio
import time
from pathlib import Path

import httpx

from talotrace.artifacts.store import ArtifactStore, json_bytes, sha256


class ComfyExecutionError(RuntimeError):
    pass


class AmbiguousSubmission(RuntimeError):
    pass


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        artifacts: ArtifactStore,
        job_id: str,
        timeout_seconds: int = 21600,
        poll_seconds: float = 5.0,
    ):
        self.base_url = base_url
        self.artifacts = artifacts
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def client(self):
        return httpx.AsyncClient(base_url=self.base_url, timeout=120, trust_env=False)

    async def upload(self, filename: str, data: bytes) -> str:
        digest = sha256(data)
        name = f"{digest}-{filename}"
        receipt_name = f"upload-{digest}.json"
        async with self.client() as client:
            saved = self.artifacts.read_json(self.job_id, receipt_name)
            if saved:
                if saved["sha256"] != digest:
                    raise ValueError("Saved reference upload input changed")
                value = saved["provider"]
            else:
                response = await client.post(
                    "/upload/image",
                    files={"image": (name, data, "image/png")},
                    data={"type": "input", "subfolder": "talotrace", "overwrite": "false"},
                )
                response.raise_for_status()
                value = response.json()
            if value.get("type") != "input" or value.get("subfolder") != "talotrace":
                raise ValueError("Unexpected Comfy upload location")
            if Path(value["name"]).name != value["name"] or "\\" in value["name"]:
                raise ValueError("Unsafe upload response")
            response = await client.get(
                "/view",
                params={"type": "input", "subfolder": "talotrace", "filename": value["name"]},
            )
            response.raise_for_status()
            if sha256(response.content) != digest:
                raise ValueError("Comfy uploaded reference bytes differ from the original")
            self.artifacts.write_json(
                self.job_id, receipt_name, {"sha256": digest, "provider": value}
            )
            return f"talotrace/{value['name']}"

    async def find_dispatch(self, client, dispatch_id: str) -> str | None:
        response = await client.get("/queue")
        response.raise_for_status()
        queued = response.json()
        for row in queued.get("queue_running", []) + queued.get("queue_pending", []):
            if row[3].get("client_id") == dispatch_id:
                return row[1]
        response = await client.get("/history", params={"max_items": 200})
        response.raise_for_status()
        for prompt_id, value in response.json().items():
            if value.get("prompt", [None] * 4)[3].get("client_id") == dispatch_id:
                return prompt_id
        return None

    async def render(self, name: str, graph: dict, attempt: int = 1) -> tuple[bytes, dict]:
        identity = sha256(json_bytes(graph))
        filename = f"{name}-submission.json"
        record = self.artifacts.read_json(self.job_id, filename)
        if record:
            confirmed_failure = record.get("state") in {"validation_rejected", "execution_failed"}
            if record["graph_sha256"] != identity and not confirmed_failure:
                raise ValueError("Saved Comfy graph changed")
            if confirmed_failure and attempt <= record.get("job_attempt", 1):
                raise ComfyExecutionError("Confirmed provider failure requires explicit job resume")
            previous_graph = self.artifacts.read_json(self.job_id, f"{name}-graph.json")
            if previous_graph:
                previous_hash = sha256(json_bytes(previous_graph))
                self.artifacts.write_json(
                    self.job_id, f"{name}-graph-{previous_hash}.json", previous_graph
                )
        self.artifacts.write_json(self.job_id, f"{name}-graph-{identity}.json", graph)
        self.artifacts.write_json(self.job_id, f"{name}-graph.json", graph)
        async with self.client() as client:
            previous = []
            if record and record.get("state") in {"validation_rejected", "execution_failed"}:
                if attempt <= record.get("job_attempt", 1):
                    raise ComfyExecutionError(
                        "Confirmed provider failure requires explicit job resume"
                    )
                # Retain every confirmed failed dispatch; only a new job attempt may resubmit.
                previous = record.get("previous_dispatches", []) + [
                    {key: value for key, value in record.items() if key != "previous_dispatches"}
                ]
                record = None
            if record:
                if record["graph_sha256"] != identity:
                    raise ValueError("Saved Comfy graph changed")
                prompt_id = record.get("prompt_id")
                if not prompt_id:
                    prompt_id = await self.find_dispatch(client, record["dispatch_id"])
                    if not prompt_id:
                        raise AmbiguousSubmission(
                            "Submission receipt is ambiguous; inspect provider before retrying"
                        )
            else:
                # Other experiments may use the same GPU. Never interrupt or jump their queue.
                waiting = time.monotonic()
                while True:
                    response = await client.get("/queue")
                    response.raise_for_status()
                    queue = response.json()
                    if not queue.get("queue_running") and not queue.get("queue_pending"):
                        break
                    if time.monotonic() - waiting > self.timeout_seconds:
                        raise TimeoutError("Waiting for the shared GPU")
                    await asyncio.sleep(self.poll_seconds)
                # Persist before the HTTP call, including the stable client_id used for discovery.
                record = {
                    "graph_sha256": identity,
                    "dispatch_id": f"talotrace:{self.job_id}:{name}:{identity[:12]}:{attempt}",
                    "state": "dispatching",
                    "job_attempt": attempt,
                    "previous_dispatches": previous,
                }
                self.artifacts.write_json(self.job_id, filename, record)
                response = await client.post(
                    "/prompt", json={"prompt": graph, "client_id": record["dispatch_id"]}
                )
                if response.status_code == 400:
                    # Validation rejects before execution; save safe diagnostics for developers.
                    value = response.json()
                    record.update(state="validation_rejected", node_errors=value.get("node_errors"))
                    self.artifacts.write_json(self.job_id, filename, record)
                response.raise_for_status()
                prompt_id = response.json()["prompt_id"]
            record.update(prompt_id=prompt_id, state="submitted")
            self.artifacts.write_json(self.job_id, filename, record)
            started = time.monotonic()
            while time.monotonic() - started < self.timeout_seconds:
                response = await client.get(f"/history/{prompt_id}")
                response.raise_for_status()
                history = response.json().get(prompt_id)
                if history:
                    status = history.get("status", {})
                    if status.get("status_str") == "error":
                        record.update(state="execution_failed")
                        self.artifacts.write_json(self.job_id, filename, record)
                        details = {"prompt_id": prompt_id, "status": status}
                        self.artifacts.write_json(
                            self.job_id, f"{name}-provider-error.json", details
                        )
                        self.artifacts.write_json(
                            self.job_id, f"{name}-provider-error-{prompt_id}.json", details
                        )
                        raise ComfyExecutionError(
                            "Comfy execution failed; saved provider diagnostic"
                        )
                    if status.get("completed"):
                        if status.get("status_str") != "success":
                            raise ComfyExecutionError("Comfy did not complete successfully")
                        output = self.video_output(history.get("outputs", {}))
                        response = await client.get("/view", params=output)
                        response.raise_for_status()
                        data = response.content
                        if record.get("sha256") and record["sha256"] != sha256(data):
                            raise ValueError("Previously completed Comfy output changed")
                        record.update(
                            state="completed", output=output, sha256=sha256(data), bytes=len(data)
                        )
                        self.artifacts.write_json(self.job_id, filename, record)
                        return data, record
                await asyncio.sleep(self.poll_seconds)
        raise TimeoutError("Comfy remains unresolved; resume polls the saved prompt ID")

    @staticmethod
    def video_output(outputs: dict) -> dict:
        found = []

        def visit(value):
            if isinstance(value, dict):
                if str(value.get("filename", "")).endswith(".mp4"):
                    found.append(value)
                else:
                    for item in value.values():
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(outputs)
        if len(found) != 1:
            raise ValueError("Expected exactly one Comfy MP4 output")
        output = found[0]
        filename = output["filename"]
        if Path(filename).name != filename or "\\" in filename or output.get("type") != "output":
            raise ValueError("Unsafe Comfy output metadata")
        folder = output.get("subfolder", "")
        if ".." in folder.replace("\\", "/").split("/"):
            raise ValueError("Unsafe Comfy output folder")
        return {"filename": filename, "subfolder": folder, "type": "output"}
