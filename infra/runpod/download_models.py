"""Download the exact model manifest; report failures without substituting weights."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/talotrace"))
    parser.add_argument("--only", choices=["kokoro", "ltx", "msr"])
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "models.lock.json").read_text())
    secret_file = args.root / ".secrets/hf_token"
    token = secret_file.read_text().strip() if secret_file.exists() else None
    results = []
    for group in manifest["models"]:
        if args.only and args.only != group["key"]:
            continue
        for item in group["files"]:
            destination = args.root / group["directory"]
            print(f"Downloading {group['repo_id']} / {item['path']}", flush=True)
            try:
                file_path = Path(hf_hub_download(
                    repo_id=group["repo_id"], revision=group["revision"],
                    filename=item["path"], local_dir=destination, token=token,
                ))
                if file_path.stat().st_size != item["bytes"]:
                    raise ValueError("Unexpected model file size")
                if item.get("sha256"):
                    with file_path.open("rb") as source:
                        actual = hashlib.file_digest(source, "sha256").hexdigest()
                    if actual != item["sha256"]:
                        raise ValueError("Model SHA256 mismatch")
                if group["key"] == "msr" and file_path.suffix == ".safetensors":
                    link = args.root / "ComfyUI/models/loras" / file_path.name
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if not link.exists():
                        link.symlink_to(file_path)
                results.append({"file": item["path"], "status": "verified"})
                print(f"Verified {item['path']}", flush=True)
            except Exception as error:
                # Provider exceptions can contain signed URLs. Persist only the safe class/status.
                status = getattr(getattr(error, "response", None), "status_code", None)
                results.append({"file": item["path"], "status": "failed",
                                "error_type": type(error).__name__, "http_status": status})
                print(f"Failed {item['path']}: {type(error).__name__}, HTTP {status}", flush=True)
    report = args.root / "logs" / f"models-{args.only or 'all'}-verification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2))
    if any(item["status"] != "verified" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
