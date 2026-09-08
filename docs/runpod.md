# Reproduce the RunPod services

The local FastAPI application reaches two private services through SSH: ComfyUI with
LTX-2.5/MSR for video, and Kokoro for narration. Model files and source persist under
`/workspace/talotrace`; Python environments live on container disk under `/opt/talotrace/venvs`.

## Provision a pod

Use your own RunPod account and current provider quote. The assessment configuration is:

| Resource | Configuration |
| --- | --- |
| GPU | One NVIDIA A100 SXM 80 GB |
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container disk | 50 GB |
| Persistent volume disk | 250 GB, mounted at `/workspace` |
| Public ports | TCP 22 for SSH; no public inference HTTP ports |

Create a dedicated local SSH key from the repository root:

```powershell
New-Item -ItemType Directory -Force .secrets | Out-Null
ssh-keygen -t ed25519 -f .secrets/runpod_ed25519 -C talotrace-assessment-runpod
```

Add the public key to RunPod's SSH configuration and enable SSH for the pod. If the private
key has a passphrase, load it into your SSH agent before using the batch-mode tunnel script.
The private key and `.secrets/` are ignored by Git. Do not overwrite an existing key when
resuming your own setup.

After deployment, use RunPod's current direct SSH connection details to update
`infra/runpod/pod.json`: `pod_id`, `ssh_host` and `ssh_port`. The checked-in values identify
the assessment environment; a new installation must use its own pod. Recheck the address
after any stop/start. The tunnel script reads this JSON file, not the optional `RUNPOD_*`
environment entries.

## Upload configuration and install models

Create `.secrets/hf_token` locally with your Hugging Face read token using a text editor.
Its account must have access to every repository in `infra/runpod/models.lock.json`,
including any applicable model-access terms. The token is read from a file, not passed on
the command line.

Run these PowerShell commands from the repository root:

```powershell
$pod = Get-Content infra/runpod/pod.json -Raw | ConvertFrom-Json
$sshPort = [string]$pod.ssh_port
$sshDestination = "root@$($pod.ssh_host)"
$keyPath = '.secrets/runpod_ed25519'

ssh -i $keyPath -p $sshPort $sshDestination `
    'mkdir -p /workspace/talotrace/infra /workspace/talotrace/.secrets /workspace/talotrace/logs'
scp -i $keyPath -P $sshPort -r infra/runpod "${sshDestination}:/workspace/talotrace/infra/"
scp -i $keyPath -P $sshPort .secrets/hf_token "${sshDestination}:/workspace/talotrace/.secrets/hf_token"
ssh -i $keyPath -p $sshPort $sshDestination `
    'chmod 600 /workspace/talotrace/.secrets/hf_token'
ssh -i $keyPath -p $sshPort $sshDestination `
    'bash /workspace/talotrace/infra/runpod/bootstrap.sh'
```

Keep the bootstrap SSH session open until it finishes. It installs system dependencies,
checks CUDA, checks out pinned ComfyUI/MSR revisions, creates separate Python environments,
downloads the exact model manifest and starts Supervisor. Downloads are roughly 82 GB;
initialization and first model loading can take substantial time.

The model groups are:

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), with the `af_heart` voice.
- [Lightricks LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), including the full BF16 dev
  transformer, text encoder, VAEs, refinement adapter and upscalers.
- [LTX-2.5 Multiple-Subject Reference](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference).

Exact revisions, filenames, sizes and available checksums are in `models.lock.json`.
Download failures stop setup without changing weights or quantizing them. Verification
reports and `bootstrap-complete` are written to `/workspace/talotrace/logs/`.

## Check services and open the tunnel

```powershell
ssh -i $keyPath -p $sshPort $sshDestination `
    '/opt/talotrace/venvs/comfy/bin/supervisorctl -c /workspace/talotrace/infra/runpod/supervisord.conf status'
pwsh -File scripts/runpod-tunnel.ps1 -Check
pwsh -File scripts/runpod-tunnel.ps1
```

Keep the tunnel running in a separate terminal:

| Local endpoint | Remote service |
| --- | --- |
| `http://127.0.0.1:18188` | ComfyUI on `127.0.0.1:8188` |
| `http://127.0.0.1:18081` | Kokoro on `127.0.0.1:8085` |

```powershell
Invoke-RestMethod http://127.0.0.1:18188/system_stats
Invoke-RestMethod http://127.0.0.1:18081/health
```

Wait for the service health response after a restart; Supervisor's `RUNNING` state alone
does not mean model initialization has finished. Kokoro runs on CPU to leave A100 memory
available for full BF16 video generation. Compose uses `host.docker.internal` to reach the
same tunnel, and exposes the learner API only on host loopback port 8000.

## Restart, diagnosis and cleanup

A pod stop/start can clear the container disk. Rerun `bootstrap.sh` to restore the Python
environments; the persistent model/source files remain under `/workspace`. Update the SSH
address and restart the tunnel if the provider assigned new connection details.

Remote logs are in `/workspace/talotrace/logs/`: `comfyui.log`, `comfyui-error.log`,
`kokoro.log` and `kokoro-error.log`. To restart one service, use Supervisor's `restart
comfyui` or `restart kokoro` command with the configuration path shown above. Coordinate
restarts with generation: the local job may require explicit resume afterward.

One GPU render at a time is the operating constraint. The video worker and scene-probe CLI
share a database advisory lock; isolated Comfy experiments must also respect the provider
queue. Keep `VIDEO_WORKER_ENABLED=false` while running a separate GPU experiment.

When finished, stop the local tunnel and stop or terminate the pod in RunPod. Stopped
persistent storage can still bill. Retrieve required artifacts before terminating the pod,
then revoke session tokens and remove the dedicated SSH key when the session is over.
