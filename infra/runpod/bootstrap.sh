#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/workspace/talotrace
mkdir -p "$ROOT"/{logs,models,.secrets,artifacts} /opt/talotrace/venvs
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

apt-get update -qq
apt-get install -y -qq --no-install-recommends git ffmpeg espeak-ng libsndfile1 python3-venv
python -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required"'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

if [ ! -d "$ROOT/ComfyUI/.git" ]; then
    git clone https://github.com/Comfy-Org/ComfyUI.git "$ROOT/ComfyUI"
fi
git -C "$ROOT/ComfyUI" checkout efa6c8f804bff78b46a0fd458ebd2e47bba07a30
MSR="$ROOT/ComfyUI/custom_nodes/ComfyUI-LTX2.5-MSR"
if [ ! -d "$MSR/.git" ]; then
    git clone https://github.com/liconstudio/ComfyUI-LTX2.5-MSR.git "$MSR"
fi
git -C "$MSR" checkout 98941179a82223a0a62219550272b2823b3e3a9c

python -m venv --system-site-packages "/opt/talotrace/venvs/comfy"
"/opt/talotrace/venvs/comfy/bin/pip" install -r "$ROOT/ComfyUI/requirements.txt" huggingface_hub supervisor
python -m venv --system-site-packages "/opt/talotrace/venvs/kokoro"
"/opt/talotrace/venvs/kokoro/bin/pip" install 'kokoro==0.9.4' soundfile fastapi uvicorn
"/opt/talotrace/venvs/kokoro/bin/pip" install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
"/opt/talotrace/venvs/comfy/bin/pip" freeze > "$ROOT/logs/comfy-installed.txt"
"/opt/talotrace/venvs/kokoro/bin/pip" freeze > "$ROOT/logs/kokoro-installed.txt"

# Public models first so the TTS endpoint can become usable before the large LTX transfer.
"/opt/talotrace/venvs/comfy/bin/python" "$ROOT/infra/runpod/download_models.py" --only kokoro
"/opt/talotrace/venvs/comfy/bin/python" "$ROOT/infra/runpod/download_models.py" --only msr
if "/opt/talotrace/venvs/comfy/bin/supervisorctl" -c "$ROOT/infra/runpod/supervisord.conf" pid >/dev/null 2>&1; then
    "/opt/talotrace/venvs/comfy/bin/supervisorctl" -c "$ROOT/infra/runpod/supervisord.conf" reread
    "/opt/talotrace/venvs/comfy/bin/supervisorctl" -c "$ROOT/infra/runpod/supervisord.conf" update
else
    "/opt/talotrace/venvs/comfy/bin/supervisord" -c "$ROOT/infra/runpod/supervisord.conf"
fi
"/opt/talotrace/venvs/comfy/bin/python" "$ROOT/infra/runpod/download_models.py" --only ltx
touch "$ROOT/logs/bootstrap-complete"
