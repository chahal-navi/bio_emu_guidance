#!/usr/bin/env bash
# One-time setup on a RunPod (or any Linux CUDA) machine. Keeps everything on /workspace
# (the persistent volume) so weights/embeddings survive pod restarts.
#   bash setup_runpod.sh
set -euo pipefail
WS=${WS:-/workspace}
BRANCH=${BRANCH:-membrane-guidance-experiments}
cd "$WS"
if [ ! -d bio_emu_guidance ]; then
  git clone -b "$BRANCH" https://github.com/chahal-navi/bio_emu_guidance.git
fi
cd bio_emu_guidance && git fetch && git checkout "$BRANCH" && git pull --ff-only

# Persist HOME-relative caches (AF2 params ~3.5 GB in ~/.cache/colabfold, BioEmu
# checkpoints via huggingface, embeddings in ~/.bioemu_embeds_cache, SO3 tables).
mkdir -p "$WS/home_cache"
for d in .cache .bioemu_embeds_cache sampling_so3_cache; do
  if [ ! -L "$HOME/$d" ]; then
    mkdir -p "$WS/home_cache/$d"
    [ -d "$HOME/$d" ] && cp -rn "$HOME/$d/." "$WS/home_cache/$d/" && rm -rf "$HOME/$d"
    ln -s "$WS/home_cache/$d" "$HOME/$d"
  fi
done

command -v uv >/dev/null || pip install -q uv
[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[cuda]" pandas matplotlib pytest
python - <<'PY'
import torch, jax
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("jax devices:", jax.devices())
PY
python -m pytest -q tests/steering/test_membrane_guidance.py
echo "Setup OK. Activate with: source $WS/bio_emu_guidance/.venv/bin/activate"
