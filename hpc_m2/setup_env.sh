#!/bin/bash
# One-time setup. Run on the LOGIN node (it has internet; compute nodes may not):
#     cd hpc_m2 && bash setup_env.sh
# Creates the conda env "ie7615-m2" and pre-downloads everything the job needs:
# YOLOv8n weights, the Ultralytics font and the rembg u2net model.
set -eo pipefail
B="$(cd "$(dirname "$0")" && pwd)"

# Conda: edit this line if `module avail anaconda` shows another name
module load anaconda3/2024.06 2>/dev/null || module load anaconda3 2>/dev/null || true
command -v conda >/dev/null || { echo "conda not found: run 'module avail anaconda' and edit the module line in setup_env.sh and m2_job.sbatch"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV=ie7615-m2
conda env list | grep -q "^$ENV " || conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
# PyTorch first, pinned to a CUDA 12.1 build: the Explorer GPU nodes have driver 545 (CUDA 12.3), and the newest
# PyTorch wheels (CUDA 13) cannot use it; cu121 also still supports the P100/V100 GPUs of courses-gpu.
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# rembg[cpu]: onnxruntime-gpu failed on Colab (cuDNN mismatch); on a node with 8 CPUs the masks take ~15–30 min
pip install -q "ultralytics>=8.3,<9" "rembg[cpu]" ipython pandas matplotlib "torch==2.5.1" "torchvision==0.20.1"

export YOLO_CONFIG_DIR="$B/ycfg"
mkdir -p "$YOLO_CONFIG_DIR" "$B/run" "$B/logs"
cd "$B/run"
python - <<'EOF'
from ultralytics import YOLO
from ultralytics.utils.checks import check_font
YOLO("yolov8n.pt")                      # weights → hpc_m2/run/yolov8n.pt
check_font("Arial.ttf")                 # font → hpc_m2/ycfg/
from rembg import new_session
new_session("u2net")                    # mask model → ~/.u2net or ~/.rembg (home is shared with compute nodes)
import torch, numpy
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| numpy", numpy.__version__,
      "| GPU visible here:", torch.cuda.is_available())
print("setup OK")
EOF
