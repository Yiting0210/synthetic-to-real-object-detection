#!/bin/bash
# M2 on the cluster: masks → D1 generation → YOLOv8n training → threshold on D1-val → D1-test.
# Submit from inside this folder (uploaded to the cluster as ~/hpc_m2):   cd hpc_m2 && sbatch m2_job.sbatch
#SBATCH --job-name=m2_yolo
#SBATCH --partition=courses-gpu    # Explorer: course GPU partition (the general `gpu` partition is often full)
#SBATCH --account=ie7615.202710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/m2_%j.out
set -eo pipefail
B="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load anaconda3/2024.06 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
# First run: build the env here on the compute node (the login node kills heavy processes such as conda/pip)
conda env list | grep -q "^ie7615-m2 " || bash "$B/setup_env.sh"
conda activate ie7615-m2
# An env built before the pin has a CUDA-13 PyTorch that this driver (CUDA 12.3) cannot use: replace it
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.5.1+cu121') else 1)" 2>/dev/null; then
  echo "replacing PyTorch with the CUDA 12.1 build"
  pip uninstall -q -y torch torchvision $(pip list --format=freeze 2>/dev/null | grep -i -E "^nvidia-" | cut -d= -f1)
  pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
fi
# Never train on the CPU by accident
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('GPU check:', torch.__version__, torch.cuda.get_device_name(0))"

export M2_DATA_DIR="$B/data/raw"           # shared course folder (images_OBJxxx folders)
export M2_PROJECT="$B/IE7615_project"      # M1 run, mask cache and M2 outputs, same layout as on Drive
export M2_WORK_DIR="$B/work/m2_data"       # scratch copy; emptied at the start of every run
export M2_REPO_DIR="$B/code"
export YOLO_CONFIG_DIR="$B/ycfg"
export MPLBACKEND=Agg

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi || echo "nvidia-smi not available"
cd "$B/run"                                # yolov8n.pt is here
CODE="$B/code/m2_detection.py"             # bundle copy; in the git repo the code lives in detection/src
[ -f "$CODE" ] || CODE="$B/../src/m2_detection.py"
export M2_REPO_DIR="$(dirname "$CODE")"
python -u "$CODE"
echo "done $(date); results in $M2_PROJECT/runs/"
