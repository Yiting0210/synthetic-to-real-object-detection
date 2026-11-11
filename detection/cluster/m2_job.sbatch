#!/bin/bash
# M2 on the cluster: masks → D1 generation → YOLOv8n training → threshold on D1-val → D1-test.
# Submit from inside this folder (uploaded to the cluster as ~/hpc_m2):   cd hpc_m2 && sbatch m2_job.sbatch
# Training augmentation for large objects:                                 M2_TRAIN_AUG=large sbatch m2_job.sbatch
# (sbatch passes the variable on to the job; without it the Ultralytics defaults are used)
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
export M2_WORK_DIR="$B/work/m2_data_${SLURM_JOB_ID:-local}"   # scratch copy per job (arms A and B can run at the same time); emptied at the start of the run
export YOLO_CONFIG_DIR="$B/ycfg"
export MPLBACKEND=Agg

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi || echo "nvidia-smi not available"
cd "$B/run"                                # yolov8n.pt is here
CODE="$B/code/m2_detection.py"             # bundle copy (zip upload as ~/hpc_m2); in a git checkout the code is ../src
if [ -e "$B/../../.git" ]; then            # this folder is detection/cluster/ of a git checkout
  command -v git >/dev/null || { echo "ERROR: git not found; it is needed to record the code commit"; exit 1; }
  # A copy in code/ would run while git_commit names the checkout's HEAD: refuse it
  if [ -e "$CODE" ]; then echo "ERROR: $CODE exists in a git checkout; delete code/ so the job runs ../src/m2_detection.py"; exit 1; fi
  CODE="$B/../src/m2_detection.py"
  # git_commit in the run record must be the code that runs: no uncommitted change to the script
  # (M2_ALLOW_DIRTY=1 runs it anyway; the record then has git_dirty = true). The job never changes the checkout.
  if ! git -C "$B" diff --quiet HEAD -- "$CODE"; then
    [ "${M2_ALLOW_DIRTY:-0}" = 1 ] || { echo "ERROR: $CODE differs from the last commit; commit it or set M2_ALLOW_DIRTY=1"; exit 1; }
  fi
  echo "git checkout at $(git -C "$B" rev-parse HEAD) ($(git -C "$B" status --porcelain | grep -q . && echo 'uncommitted changes' || echo clean))"
  # The M1 split is read from git, so it need not be copied into IE7615_project/; an M2_M1_RUN set before sbatch wins
  M1_IN_GIT="$B/../../classification/results/m1"
  if [ -z "${M2_M1_RUN:-}" ] && [ -f "$M1_IN_GIT/data/split_manifest.csv" ]; then export M2_M1_RUN="$(cd "$M1_IN_GIT" && pwd)"; fi
elif [ ! -f "$CODE" ]; then
  CODE="$B/../src/m2_detection.py"
fi
[ -f "$CODE" ] || { echo "ERROR: m2_detection.py not found ($CODE)"; exit 1; }
export M2_REPO_DIR="$(dirname "$CODE")"
echo "code: $CODE (sha256 $(sha256sum "$CODE" | cut -d' ' -f1)); M1 split: ${M2_M1_RUN:-$M2_PROJECT/runs/m1_20261018_022316}"
python -u "$CODE" && rm -rf "$M2_WORK_DIR"     # the run folder keeps everything needed; the scratch copy only takes space
echo "done $(date); results in $M2_PROJECT/runs/"
