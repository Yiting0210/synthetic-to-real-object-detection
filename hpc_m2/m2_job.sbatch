#!/bin/bash
# M2 on the cluster: masks → D1 generation → YOLOv8n training → threshold on D1-val → D1-test.
# Submit from inside hpc_m2:   cd hpc_m2 && sbatch m2_job.sbatch
#SBATCH --job-name=m2_yolo
#SBATCH --partition=gpu            # check with `sinfo -s`; course accounts may need e.g. courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/m2_%j.out
set -eo pipefail
B="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load anaconda3/2024.06 2>/dev/null || module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ie7615-m2

export M2_DATA_DIR="$B/data/raw"           # shared course folder (images_OBJxxx folders)
export M2_PROJECT="$B/IE7615_project"      # M1 run, mask cache and M2 outputs, same layout as on Drive
export M2_WORK_DIR="$B/work/m2_data"       # scratch copy; emptied at the start of every run
export M2_REPO_DIR="$B/code"
export YOLO_CONFIG_DIR="$B/ycfg"
export MPLBACKEND=Agg

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi || echo "nvidia-smi not available"
cd "$B/run"                                # yolov8n.pt is here
CODE="$B/code/m2_detection.py"             # bundle copy; in the git repo the code lives in m2/src
[ -f "$CODE" ] || CODE="$B/../m2/src/m2_detection.py"
export M2_REPO_DIR="$(dirname "$CODE")"
python -u "$CODE"
echo "done $(date); results in $M2_PROJECT/runs/"
