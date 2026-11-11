"""Shared paths and helpers for the MLflow tools (import_runs, register_models, export_bundle, rollback_demo).

The tracking store lives under mlops/store/ in the repository and is gitignored:
    mlops/store/mlflow.db       SQLite backend (runs, metrics, Model Registry)
    mlops/store/mlartifacts/    artifact root (weights, figures, reports)
mlops/logs/ (in git) holds run_id_map.csv and rollback_demo.log.
It must stay on a local disk (SQLite locking is unreliable on Drive or network mounts).
"""
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "mlops" / "store"
LOGS = REPO / "mlops" / "logs"
DB = STORE / "mlflow.db"
ARTIFACT_ROOT = STORE / "mlartifacts"
TRACKING_URI = "sqlite:///" + DB.as_posix()
BUNDLES = REPO / "model_bundles"

DETECTOR_MODEL = "lab2real-detector"          # registered model name for the YOLOv8n detector
CLASSIFIER_MODEL = "lab2real-classifier"      # registered model name for the M1 Keras classifier
RUNS = REPO / "runs"                           # run folders downloaded from the cluster (not in git)
READINESS_SUBDIR = "readiness"                # model/readiness/: development image + expected output for the startup check
MODEL_SUBDIR = "model"                        # run artifact folder that a model version points to


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def client():
    """MlflowClient on the local store (creates mlops/ on first use)."""
    import mlflow
    from mlflow import MlflowClient
    STORE.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)


def experiment_id(c, name):
    exp = c.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return c.create_experiment(name, artifact_location=(ARTIFACT_ROOT / name).as_uri())
