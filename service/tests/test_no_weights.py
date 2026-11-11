"""Tests that run on a fresh clone (no weights, no MLflow store): class maps, selection record and the loader's error paths.

test_service.py and test_demo_parity.py need trained weights and skip without them; these tests always run.
"""
import hashlib
import json
from pathlib import Path

import pytest

from service.model_loader import Detector, ModelLoadError, load_bundle, load_from_env, read_classes

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"

# models/ holds deployment copies of files whose source of truth is a results folder; they must stay identical
COPIES = [
    (MODELS / "classes.json", REPO / "classification/results/m1/data/classes.json"),
    (MODELS / "classes_with_others.json", REPO / "classification/results/others/data/classes_with_others.json"),
    (MODELS / "others_threshold.json", REPO / "classification/results/others/threshold/threshold.json"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.mark.parametrize("copy,source", COPIES, ids=[c.name for c, _ in COPIES])
def test_models_copies_match_results(copy, source):
    assert sha(copy) == sha(source)


def test_classes_json_is_the_m1_class_map():
    # M2 checks the same prefix (detection/src/m2_detection.py, EXPECTED)
    assert sha(MODELS / "classes.json").startswith("511d30a1")
    classes = read_classes(MODELS / "classes.json")
    assert sorted(classes) == list(range(20))
    assert all(v.startswith("OBJ") for v in classes.values())


def test_classes_with_others_extends_classes():
    base = read_classes(MODELS / "classes.json")
    others = read_classes(MODELS / "classes_with_others.json")
    assert {k: others[k] for k in base} == base
    assert others[20] == "OTHERS" and len(others) == 21


def test_selection_record():
    sel = json.loads((MODELS / "selection.json").read_text(encoding="utf-8"))
    assert 0 < sel["general_threshold"] < 1
    assert len(sel["weights_sha256"]) == 64
    assert sel["registry"]["version"] and sel["registry"]["run_id"]


def test_missing_weights(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        Detector(tmp_path / "missing.pt", {0: "OBJ000"}, conf=0.5)


def test_not_a_bundle(tmp_path):
    with pytest.raises(ModelLoadError, match="no bundle.json"):
        load_bundle(tmp_path)


def write_bundle(d, files_sha256):
    (d / "model.pt").write_bytes(b"not a real model")
    (d / "classes.json").write_text(json.dumps({"0": "OBJ000"}), encoding="utf-8")
    (d / "serving.json").write_text(json.dumps({"weights": "model.pt", "conf_threshold": 0.5}), encoding="utf-8")
    (d / "bundle.json").write_text(json.dumps({"model_name": "test", "version": 1, "files_sha256": files_sha256}),
                                   encoding="utf-8")


def test_bundle_checksum_mismatch(tmp_path):
    write_bundle(tmp_path, {"model.pt": "0" * 64})
    with pytest.raises(ModelLoadError, match="checksum mismatch"):
        load_bundle(tmp_path)


def test_bundle_missing_file(tmp_path):
    write_bundle(tmp_path, {"model.pt": "0" * 64, "extra.bin": "0" * 64})
    (tmp_path / "model.pt").unlink()
    with pytest.raises(ModelLoadError, match="bundle file missing"):
        load_bundle(tmp_path)


def test_env_with_missing_model_path(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_from_env({"MODEL_PATH": str(tmp_path / "missing.pt")})


def test_service_does_not_start_without_model(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from service.app import app
    for k in ("MODEL_BUNDLE", "MODEL_ALIAS", "MODEL_CLASSES", "MODEL_CONF"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(ModelLoadError):
        with TestClient(app):
            pass


# ---------- readiness comparison rule (no model needed) ----------

def _d(cls, conf, box):
    return {"class_id": cls, "class_name": f"OBJ{cls:03d}", "confidence": conf, "box_xyxy": box}


def test_readiness_compare_accepts_same_output():
    from service.model_loader import compare_detections
    exp = [_d(6, 0.95, [10, 10, 100, 100]), _d(2, 0.88, [200, 50, 300, 160])]
    got = [_d(2, 0.87, [201, 51, 301, 161]), _d(6, 0.96, [10, 11, 100, 101])]
    assert compare_detections(got, exp, conf=0.7) == []


def test_readiness_compare_flags_missing_wrong_and_extra():
    from service.model_loader import compare_detections
    exp = [_d(6, 0.95, [10, 10, 100, 100])]
    assert compare_detections([], exp, 0.7)                                         # missing
    assert compare_detections([_d(5, 0.95, [10, 10, 100, 100])], exp, 0.7)          # other class
    assert compare_detections([_d(6, 0.95, [10, 10, 100, 100]), _d(3, 0.9, [0, 0, 5, 5])], exp, 0.7)   # extra, firm


def test_readiness_compare_ignores_detections_next_to_threshold():
    from service.model_loader import compare_detections
    exp = [_d(6, 0.95, [10, 10, 100, 100]), _d(4, 0.72, [0, 0, 50, 50])]            # 0.72 < 0.7 + 0.1: optional
    got = [_d(6, 0.95, [10, 10, 100, 100]), _d(9, 0.75, [300, 300, 340, 340])]      # new 0.75 detection: optional too
    assert compare_detections(got, exp, conf=0.7) == []
