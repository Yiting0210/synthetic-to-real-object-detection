"""Rollback demonstration for the detector (proposal §7.7). Writes mlops/rollback_demo.log.

Steps:
  1. promote: champion → arm B (version 2), previous → arm A (version 1)
  2. export the champion bundle, start the service on it (uvicorn subprocess, MODEL_BUNDLE), check /health and /detect
     on a fixed photo; also start it with MODEL_ALIAS=champion and check it resolves to the same version
  3. rollback: champion → arm A (version 1), previous → arm B (version 2)
  4. export the champion bundle again, restart the service, check that /health and /detect now report version 1
Time from process start to ready is recorded for every start.

Usage:  python mlops/tools/rollback_demo.py [--port 8765]
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bundle import export  # noqa: E402
from mlops_common import DETECTOR_MODEL, LOGS, REPO, STORE, TRACKING_URI, client  # noqa: E402

FIXTURE = REPO / "detection/data/real_photos/images/IMG_2291.jpg"
LOG = LOGS / "rollback_demo.log"
LINES = []


def say(msg=""):
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}" if msg else ""
    print(line, flush=True)
    LINES.append(line)


def versions_by_arm(c):
    return {mv.tags.get("arm"): mv.version for mv in c.search_model_versions(f"name = '{DETECTOR_MODEL}'")}


def set_aliases(c, champion, previous):
    c.set_registered_model_alias(DETECTOR_MODEL, "champion", champion)
    c.set_registered_model_alias(DETECTOR_MODEL, "previous", previous)
    say(f"aliases now: {dict(sorted(c.get_registered_model(DETECTOR_MODEL).aliases.items()))}")


def serve_and_check(port, label, **env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL_BUNDLE", "MODEL_ALIAS", "MODEL_PATH")}
    env.update(env_extra)
    env.setdefault("YOLO_CONFIG_DIR", str(STORE / "_yolo_cfg"))
    mode = ", ".join(f"{k}={Path(v).relative_to(REPO).as_posix() if k == 'MODEL_BUNDLE' else v}" for k, v in env_extra.items()
                     if k.startswith("MODEL_"))
    say(f"{label}: start service ({mode})")
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.app:app", "--port", str(port), "--log-level", "warning"],
                            cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        health = None
        while time.perf_counter() - t0 < 180:
            if proc.poll() is not None:
                raise SystemExit(f"service exited during startup:\n{proc.stderr.read()[-2000:]}")
            try:
                r = httpx.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    health = r.json()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        if health is None:
            raise SystemExit("service not ready within 180 s")
        ready = time.perf_counter() - t0
        m = health["model"]
        say(f"  ready in {ready:.1f} s (process start → /health 200; model load + self-check {health['ready_seconds']:.1f} s)")
        say(f"  /health: model {m['name']} version {m['version']} (arm {m.get('arm')}, seed {m.get('seed')}), "
            f"source={m['source']}, threshold {health['confidence_threshold']}")
        with open(FIXTURE, "rb") as f:
            d = httpx.post(f"{url}/detect", files={"file": (FIXTURE.name, f, "image/jpeg")}, timeout=60).json()
        dets = ", ".join(f"{x['class_name']} {x['confidence']:.2f}" for x in d["detections"])
        say(f"  /detect {FIXTURE.name}: version {d['model']['version']}, {d['n_detections']} detections [{dets}]")
        return health, d, ready
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    c = client()
    by_arm = versions_by_arm(c)
    if not {"A", "B"} <= set(by_arm):
        raise SystemExit("register arm A and arm B first (mlops/tools/register_models.py)")
    vA, vB = by_arm["A"], by_arm["B"]
    say(f"Rollback demo for registered model {DETECTOR_MODEL} (store {TRACKING_URI.replace(REPO.as_posix() + '/', '')})")
    say(f"versions: arm A (D1) = v{vA}, arm B (D1+D3) = v{vB}; aliases before: "
        f"{dict(sorted(c.get_registered_model(DETECTOR_MODEL).aliases.items()))}")
    say()
    say(f"STEP 1 - promote arm B: champion → v{vB}, previous → v{vA}")
    set_aliases(c, vB, vA)
    bundle, mv = export(c, alias="champion")
    say(f"exported champion → {bundle.relative_to(REPO).as_posix()} (version {mv.version})")
    h1, d1, _ = serve_and_check(a.port, "serve B bundle", MODEL_BUNDLE=str(bundle))
    h1a, _, _ = serve_and_check(a.port, "serve by alias", MODEL_ALIAS="champion", MLFLOW_TRACKING_URI=TRACKING_URI)
    assert h1["model"]["version"] == d1["model"]["version"] == h1a["model"]["version"] == str(vB)
    say()
    say(f"STEP 2 - rollback: champion → v{vA}, previous → v{vB}")
    set_aliases(c, vA, vB)
    bundle, mv = export(c, alias="champion")
    say(f"exported champion → {bundle.relative_to(REPO).as_posix()} (version {mv.version})")
    h2, d2, ready = serve_and_check(a.port, "restart on A bundle", MODEL_BUNDLE=str(bundle))
    h2a, _, _ = serve_and_check(a.port, "serve by alias", MODEL_ALIAS="champion", MLFLOW_TRACKING_URI=TRACKING_URI)
    assert h2["model"]["version"] == d2["model"]["version"] == h2a["model"]["version"] == str(vA)
    say()
    say(f"RESULT: rollback OK. Before: champion = v{vB} (arm B, threshold {h1['confidence_threshold']}). "
        f"After: champion = v{vA} (arm A, threshold {h2['confidence_threshold']}); the restarted service reports "
        f"version {h2['model']['version']} in /health and /detect; time to ready after rollback {ready:.1f} s.")
    LOG.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"log written to {LOG.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
