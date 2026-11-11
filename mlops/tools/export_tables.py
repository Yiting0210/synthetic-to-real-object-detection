"""Export the report tables from the MLflow store instead of copying numbers by hand (proposal §7.5).

Reads only the local MLflow store (runs, metrics, Model Registry) and writes mlops/logs/tables_from_mlflow.md:
    1. M1 classifiers                       (experiment M1-classification)
    2. Open-set rejection methods           (experiment openset-rejection, child runs)
    3. M2 ablation: mean ± sample std per arm, then one row per run   (experiment M2-detection)
    4. Real multi-object photos per arm     (test.real.* metrics)
    5. Model Registry: versions, aliases, re-evaluation status

--check compares tables 3 and 4 cell by cell with the summaries generated from the run folders
(detection/results/ablation/README.md, detection/results/real_photos/README.md), at the precision those files use,
and exits with 1 if any cell differs. That shows the MLflow store and the report sources hold the same numbers.

Usage:
    python mlops/tools/export_tables.py [--check]
"""
import argparse
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlops_common import CLASSIFIER_MODEL, DETECTOR_MODEL, LOGS, REPO, client, experiment_id  # noqa: E402

ABLATION_MD = REPO / "detection/results/ablation/README.md"
REAL_MD = REPO / "detection/results/real_photos/README.md"
ARM_NAME = {"A": "arm A (D1)", "B": "arm B (D1+D3)", "C": "arm C (D3)"}

# label in the summary tables → (MLflow metric, scale)
ABLATION_ROWS = [
    ("D1-val mAP50", "val.d1.map50", 1), ("D1-val mAP50-95", "val.d1.map50_95", 1), ("D1-val F1", "val.d1.macro_f1", 1),
    ("hard-val mAP50", "val.hard.map50", 1), ("hard-val mAP50-95", "val.hard.map50_95", 1), ("hard-val F1", "val.hard.macro_f1", 1),
    ("D1-test mAP50", "test.d1.map50", 1), ("D1-test mAP50-95", "test.d1.map50_95", 1), ("D1-test F1", "test.d1.macro_f1", 1),
    ("T-hard mAP50", "test.thard.map50", 1), ("T-hard mAP50-95", "test.thard.map50_95", 1), ("T-hard F1", "test.thard.macro_f1", 1),
    ("val_score", "val.score", 1),
    ("single-object test photos /303", "test.single_object.top1_acc", 303),
    ("single-object val photos /303", "val.single_object.top1_acc", 303),
]
REAL_ROWS = [
    ("AP50 OBJ011", "test.real.ap50_OBJ011"), ("AP50 OBJ002", "test.real.ap50_OBJ002"), ("mAP50 (2 classes)", "test.real.map50_2cls"),
    ("recall OBJ011", "test.real.recall_OBJ011"), ("recall OBJ002", "test.real.recall_OBJ002"), ("precision", "test.real.precision"),
    ("false alarms (other classes)", "test.real.false_alarms_other_classes"),
    ("other FP (bad box / duplicate)", "test.real.other_fp_bad_box_or_duplicate"),
]


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def mean_std(values, p):
    return f"{st.mean(values):.{p}f} ± {st.stdev(values):.{p}f}" if len(values) > 1 else f"{values[0]:.{p}f}"


def m2_runs(c):
    runs = c.search_runs([experiment_id(c, "M2-detection")], max_results=500)
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r.data.tags["arm"], []).append(r)
    return {a: sorted(v, key=lambda r: int(r.data.tags["seed"]), reverse=True) for a, v in sorted(by_arm.items())}


def summary_cells(by_arm, rows, precision, round_inputs=None):
    """{label: {arm: 'mean ± std'}}; precision(label) gives the decimals; round_inputs rounds each run's value first."""
    rnd = (lambda v: round(v, round_inputs)) if round_inputs is not None else (lambda v: v)
    out = {}
    for label, key, *scale in rows:
        s = scale[0] if scale else 1
        out[label] = {a: mean_std([rnd(r.data.metrics[key]) * s for r in rs if key in r.data.metrics], precision(label))
                      for a, rs in by_arm.items() if any(key in r.data.metrics for r in rs)}
    return out


def read_md_table(path, first_header):
    """First markdown table whose first header cell is `first_header`: {row label: [cells]} and the header."""
    lines, rows, header = path.read_text(encoding="utf-8").splitlines(), {}, None
    for line in lines:
        if not line.startswith("|"):
            if header and rows:
                break
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        if header is None:
            if cells[0] == first_header:
                header = cells
        elif not set(cells[0]) <= set("-: "):
            rows[cells[0]] = cells[1:]
    return header, rows


def decimals(cell):
    m = re.match(r"-?\d+(?:\.(\d+))?", cell)
    return len(m.group(1) or "") if m else 3


def check(by_arm):
    """(problems, cells checked, cells equal only after rounding). The ablation summary averages the 4-decimal values
    printed in each run's M2_results_for_report.md, so a cell may differ by one unit in the last digit from the
    full-precision MLflow mean; such a cell counts as equal if rounding the MLflow values to 4 decimals first reproduces it."""
    problems, checked, rounded = [], 0, []
    for path, rows, input_dp in ((ABLATION_MD, ABLATION_ROWS, 4), (REAL_MD, [(l, k) for l, k in REAL_ROWS], None)):
        header, ref = read_md_table(path, "metric")
        arms = [h.split()[1] for h in header[1:]]            # "arm A (D1)" → "A"
        prec = {label: decimals(ref[label][0]) for label, *_ in rows if label in ref}
        ours = summary_cells(by_arm, rows, lambda label: prec.get(label, 3))
        ours_r = summary_cells(by_arm, rows, lambda label: prec.get(label, 3), input_dp) if input_dp else ours
        for label, cells in ref.items():
            if label not in ours:
                continue
            for arm, cell in zip(arms, cells):
                checked += 1
                where = f"{path.relative_to(REPO).as_posix()}: {label}, arm {arm}: source {cell!r}, MLflow {ours[label].get(arm)!r}"
                if ours[label].get(arm) == cell:
                    continue
                if ours_r[label].get(arm) == cell:
                    rounded.append(where)
                else:
                    problems.append(where)
    return problems, checked, rounded


def build(c):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = ["# Report tables exported from MLflow", "",
           f"Generated by `mlops/tools/export_tables.py` from the local MLflow store (`mlops/store/mlflow.db`) on {now}. "
           "Do not edit by hand; re-run the script.", ""]

    # 1. M1
    m1 = sorted(c.search_runs([experiment_id(c, "M1-classification")]), key=lambda r: -r.data.metrics.get("val.accuracy", 0))
    out += ["## 1. M1 classifiers (20 classes, 303 val / 303 test images)", ""]
    out += table(["model", "params", "val acc", "val macro-F1", "test acc", "test macro-F1", "train min", "selected"],
                 [[r.data.params["model"], f"{int(r.data.params['params_total']) / 1e6:.2f} M",
                   f"{r.data.metrics['val.accuracy']:.4f}", f"{r.data.metrics['val.macro_f1']:.4f}",
                   f"{r.data.metrics['test.accuracy']:.4f}", f"{r.data.metrics['test.macro_f1']:.4f}",
                   f"{r.data.metrics['train_minutes']:.1f}", "yes" if r.data.tags.get("selected_as_best") == "True" else ""]
                  for r in m1])

    # 2. open set
    os_runs = [r for r in c.search_runs([experiment_id(c, "openset-rejection")]) if "mlflow.parentRunId" in r.data.tags]
    out += ["", "## 2. Open-set rejection methods (test: 303 known, 15 OTHERS, 36 unseen photos)", ""]
    out += table(["method", "rule", "threshold", "known correct /303", "OTHERS rejected /15", "unseen rejected /36", "AUROC"],
                 [[r.data.params["method"], r.data.params["label"], r.data.params["threshold"],
                   f"{r.data.metrics.get('test.test_known_correct', float('nan')):.0f}",
                   f"{r.data.metrics.get('test.others_test_rejected', float('nan')):.0f}",
                   f"{r.data.metrics.get('test.unseen_rejected', float('nan')):.0f}",
                   f"{r.data.metrics['test.auroc']:.3f}" if "test.auroc" in r.data.metrics else "-"]
                  for r in sorted(os_runs, key=lambda r: r.data.params["method"])])

    # 3. ablation
    by_arm = m2_runs(c)
    seeds = {a: "/".join(r.data.tags["seed"] for r in rs) for a, rs in by_arm.items()}
    summ = summary_cells(by_arm, ABLATION_ROWS, lambda label: 1 if "/303" in label else 3)
    out += ["", "## 3. M2 replacement ablation (mean ± sample std over seeds " + ", ".join(f"{a}: {s}" for a, s in seeds.items()) + ")", ""]
    out += table(["metric"] + [ARM_NAME[a] for a in by_arm], [[label] + [summ[label].get(a, "-") for a in by_arm] for label, *_ in ABLATION_ROWS])
    out += ["", "Per run:", ""]
    out += table(["run", "arm", "seed", "threshold", "D1-test mAP50", "D1-test mAP50-95", "T-hard mAP50", "T-hard mAP50-95", "real mAP50 (2 cls)", "val_score"],
                 [[r.info.run_name, a, r.data.tags["seed"], r.data.params["selection.general_threshold"]] +
                  [f"{r.data.metrics[k]:.3f}" if k in r.data.metrics else "-" for k in
                   ("test.d1.map50", "test.d1.map50_95", "test.thard.map50", "test.thard.map50_95", "test.real.map50_2cls", "val.score")]
                  for a, rs in by_arm.items() for r in rs])

    # 4. real photos
    real = summary_cells(by_arm, REAL_ROWS, lambda label: 1 if label.startswith(("false", "other")) else 3)
    out += ["", "## 4. Real multi-object photos (29 photos, OBJ011 + OBJ002 only; mean ± sample std over seeds)", ""]
    out += table(["metric"] + [ARM_NAME[a] for a in by_arm], [[label] + [real[label].get(a, "-") for a in by_arm] for label, _ in REAL_ROWS])

    # 5. registry
    out += ["", "## 5. Model Registry", ""]
    rows = []
    for name in (DETECTOR_MODEL, CLASSIFIER_MODEL):
        aliases = {}
        for al, v in c.get_registered_model(name).aliases.items():
            aliases.setdefault(str(v), []).append(al)
        for mv in sorted(c.search_model_versions(f"name = '{name}'"), key=lambda m: int(m.version)):
            t = mv.tags
            what = f"arm {t['arm']}, seed {t['seed']}, threshold {t['conf_threshold']}" if name == DETECTOR_MODEL else f"{t.get('model')}, seed {t.get('seed')}"
            reeval = f"{'passed' if t.get('reeval_passed') == 'True' else 'FAILED'} ({t.get('reeval_date')})" if "reeval_passed" in t else "not run"
            rows.append([name, mv.version, what, ", ".join(sorted(aliases.get(str(mv.version), []))) or "-", t.get("source_run_tag"), reeval])
    out += table(["model", "version", "what", "aliases", "source run", "re-evaluation"], rows)
    return out, by_arm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the summaries generated from the run folders")
    a = ap.parse_args(argv)
    c = client()
    lines, by_arm = build(c)
    path = LOGS / "tables_from_mlflow.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {path.relative_to(REPO).as_posix()}")
    if a.check:
        problems, n, rounded = check(by_arm)
        for p in rounded:
            print("  rounding only (source averaged 4-decimal report values) " + p)
        for p in problems:
            print("  DIFF " + p)
        print(f"check: {n - len(problems) - len(rounded)}/{n} cells identical, {len(rounded)} equal after rounding the inputs "
              f"to the reports' 4 decimals, {len(problems)} different")
        if problems:
            sys.exit(1)


if __name__ == "__main__":
    main()
