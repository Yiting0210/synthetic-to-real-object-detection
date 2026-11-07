"""Five ways to answer "this is not one of the 20 course objects".

Every method has the same interface. Given the model outputs for a set of images it returns
    pred    the course class index it would output (0–19)
    score   higher = more like one of the 20 course objects; the image is rejected when score < threshold
    forced  images rejected whatever the threshold is (the 21-class model predicted OTHERS)
so one threshold rule and one evaluation apply to all of them (see openset_eval.py).

| Method | Model | Rejects when |
|---|---|---|
| A | V1, 20 classes          | the highest softmax probability is below the threshold |
| B | V1, 20 classes          | the image feature is far from all 20 class prototypes (max cosine similarity below the threshold) |
| C | V2, 21 classes (OTHERS) | it predicts OTHERS (no threshold) |
| D | V2, 21 classes (OTHERS) | it predicts OTHERS, or the best course-class probability is below the threshold |
| E | V2, 21 classes (OTHERS) | it predicts OTHERS, or the feature is far from all 20 course prototypes |

A class prototype is the mean feature (1280-d, before the classification layer) of the class's training images,
L2-normalized; the similarity of an image to a prototype is the cosine of the angle between them.
B and E still output the softmax class: only the decision to reject uses the prototypes.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelOutputs:
    """Outputs of one model on one set of images."""
    probs: np.ndarray      # (N, C) softmax probabilities
    feats: np.ndarray      # (N, 1280) features before the classification layer


def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def class_prototypes(feats, labels, n_classes):
    """Mean L2-normalized feature of each class, normalized again: one row per class."""
    f = l2_normalize(feats)
    return l2_normalize(np.stack([f[labels == k].mean(0) for k in range(n_classes)]))


def max_cosine(feats, prototypes):
    return (l2_normalize(feats) @ prototypes.T).max(1)


def _none(n):
    return np.zeros(n, dtype=bool)


# ---------------------------------------------------------------- the five methods
def method_a(v1: ModelOutputs, **_):
    p = v1.probs
    return p.argmax(1), p.max(1), _none(len(p))


def method_b(v1: ModelOutputs, prototypes_v1, **_):
    return v1.probs.argmax(1), max_cosine(v1.feats, prototypes_v1), _none(len(v1.probs))


def method_c(v2: ModelOutputs, others_idx, **_):
    p = v2.probs
    forced = p.argmax(1) == others_idx
    return p[:, :others_idx].argmax(1), 1.0 - p[:, others_idx], forced     # score only used for AUROC


def method_d(v2: ModelOutputs, others_idx, **_):
    p = v2.probs
    course = p[:, :others_idx]
    return course.argmax(1), course.max(1), p.argmax(1) == others_idx


def method_e(v2: ModelOutputs, prototypes_v2, others_idx, **_):
    p = v2.probs
    return p[:, :others_idx].argmax(1), max_cosine(v2.feats, prototypes_v2), p.argmax(1) == others_idx


METHODS = {
    "A": dict(fn=method_a, model="V1", uses_threshold=True, label="V1 + softmax threshold"),
    "B": dict(fn=method_b, model="V1", uses_threshold=True, label="V1 + prototype distance"),
    "C": dict(fn=method_c, model="V2", uses_threshold=False, label="V2 OTHERS class only"),
    "D": dict(fn=method_d, model="V2", uses_threshold=True, label="V2 OTHERS + softmax threshold"),
    "E": dict(fn=method_e, model="V2", uses_threshold=True, label="V2 OTHERS + prototype distance"),
}


# ---------------------------------------------------------------- decision, threshold rule, metrics
def rejected(score, forced, threshold):
    return forced | (score < threshold)


def choose_threshold(score_known, forced_known, max_false_rejection=0.01):
    """Largest threshold that rejects at most max_false_rejection of the known (course) images.
    Images forced to OTHERS count as rejections too. Returns -inf if even no threshold exceeds the limit."""
    allowed = int(np.floor(max_false_rejection * len(score_known))) - int(forced_known.sum())
    if allowed < 0:
        return -np.inf
    s = np.sort(score_known[~forced_known])
    return float(s[allowed]) if allowed < len(s) else np.inf      # rejects the `allowed` lowest scores (strictly below)


def ranking_score(score, forced):
    """Single score whose ranking follows the decision rule: forced rejections rank below everything."""
    return np.where(forced, -np.inf, score)


def auroc(score_known, score_unknown):
    """P(known scores higher than unknown) + 0.5·P(tie): area under the ROC curve, threshold-free."""
    k, u = score_known[:, None], score_unknown[None, :]
    return float((k > u).mean() + 0.5 * (k == u).mean())


def auroc_ci(score_known, score_unknown, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [auroc(rng.choice(score_known, len(score_known)), rng.choice(score_unknown, len(score_unknown))) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
