"""Pure learning-to-rank math: logistic model + ranking metrics.

No DB, no third-party ML framework — a small pointwise logistic regression
trained by gradient descent over the score-breakdown features. This keeps the
image light while still *learning from feedback* instead of hardcoding weights.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Feature schema v2 intentionally excludes top-level rule_score and semantic_score.
# Those signals are already blended directly into final_score; using them again
# inside ml_rank double-counted the same evidence.
FEATURE_SCHEMA_VERSION = "ltr-v2-nonoverlap"
FEATURE_KEYS = [
    "experience_project_score",
    "education_language_certification_score",
    "completeness_score",
    "confidence_score",
]


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def features_from_breakdown(breakdown: dict, feature_keys: List[str] | None = None) -> List[float]:
    keys = feature_keys or FEATURE_KEYS
    return [max(0.0, min(1.0, float(breakdown.get(key, 0) or 0) / 100.0)) for key in keys]


def is_model_compatible(model: dict) -> bool:
    return (
        model.get("feature_schema_version") == FEATURE_SCHEMA_VERSION
        and model.get("feature_keys") == FEATURE_KEYS
        and len(model.get("weights") or []) == len(FEATURE_KEYS)
    )


def predict_ml_rank(model: dict, breakdown: dict) -> Optional[int]:
    if not is_model_compatible(model):
        return None
    weights = model.get("weights") or []
    bias = float(model.get("bias", 0.0))
    x = features_from_breakdown(breakdown)
    z = bias + sum(w * xi for w, xi in zip(weights, x))
    return round(_sigmoid(z) * 100)


def train_logistic(X: List[List[float]], y: List[int], epochs: int = 500, lr: float = 0.5, l2: float = 1e-3) -> Tuple[List[float], float]:
    if not X:
        return [], 0.0
    n = len(X[0])
    weights = [0.0] * n
    bias = 0.0
    m = len(X)
    for _ in range(epochs):
        grad_w = [0.0] * n
        grad_b = 0.0
        for xi, yi in zip(X, y):
            p = _sigmoid(bias + sum(weights[j] * xi[j] for j in range(n)))
            err = p - yi
            for j in range(n):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(n):
            weights[j] -= lr * (grad_w[j] / m + l2 * weights[j])
        bias -= lr * grad_b / m
    return weights, bias


def _dcg(labels: List[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(labels))


def ndcg_at_k(labels: List[int], k: int = 5) -> float:
    labels = labels[:k]
    ideal = sorted(labels, reverse=True)
    idcg = _dcg(ideal)
    return round(_dcg(labels) / idcg, 4) if idcg > 0 else 0.0


def average_precision(binary: List[int]) -> float:
    hits = 0
    total = 0.0
    for i, rel in enumerate(binary):
        if rel:
            hits += 1
            total += hits / (i + 1)
    return total / hits if hits else 0.0
