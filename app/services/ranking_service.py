"""Learning-to-rank training, registry, and offline evaluation.

Labels come from recruiter signals already in the system:
- Shortlisted / good_match feedback  -> relevant (positive)
- Rejected / bad_match feedback       -> not relevant (negative)

Training is pointwise logistic regression over versioned score-breakdown
features. A candidate model is evaluated on a deterministic hold-out set and is
activated only when it improves out-of-sample ranking quality.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.database import match_feedback_collection, match_results_collection, ranking_models_collection
from app.services.ranking_model import (
    FEATURE_KEYS,
    FEATURE_SCHEMA_VERSION,
    average_precision,
    features_from_breakdown,
    is_model_compatible,
    ndcg_at_k,
    predict_ml_rank,
    train_logistic,
)

MIN_TRAIN_SAMPLES = 5
HOLDOUT_FRACTION = 0.25
# Do not trust (and therefore never activate on) a hold-out smaller than this;
# a 1-sample test set makes NDCG@5 trivially perfect and the lift meaningless.
MIN_HOLDOUT_SAMPLES = 2


def _relevance(match: dict, verdict: Optional[str]) -> int:
    """Graded relevance for NDCG (0/1/2)."""
    if verdict == "good_match" or match.get("pipeline_status") == "Shortlisted":
        return 2
    if match.get("pipeline_status") == "Reviewed":
        return 1
    if verdict == "bad_match" or match.get("pipeline_status") == "Rejected":
        return 0
    return 0


def _binary_label(match: dict, verdict: Optional[str]) -> Optional[int]:
    """Training label: 1 relevant, 0 not, None if unlabeled."""
    if verdict == "good_match" or match.get("pipeline_status") == "Shortlisted":
        return 1
    if verdict == "bad_match" or match.get("pipeline_status") == "Rejected":
        return 0
    return None


async def load_active_ranking_model(owner_id: str) -> Optional[dict]:
    model = await ranking_models_collection.find_one({"owner_id": owner_id, "active": True})
    if model and is_model_compatible(model):
        return model
    return None


async def maybe_retrain_ranker(owner_id: str) -> None:
    """Auto-retrain in the background after HR labeling actions.

    Only attempts retraining when there are enough labels and the label set has
    changed. Activation is still gated by out-of-sample lift in
    ``train_ranking_model``.
    """
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    feedback = await _feedback_for_owner(owner_id)
    verdicts = _verdict_by_match(feedback)
    labels = [lab for lab in (_binary_label(m, verdicts.get(str(m["_id"]))) for m in matches) if lab is not None]
    positives = sum(labels)
    if len(labels) < MIN_TRAIN_SAMPLES or positives == 0 or positives == len(labels):
        return
    active = await load_active_ranking_model(owner_id)
    if active and active.get("labeled_samples") == len(labels) and active.get("positive") == positives:
        return
    await train_ranking_model(owner_id)


async def _feedback_for_owner(owner_id: str) -> List[dict]:
    return await match_feedback_collection.find({"owner_id": owner_id}).to_list(length=10000)


def _verdict_by_match(feedback: List[dict]) -> Dict[str, str]:
    return {str(item["match_id"]): item.get("verdict") for item in feedback if item.get("match_id")}


def _group_by_job(items: List[dict], match_key: str = "match") -> Dict[str, List[dict]]:
    jobs: Dict[str, List[dict]] = {}
    for item in items:
        match = item.get(match_key, item)
        jobs.setdefault(str(match.get("job_id")), []).append(item)
    return jobs


def _sort_value(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _labeled_examples(matches: List[dict], verdicts: Dict[str, str]) -> List[dict]:
    examples = []
    for match in matches:
        verdict = verdicts.get(str(match["_id"]))
        label = _binary_label(match, verdict)
        if label is None:
            continue
        examples.append(
            {
                "match": match,
                "label": label,
                "relevance": _relevance(match, verdict),
                "features": features_from_breakdown(match.get("score_breakdown", {})),
                "sort_key": (_sort_value(match.get("created_at")), str(match.get("_id"))),
            }
        )
    return sorted(examples, key=lambda item: item["sort_key"])


def _has_both_classes(examples: List[dict]) -> bool:
    labels = {item["label"] for item in examples}
    return labels == {0, 1}


def split_train_test(examples: List[dict]) -> tuple[List[dict], List[dict]]:
    """Deterministic stratified hold-out that keeps both labels in train."""
    positives = [item for item in examples if item["label"] == 1]
    negatives = [item for item in examples if item["label"] == 0]
    if len(examples) < MIN_TRAIN_SAMPLES + 2 or not positives or not negatives:
        return examples, []

    test: List[dict] = []
    train: List[dict] = []
    for bucket in (positives, negatives):
        test_count = max(1, round(len(bucket) * HOLDOUT_FRACTION)) if len(bucket) > 1 else 0
        test_count = min(test_count, len(bucket) - 1)
        if test_count:
            train.extend(bucket[:-test_count])
            test.extend(bucket[-test_count:])
        else:
            train.extend(bucket)

    train = sorted(train, key=lambda item: item["sort_key"])
    test = sorted(test, key=lambda item: item["sort_key"])
    while len(train) < MIN_TRAIN_SAMPLES and test:
        train.append(test.pop(0))
        train = sorted(train, key=lambda item: item["sort_key"])

    if len(train) < MIN_TRAIN_SAMPLES or not test or not _has_both_classes(train):
        return examples, []
    return train, test


def _ndcg_map_for_order(labeled: List[tuple], order_key: Callable[[tuple], float]) -> tuple[float, float]:
    ordered = sorted(labeled, key=order_key, reverse=True)
    labels = [rel for _, rel in ordered]
    return ndcg_at_k(labels, 5), average_precision([1 if rel >= 2 else 0 for rel in labels])


def _ranking_metrics_for_examples(examples: List[dict], order_key: Callable[[dict], float]) -> dict:
    ndcgs: List[float] = []
    maps: List[float] = []
    for items in _group_by_job(examples).values():
        labeled = [(item, item["relevance"]) for item in items]
        if not any(rel > 0 for _, rel in labeled):
            continue
        ndcg, ap = _ndcg_map_for_order(labeled, lambda pair: order_key(pair[0]))
        ndcgs.append(ndcg)
        maps.append(ap)
    if not ndcgs:
        return {"jobs_evaluated": 0, "ndcg_at_5": None, "map": None}
    return {
        "jobs_evaluated": len(ndcgs),
        "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 4),
        "map": round(sum(maps) / len(maps), 4),
    }


def evaluate_model_on_examples(examples: List[dict], model: dict) -> dict:
    baseline = _ranking_metrics_for_examples(examples, lambda item: item["match"].get("final_score", 0))
    candidate = _ranking_metrics_for_examples(
        examples,
        lambda item: predict_ml_rank(model, item["match"].get("score_breakdown", {})) or -1,
    )
    ndcg_lift = None
    map_lift = None
    if baseline["ndcg_at_5"] is not None and candidate["ndcg_at_5"] is not None:
        ndcg_lift = round(candidate["ndcg_at_5"] - baseline["ndcg_at_5"], 4)
    if baseline["map"] is not None and candidate["map"] is not None:
        map_lift = round(candidate["map"] - baseline["map"], 4)
    return {"baseline": baseline, "candidate": candidate, "ndcg_lift": ndcg_lift, "map_lift": map_lift}


async def evaluate_current_ranking(owner_id: str) -> dict:
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    feedback = await _feedback_for_owner(owner_id)
    verdicts = _verdict_by_match(feedback)
    examples = _labeled_examples(matches, verdicts)
    baseline = _ranking_metrics_for_examples(examples, lambda item: item["match"].get("final_score", 0))
    active = await load_active_ranking_model(owner_id)
    active_metrics = evaluate_model_on_examples(examples, active)["candidate"] if active else None

    if baseline["jobs_evaluated"] == 0:
        return {
            "status": "insufficient_labels",
            "evaluated_jobs": 0,
            "ndcg_at_5": None,
            "map": None,
            "active_model": bool(active),
            "active_model_metrics": active_metrics,
            "note": "Shortlist/reject candidates or submit feedback to create labels for evaluation.",
        }
    return {
        "status": "ok",
        "evaluated_jobs": baseline["jobs_evaluated"],
        "ndcg_at_5": baseline["ndcg_at_5"],
        "map": baseline["map"],
        "sample_size": len(matches),
        "labeled_samples": len(examples),
        "active_model": bool(active),
        "active_model_metrics": active_metrics,
    }


def _position_bias_summary(feedback: List[dict], labeled_matches: List[dict]) -> dict:
    explicit_feedback = [item for item in feedback if item.get("verdict") in {"good_match", "bad_match", "override"}]
    missing_rank = sum(1 for item in explicit_feedback if item.get("displayed_rank") is None)
    status_labels = max(0, len(labeled_matches) - len(explicit_feedback))
    return {
        "risk": "documented",
        "explicit_feedback_count": len(explicit_feedback),
        "missing_displayed_rank_count": missing_rank,
        "status_label_count": status_labels,
        "note": (
            "Recruiter labels can be position-biased. Store displayed_rank when available; "
            "status-derived labels are treated as potentially biased monitoring signals."
        ),
    }


def _train_candidate_model(owner_id: str, train_examples: List[dict], active: bool, now: datetime) -> dict:
    X = [item["features"] for item in train_examples]
    y = [item["label"] for item in train_examples]
    weights, bias = train_logistic(X, y)
    return {
        "owner_id": owner_id,
        "version": "lr-" + now.strftime("%Y%m%d%H%M%S"),
        "weights": weights,
        "bias": bias,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_keys": FEATURE_KEYS,
        "labeled_samples": len(y),
        "positive": sum(y),
        "negative": len(y) - sum(y),
        "trained_at": now,
        "active": active,
    }


async def train_ranking_model(owner_id: str) -> dict:
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    feedback = await _feedback_for_owner(owner_id)
    verdicts = _verdict_by_match(feedback)
    examples = _labeled_examples(matches, verdicts)
    positives = sum(item["label"] for item in examples)
    negatives = len(examples) - positives
    if len(examples) < MIN_TRAIN_SAMPLES or positives == 0 or negatives == 0:
        return {
            "status": "insufficient_data",
            "labeled_samples": len(examples),
            "positive": positives,
            "negative": negatives,
            "note": f"Need >= {MIN_TRAIN_SAMPLES} labeled samples with both shortlisted and rejected examples.",
        }

    train_examples, test_examples = split_train_test(examples)
    if not test_examples:
        return {
            "status": "insufficient_holdout",
            "labeled_samples": len(examples),
            "positive": positives,
            "negative": negatives,
            "note": "Need enough labeled data to reserve a deterministic train/test hold-out.",
        }

    now = datetime.now(timezone.utc)
    candidate_model = _train_candidate_model(owner_id, train_examples, active=False, now=now)
    candidate_model["labeled_samples"] = len(examples)
    candidate_model["positive"] = positives
    candidate_model["negative"] = negatives
    candidate_model["train_samples"] = len(train_examples)
    candidate_model["test_samples"] = len(test_examples)
    holdout = evaluate_model_on_examples(test_examples, candidate_model)
    holdout_trustworthy = (
        len(test_examples) >= MIN_HOLDOUT_SAMPLES
        and holdout["baseline"].get("jobs_evaluated", 0) >= 1
    )
    positive_lift = (holdout["ndcg_lift"] is not None and holdout["ndcg_lift"] > 0) or (
        holdout["map_lift"] is not None and holdout["map_lift"] > 0
    )
    gate_passed = holdout_trustworthy and positive_lift
    candidate_model["active"] = gate_passed
    metrics = {
        "train_samples": len(train_examples),
        "test_samples": len(test_examples),
        "out_of_sample": holdout,
        "activation_gate": {
            "passed": gate_passed,
            "holdout_trustworthy": holdout_trustworthy,
            "min_holdout_samples": MIN_HOLDOUT_SAMPLES,
            "rule": "activate only when the hold-out is large enough AND NDCG or MAP lift is positive",
        },
        "position_bias": _position_bias_summary(feedback, examples),
    }
    candidate_model["metrics"] = metrics

    if gate_passed:
        await ranking_models_collection.update_many({"owner_id": owner_id, "active": True}, {"$set": {"active": False}})
    await ranking_models_collection.insert_one(candidate_model)

    response = {
        "status": "trained" if gate_passed else "not_activated",
        "version": candidate_model["version"],
        "activated": gate_passed,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_keys": FEATURE_KEYS,
        "labeled_samples": len(examples),
        "train_samples": len(train_examples),
        "test_samples": len(test_examples),
        "positive": positives,
        "negative": negatives,
        "feature_weights": {key: round(w, 3) for key, w in zip(FEATURE_KEYS, candidate_model["weights"])},
        "bias": round(candidate_model["bias"], 3),
        "metrics": metrics,
    }
    response["note"] = (
        "Model is now active; new matches use it for ml_rank_score."
        if gate_passed
        else "Candidate model was stored inactive (hold-out too small or no positive out-of-sample lift)."
    )
    return response
