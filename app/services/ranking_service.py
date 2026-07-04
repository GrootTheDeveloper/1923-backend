"""Learning-to-rank training, registry, and offline evaluation.

Labels come from recruiter signals already in the system:
- Shortlisted / good_match feedback  -> relevant (positive)
- Rejected / bad_match feedback       -> not relevant (negative)

Training is pointwise logistic regression over the score-breakdown features.
Trained models are versioned in ranking_models (one active per owner). The
pipeline uses the active model for ml_rank_score; with no model it falls back to
the heuristic proxy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.database import match_feedback_collection, match_results_collection, ranking_models_collection
from app.services.ranking_model import (
    FEATURE_KEYS,
    average_precision,
    features_from_breakdown,
    ndcg_at_k,
    predict_ml_rank,
    train_logistic,
)

MIN_TRAIN_SAMPLES = 5


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
    return await ranking_models_collection.find_one({"owner_id": owner_id, "active": True})


async def maybe_retrain_ranker(owner_id: str) -> None:
    """Auto-retrain in the background after HR labeling actions.

    Only retrains when there are enough labels AND the label set has changed
    since the last trained model, so it isn't redundant on every click.
    """
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    verdicts = await _verdict_by_match(owner_id)
    labels = [lab for lab in (_binary_label(m, verdicts.get(str(m["_id"]))) for m in matches) if lab is not None]
    positives = sum(labels)
    if len(labels) < MIN_TRAIN_SAMPLES or positives == 0 or positives == len(labels):
        return
    active = await load_active_ranking_model(owner_id)
    if active and active.get("labeled_samples") == len(labels) and active.get("positive") == positives:
        return  # label set unchanged since the last training
    await train_ranking_model(owner_id)


async def _verdict_by_match(owner_id: str) -> Dict[str, str]:
    feedback = await match_feedback_collection.find({"owner_id": owner_id}).to_list(length=10000)
    return {str(item["match_id"]): item.get("verdict") for item in feedback if item.get("match_id")}


def _group_by_job(matches: List[dict]) -> Dict[str, List[dict]]:
    jobs: Dict[str, List[dict]] = {}
    for match in matches:
        jobs.setdefault(str(match.get("job_id")), []).append(match)
    return jobs


def _ndcg_map_for_order(labeled: List[tuple], order_key) -> tuple:
    ordered = sorted(labeled, key=order_key, reverse=True)
    labels = [rel for _, rel in ordered]
    return ndcg_at_k(labels, 5), average_precision([1 if rel >= 2 else 0 for rel in labels])


async def evaluate_current_ranking(owner_id: str) -> dict:
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    verdicts = await _verdict_by_match(owner_id)

    ndcgs: List[float] = []
    maps: List[float] = []
    for items in _group_by_job(matches).values():
        labeled = [(m, _relevance(m, verdicts.get(str(m["_id"])))) for m in items]
        if not any(rel > 0 for _, rel in labeled):
            continue
        ndcg, ap = _ndcg_map_for_order(labeled, lambda t: t[0].get("final_score", 0))
        ndcgs.append(ndcg)
        maps.append(ap)

    if not ndcgs:
        return {
            "status": "insufficient_labels",
            "evaluated_jobs": 0,
            "ndcg_at_5": None,
            "map": None,
            "note": "Shortlist/reject candidates or submit feedback to create labels for evaluation.",
        }
    return {
        "status": "ok",
        "evaluated_jobs": len(ndcgs),
        "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 4),
        "map": round(sum(maps) / len(maps), 4),
        "sample_size": len(matches),
    }


async def train_ranking_model(owner_id: str) -> dict:
    matches = await match_results_collection.find({"owner_id": owner_id}).to_list(length=10000)
    verdicts = await _verdict_by_match(owner_id)

    X: List[List[float]] = []
    y: List[int] = []
    for match in matches:
        label = _binary_label(match, verdicts.get(str(match["_id"])))
        if label is None:
            continue
        X.append(features_from_breakdown(match.get("score_breakdown", {})))
        y.append(label)

    positives = sum(y)
    negatives = len(y) - positives
    if len(y) < MIN_TRAIN_SAMPLES or positives == 0 or negatives == 0:
        return {
            "status": "insufficient_data",
            "labeled_samples": len(y),
            "positive": positives,
            "negative": negatives,
            "note": f"Need >= {MIN_TRAIN_SAMPLES} labeled samples with both shortlisted and rejected examples.",
        }

    weights, bias = train_logistic(X, y)
    now = datetime.now(timezone.utc)
    version = "lr-" + now.strftime("%Y%m%d%H%M%S")
    model = {
        "owner_id": owner_id,
        "version": version,
        "weights": weights,
        "bias": bias,
        "feature_keys": FEATURE_KEYS,
        "labeled_samples": len(y),
        "positive": positives,
        "negative": negatives,
        "trained_at": now,
        "active": True,
    }
    metrics = _training_metrics(matches, verdicts, model)
    model["metrics"] = metrics

    await ranking_models_collection.update_many({"owner_id": owner_id, "active": True}, {"$set": {"active": False}})
    await ranking_models_collection.insert_one(model)

    return {
        "status": "trained",
        "version": version,
        "labeled_samples": len(y),
        "positive": positives,
        "negative": negatives,
        "feature_weights": {key: round(w, 3) for key, w in zip(FEATURE_KEYS, weights)},
        "bias": round(bias, 3),
        "metrics": metrics,
        "note": "Model is now active; new matches use it for ml_rank_score.",
    }


def _training_metrics(matches: List[dict], verdicts: Dict[str, str], model: dict) -> dict:
    before: List[float] = []
    after: List[float] = []
    for items in _group_by_job(matches).values():
        labeled = [(m, _relevance(m, verdicts.get(str(m["_id"])))) for m in items]
        if not any(rel > 0 for _, rel in labeled):
            continue
        ndcg_before, _ = _ndcg_map_for_order(labeled, lambda t: t[0].get("final_score", 0))
        ndcg_after, _ = _ndcg_map_for_order(labeled, lambda t: predict_ml_rank(model, t[0].get("score_breakdown", {})))
        before.append(ndcg_before)
        after.append(ndcg_after)
    if not before:
        return {"ndcg_at_5_before": None, "ndcg_at_5_after": None, "jobs_evaluated": 0}
    return {
        "ndcg_at_5_before": round(sum(before) / len(before), 4),
        "ndcg_at_5_after": round(sum(after) / len(after), 4),
        "jobs_evaluated": len(before),
    }
