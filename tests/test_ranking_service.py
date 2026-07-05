import unittest
from datetime import datetime, timezone

import app.services.ranking_service as ranking_service
from app.services.ranking_model import FEATURE_KEYS, FEATURE_SCHEMA_VERSION


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    async def to_list(self, length=10000):
        return list(self.docs)


class FakeCollection:
    def __init__(self, docs=None, active_model=None):
        self.docs = docs or []
        self.active_model = active_model
        self.inserted = []
        self.updated = []

    def find(self, query):
        owner_id = query.get("owner_id")
        docs = [doc for doc in self.docs if owner_id is None or doc.get("owner_id") == owner_id]
        return FakeCursor(docs)

    async def find_one(self, query):
        if self.active_model and query.get("active") is True:
            return self.active_model
        return None

    async def update_many(self, query, update):
        self.updated.append((query, update))

    async def insert_one(self, document):
        self.inserted.append(document)
        return type("InsertResult", (), {"inserted_id": "fake-id"})()


def labeled_match(match_id, label, feature_score, final_score):
    return {
        "_id": match_id,
        "owner_id": "owner-a",
        "job_id": "job-a",
        "pipeline_status": "Shortlisted" if label else "Rejected",
        "final_score": final_score,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "score_breakdown": {
            "experience_project_score": feature_score,
            "education_language_certification_score": 50,
            "completeness_score": 50,
            "confidence_score": 50,
        },
    }


class RankingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_incompatible_active_model_is_ignored(self):
        original_models = ranking_service.ranking_models_collection
        ranking_service.ranking_models_collection = FakeCollection(
            active_model={"active": True, "weights": [1, 0, 0, 0, 0, 0], "feature_keys": ["old"]}
        )
        try:
            self.assertIsNone(await ranking_service.load_active_ranking_model("owner-a"))
        finally:
            ranking_service.ranking_models_collection = original_models

    async def test_train_model_activates_only_after_positive_holdout_lift(self):
        positives = [labeled_match(f"p{i}", 1, 95, 10) for i in range(4)]
        negatives = [labeled_match(f"n{i}", 0, 5, 90) for i in range(4)]
        matches = positives + negatives

        original_matches = ranking_service.match_results_collection
        original_feedback = ranking_service.match_feedback_collection
        original_models = ranking_service.ranking_models_collection
        model_collection = FakeCollection()
        ranking_service.match_results_collection = FakeCollection(matches)
        ranking_service.match_feedback_collection = FakeCollection([])
        ranking_service.ranking_models_collection = model_collection
        try:
            result = await ranking_service.train_ranking_model("owner-a")
        finally:
            ranking_service.match_results_collection = original_matches
            ranking_service.match_feedback_collection = original_feedback
            ranking_service.ranking_models_collection = original_models

        self.assertEqual(result["status"], "trained")
        self.assertTrue(result["activated"])
        self.assertEqual(result["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(result["feature_keys"], FEATURE_KEYS)
        self.assertGreater(result["metrics"]["out_of_sample"]["ndcg_lift"], 0)
        self.assertGreater(result["metrics"]["out_of_sample"]["map_lift"], 0)
        self.assertTrue(model_collection.updated)
        self.assertTrue(model_collection.inserted[0]["active"])
        self.assertEqual(model_collection.inserted[0]["labeled_samples"], 8)
        self.assertEqual(model_collection.inserted[0]["train_samples"], 6)
        self.assertEqual(model_collection.inserted[0]["test_samples"], 2)
        self.assertEqual(model_collection.inserted[0]["metrics"]["position_bias"]["risk"], "documented")


if __name__ == "__main__":
    unittest.main()
