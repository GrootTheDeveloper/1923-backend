from __future__ import annotations

from datetime import datetime, timezone
import unittest

from bson import ObjectId

import app.services.match_pipeline as pipeline


class AsyncCursor:
    def __init__(self, documents):
        self.documents = documents
        self.requested_length = None

    async def to_list(self, length):
        self.requested_length = length
        return self.documents[:length]


class FakeMatchResultsCollection:
    def __init__(self, existing_documents):
        self.existing_documents = existing_documents
        self.find_calls = []
        self.bulk_write_calls = []

    def find(self, query):
        self.find_calls.append(query)
        return AsyncCursor(self.existing_documents)

    async def bulk_write(self, operations, ordered=False):
        self.bulk_write_calls.append({"operations": operations, "ordered": ordered})


class MatchPipelineBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_upsert_matches_reads_existing_once_and_bulk_writes(self):
        job_id = ObjectId()
        existing_cv_id = ObjectId()
        new_cv_id = ObjectId()
        existing_match_id = ObjectId()
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        collection = FakeMatchResultsCollection([
            {
                "_id": existing_match_id,
                "job_id": job_id,
                "cv_id": existing_cv_id,
                "owner_id": "owner-1",
                "pipeline_status": "Reviewed",
                "note": "keep recruiter note",
                "created_at": created_at,
            }
        ])

        async def fake_load_active_ranking_model(owner_id):
            return None

        seen_contexts = []

        def fake_calculate_match(cv_document, job, semantic_override=None, ranking_model=None, job_context=None):
            seen_contexts.append(job_context)
            return {
                "final_score": 72,
                "final_recommendation_score": 72,
                "rule_score": 70,
                "semantic_score": semantic_override or 50,
                "ml_rank_score": 75,
                "confidence_score": 80,
                "score_breakdown": {},
            }

        original_collection = pipeline.match_results_collection
        original_loader = pipeline.load_active_ranking_model
        original_calculator = pipeline.calculate_match
        pipeline.match_results_collection = collection
        pipeline.load_active_ranking_model = fake_load_active_ranking_model
        pipeline.calculate_match = fake_calculate_match
        try:
            results = await pipeline.upsert_matches(
                {"_id": job_id, "title": "Backend", "requirements_version": 3},
                [
                    {
                        "_id": existing_cv_id,
                        "raw_text": "Python APIs",
                        "_vector_score": 88,
                        "extracted_data": {"candidate_name": "Alex", "email": "alex@example.com"},
                    },
                    {
                        "_id": new_cv_id,
                        "raw_text": "FastAPI services",
                        "extracted_data": {"candidate_name": "Bao", "email": "bao@example.com"},
                    },
                ],
                "owner-1",
            )
        finally:
            pipeline.match_results_collection = original_collection
            pipeline.load_active_ranking_model = original_loader
            pipeline.calculate_match = original_calculator

        self.assertEqual(len(collection.find_calls), 1)
        query = collection.find_calls[0]
        self.assertEqual(query["job_id"], job_id)
        self.assertEqual(query["owner_id"], "owner-1")
        self.assertEqual(set(query["cv_id"]["$in"]), {existing_cv_id, new_cv_id})

        self.assertEqual(len(collection.bulk_write_calls), 1)
        self.assertEqual(len(seen_contexts), 2)
        self.assertIs(seen_contexts[0], seen_contexts[1])
        bulk_call = collection.bulk_write_calls[0]
        self.assertFalse(bulk_call["ordered"])
        self.assertEqual([op.__class__.__name__ for op in bulk_call["operations"]], ["ReplaceOne", "InsertOne"])

        self.assertEqual(results[0]["_id"], existing_match_id)
        self.assertEqual(results[0]["pipeline_status"], "Reviewed")
        self.assertEqual(results[0]["note"], "keep recruiter note")
        self.assertEqual(results[0]["created_at"], created_at)
        self.assertIsInstance(results[1]["_id"], ObjectId)
        self.assertEqual(results[1]["pipeline_status"], "New")


class FakeCandidateCursor:
    def __init__(self, documents):
        self.documents = documents
        self.limit_value = None
        self.to_list_length = None

    def limit(self, value):
        self.limit_value = value
        return self

    async def to_list(self, length):
        self.to_list_length = length
        return self.documents[:length]


class FakeCVDocumentsCollection:
    def __init__(self, documents):
        self.documents = documents
        self.cursor = None
        self.find_calls = []

    def find(self, query):
        self.find_calls.append(query)
        self.cursor = FakeCandidateCursor(self.documents)
        return self.cursor


class MatchPipelineRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_retrieval_respects_requested_top_k(self):
        collection = FakeCVDocumentsCollection([
            {"_id": ObjectId(), "owner_id": "owner-1", "status": "Ready"} for _ in range(5)
        ])

        async def empty_vector_retrieve(job, owner_id, top_k):
            return []

        async def empty_skill_retrieve(job, owner_id, top_k):
            return []

        original_collection = pipeline.cv_documents_collection
        original_vector = pipeline._vector_retrieve
        original_skill = pipeline._skill_retrieve
        pipeline.cv_documents_collection = collection
        pipeline._vector_retrieve = empty_vector_retrieve
        pipeline._skill_retrieve = empty_skill_retrieve
        try:
            candidates = await pipeline.retrieve_candidates({"_id": ObjectId()}, "owner-1", None, top_k=3)
        finally:
            pipeline.cv_documents_collection = original_collection
            pipeline._vector_retrieve = original_vector
            pipeline._skill_retrieve = original_skill

        self.assertEqual(len(candidates), 3)
        self.assertEqual(collection.cursor.limit_value, 3)
        self.assertEqual(collection.cursor.to_list_length, 3)


if __name__ == "__main__":
    unittest.main()
