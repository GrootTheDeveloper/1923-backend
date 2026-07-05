from __future__ import annotations

from datetime import datetime, timezone
import unittest

from bson import ObjectId
from pymongo.errors import BulkWriteError

import app.services.match_pipeline as pipeline


class AsyncCursor:
    def __init__(self, documents):
        self.documents = documents
        self.requested_length = None

    async def to_list(self, length):
        self.requested_length = length
        return self.documents[:length]


class FakeMatchResultsCollection:
    def __init__(self, existing_documents, bulk_write_error=None):
        self.existing_documents = existing_documents
        self.find_calls = []
        self.bulk_write_calls = []
        self.bulk_write_error = bulk_write_error

    def find(self, query):
        self.find_calls.append(query)
        return AsyncCursor(self.existing_documents)

    async def bulk_write(self, operations, ordered=False):
        self.bulk_write_calls.append({"operations": operations, "ordered": ordered})
        if self.bulk_write_error is not None:
            raise self.bulk_write_error


def _stub_calculate_match(cv_document, job, semantic_override=None, semantic_source=None, ranking_model=None, job_context=None):
    return {
        "final_score": 70,
        "semantic_score": semantic_override or 50,
        "score_breakdown": {},
    }


async def _run_single_upsert(collection):
    async def fake_load_active_ranking_model(owner_id):
        return None

    original_collection = pipeline.match_results_collection
    original_loader = pipeline.load_active_ranking_model
    original_calculator = pipeline.calculate_match
    pipeline.match_results_collection = collection
    pipeline.load_active_ranking_model = fake_load_active_ranking_model
    pipeline.calculate_match = _stub_calculate_match
    try:
        return await pipeline.upsert_matches(
            {"_id": ObjectId(), "title": "Backend", "requirements_version": 1},
            [{"_id": ObjectId(), "raw_text": "Python", "extracted_data": {"candidate_name": "A", "email": "a@x.io"}}],
            "owner-1",
        )
    finally:
        pipeline.match_results_collection = original_collection
        pipeline.load_active_ranking_model = original_loader
        pipeline.calculate_match = original_calculator


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

        def fake_calculate_match(cv_document, job, semantic_override=None, semantic_source=None, ranking_model=None, job_context=None):
            seen_contexts.append(job_context)
            return {
                "final_score": 72,
                "final_recommendation_score": 72,
                "rule_score": 70,
                "semantic_score": semantic_override or 50,
                "semantic_source": semantic_source or "lexical",
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
        # Race-safe idempotent upsert on the natural key for both existing and new rows.
        self.assertEqual([op.__class__.__name__ for op in bulk_call["operations"]], ["UpdateOne", "UpdateOne"])
        existing_op, new_op = bulk_call["operations"]
        self.assertTrue(existing_op._upsert)
        self.assertEqual(existing_op._filter, {"job_id": job_id, "cv_id": existing_cv_id, "owner_id": "owner-1"})
        # Recruiter state is set only on insert so a re-match cannot clobber it.
        self.assertEqual(existing_op._doc["$setOnInsert"]["_id"], existing_match_id)
        self.assertNotIn("pipeline_status", existing_op._doc["$set"])

        self.assertEqual(results[0]["_id"], existing_match_id)
        self.assertEqual(results[0]["pipeline_status"], "Reviewed")
        self.assertEqual(results[0]["note"], "keep recruiter note")
        self.assertEqual(results[0]["created_at"], created_at)
        self.assertIsInstance(results[1]["_id"], ObjectId)
        self.assertEqual(results[1]["pipeline_status"], "New")

    async def test_benign_duplicate_key_race_is_swallowed(self):
        error = BulkWriteError({"writeErrors": [{"code": 11000, "errmsg": "duplicate key"}]})
        results = await _run_single_upsert(FakeMatchResultsCollection([], bulk_write_error=error))
        self.assertEqual(len(results), 1)

    async def test_unexpected_bulk_write_error_propagates(self):
        error = BulkWriteError({"writeErrors": [{"code": 121, "errmsg": "document validation failure"}]})
        with self.assertRaises(BulkWriteError):
            await _run_single_upsert(FakeMatchResultsCollection([], bulk_write_error=error))


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
