from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import app.database as db


class AsyncItems:
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        self.iterator = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class AggregateResult:
    def __init__(self, groups):
        self.groups = groups

    async def to_list(self, length):
        return self.groups[:length]


class FakeCollection:
    def __init__(self, *, groups=None, indexes=None):
        self.groups = groups or []
        self.indexes = indexes or []
        self.created = []
        self.dropped = []

    def aggregate(self, pipeline):
        self.pipeline = pipeline
        return AggregateResult(self.groups)

    def list_indexes(self):
        return AsyncItems(self.indexes)

    async def create_index(self, keys, **options):
        self.created.append((keys, options))

    async def drop_index(self, name):
        self.dropped.append(name)


class DatabaseIndexTests(unittest.TestCase):
    def test_cv_hash_index_replaces_legacy_index_and_is_unique(self):
        cv_collection = FakeCollection(indexes=[
            {"name": "owner_id_1_file_hash_1", "key": {"owner_id": 1, "file_hash": 1}}
        ])
        other = FakeCollection()
        replacements = {
            "users_collection": other,
            "cv_documents_collection": cv_collection,
            "jobs_collection": other,
            "match_results_collection": other,
            "candidates_collection": other,
            "match_jobs_collection": other,
            "match_feedback_collection": other,
            "audit_logs_collection": other,
            "fairness_attributes_collection": other,
            "ranking_models_collection": other,
        }
        with patch.multiple(db, **replacements):
            asyncio.run(db.create_indexes())

        self.assertEqual(cv_collection.dropped, ["owner_id_1_file_hash_1"])
        unique_calls = [options for _, options in cv_collection.created if options.get("name") == "uq_cv_owner_file_hash"]
        self.assertEqual(len(unique_calls), 1)
        self.assertTrue(unique_calls[0]["unique"])
        self.assertIn("partialFilterExpression", unique_calls[0])

    def test_duplicates_are_reported_without_mutation(self):
        group = {"_id": {"owner_id": "owner", "file_hash": "hash"}, "document_ids": [1, 2], "count": 2}
        collection = FakeCollection(groups=[group])
        with patch.object(db, "cv_documents_collection", collection):
            actual = asyncio.run(db.duplicate_cv_file_hash_groups())
        self.assertEqual(actual, [group])
        self.assertEqual(collection.dropped, [])
        self.assertEqual(collection.created, [])


if __name__ == "__main__":
    unittest.main()
