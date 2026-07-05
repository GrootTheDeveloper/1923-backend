import unittest
from types import SimpleNamespace

import app.services.vector_store as vector_store


class VectorStoreVersioningTests(unittest.TestCase):
    def test_active_collection_name_includes_embedding_index_version(self):
        name = vector_store.active_collection_name()
        version = vector_store.vector_index_version()

        self.assertTrue(name.startswith(vector_store.QDRANT_COLLECTION))
        self.assertTrue(name.endswith(version))
        self.assertIn(str(vector_store.EMBEDDING_DIM), version)

    def test_vector_tags_capture_provider_dimension_and_model(self):
        tags = vector_store._vector_tags()

        self.assertEqual(tags["embedding_provider"], vector_store.EMBEDDING_PROVIDER)
        self.assertEqual(tags["embedding_dim"], vector_store.EMBEDDING_DIM)
        self.assertEqual(tags["embedding_model_version"], vector_store.EMBEDDING_MODEL_VERSION)
        self.assertEqual(tags["vector_index_version"], vector_store.vector_index_version())

    def test_point_id_changes_when_index_version_changes(self):
        original = vector_store.vector_index_version
        try:
            vector_store.vector_index_version = lambda: "index-a"
            point_a = vector_store._point_id("cv-1")
            vector_store.vector_index_version = lambda: "index-b"
            point_b = vector_store._point_id("cv-1")
        finally:
            vector_store.vector_index_version = original

        self.assertNotEqual(point_a, point_b)

    def test_collection_size_extractor_handles_client_objects(self):
        info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=vector_store.EMBEDDING_DIM))
            )
        )

        self.assertEqual(vector_store._collection_vector_size(info), vector_store.EMBEDDING_DIM)

    async def _upsert_wrong_dim(self):
        original_enabled = vector_store.VECTOR_SEARCH_ENABLED
        try:
            vector_store.VECTOR_SEARCH_ENABLED = True
            return await vector_store.upsert_cv_vector("cv-1", "owner-1", [0.1, 0.2], [])
        finally:
            vector_store.VECTOR_SEARCH_ENABLED = original_enabled

    def test_upsert_skips_vectors_with_wrong_dimension(self):
        import asyncio

        self.assertFalse(asyncio.run(self._upsert_wrong_dim()))


if __name__ == "__main__":
    unittest.main()
