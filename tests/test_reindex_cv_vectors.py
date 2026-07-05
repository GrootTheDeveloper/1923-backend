import unittest

from scripts.reindex_cv_vectors import profile_for_indexing


class ReindexCVVectorsTests(unittest.TestCase):
    def test_profile_for_indexing_prefers_masked_data(self):
        profile = profile_for_indexing({"masked_data": {"skills": ["Python"]}, "extracted_data": {"skills": ["Java"]}})
        self.assertEqual(profile["skills"], ["Python"])

    def test_profile_for_indexing_masks_extracted_data_fallback(self):
        profile = profile_for_indexing(
            {
                "extracted_data": {
                    "candidate_name": "Alex Nguyen",
                    "email": "alex@example.com",
                    "phone": "+84 900 111 222",
                    "skills": ["React"],
                }
            }
        )
        self.assertIsNone(profile["candidate_name"])
        self.assertIsNone(profile["email"])
        self.assertIsNone(profile["phone"])
        self.assertEqual(profile["skills"], ["React"])


if __name__ == "__main__":
    unittest.main()
