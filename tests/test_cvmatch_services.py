import importlib
import os
import unittest

from bson import ObjectId

from app.config import MATCH_RECALL_TOP_K
from app.models.cvmatch import AsyncMatchRequest, MatchFeedbackCreate
from app.routes.platform import serialize_match_job
from app.routes.jobs import serialize_job
from app.routes.matches import serialize_match
from app.services.extraction_service import extract_cv_data, extract_jd_data
from app.services.matching_service import calculate_match, calibrate_semantic_score, score_semantic_similarity
from app.services.skill_service import normalize_skill, normalize_skills
from app.routes.demo import DEMO_CVS, DEMO_JD_TEXT


class CVMatchServiceTests(unittest.TestCase):
    def test_skill_alias_normalization(self):
        self.assertEqual(normalize_skill("JS"), "JavaScript")
        self.assertEqual(normalize_skill("ReactJS"), "React")
        self.assertEqual(normalize_skills(["JS", "Javascript", "ReactJS"]), ["JavaScript", "React"])


    def test_job_serialization_defaults_requirement_version(self):
        job = serialize_job(
            {
                "_id": ObjectId(),
                "title": "Frontend Intern",
                "extracted_requirements": {},
                "owner_id": "demo-user",
            }
        )

        self.assertEqual(job["requirements_version"], 1)

    def test_match_serialization_exposes_requirement_version_and_outdated_flag(self):
        match = serialize_match(
            {
                "_id": ObjectId(),
                "job_id": ObjectId(),
                "cv_id": ObjectId(),
                "scoring_config_version": "2026-07-s2-v1",
                "matched_requirements_version": 2,
                "job_requirements_version": 3,
                "cv_snapshot": {"candidate_name": "Alex Nguyen"},
                "job_snapshot": {"title": "Frontend Intern"},
            }
        )

        self.assertEqual(match["matched_requirements_version"], 2)
        self.assertEqual(match["job_requirements_version"], 3)
        self.assertTrue(match["is_outdated"])
        self.assertEqual(match["recruiter_priority_score"], 0)
        self.assertEqual(match["scoring_config_version"], "2026-07-s2-v1")
        self.assertEqual(match["match_explanation"], {})

    def test_matching_recall_default_is_shared(self):
        self.assertEqual(AsyncMatchRequest().top_k, MATCH_RECALL_TOP_K)
        match_job = serialize_match_job({"_id": ObjectId(), "job_id": ObjectId()})
        self.assertEqual(match_job["top_k"], MATCH_RECALL_TOP_K)

    def test_rule_based_cv_and_jd_extraction(self):
        cv_text = """
        Alex Nguyen
        alex@example.com
        +84 900 111 222
        Skills
        React, JS, Tailwind, Git
        Projects
        Built a responsive dashboard using React and REST API integrations.
        Education
        Bachelor of Software Engineering
        """
        jd_text = """
        Frontend Intern
        Requirements
        Must have React, JavaScript, CSS, Git
        Preferred
        Tailwind, REST API
        Responsibilities
        Build responsive user interfaces and collaborate with backend teams.
        """

        cv_data = extract_cv_data(cv_text)
        jd_data = extract_jd_data(jd_text)

        self.assertEqual(cv_data["email"], "alex@example.com")
        self.assertIn("React", cv_data["skills"])
        self.assertIn("JavaScript", jd_data["required_skills"])
        self.assertIn("Tailwind CSS", jd_data["preferred_skills"])

    def test_semantic_calibration_maps_embedding_and_lexical_to_common_scale(self):
        self.assertEqual(calibrate_semantic_score(70, "embedding"), 55)
        self.assertEqual(calibrate_semantic_score(25, "lexical"), 55)
        self.assertLess(calibrate_semantic_score(45, "embedding"), calibrate_semantic_score(45, "lexical"))
        self.assertEqual(score_semantic_similarity({}, {}, "", ""), 45)

    def test_semantic_similarity_rewards_related_cv_and_jd_content(self):
        cv_data = {
            "skills": ["React", "TypeScript", "REST API"],
            "projects": ["Built React dashboards with REST API integration, state management, and responsive UI."],
            "experience": ["Implemented frontend features for analytics products using TypeScript."],
        }
        related_job = {
            "required_skills": ["React", "TypeScript", "REST API"],
            "responsibilities": ["Build responsive dashboards and integrate REST APIs for analytics workflows."],
        }
        unrelated_job = {
            "required_skills": ["MongoDB", "ETL", "Data Warehouse"],
            "responsibilities": ["Design batch pipelines for warehouse ingestion and reporting."],
        }

        related_score = score_semantic_similarity(cv_data, related_job, "", "")
        unrelated_score = score_semantic_similarity(cv_data, unrelated_job, "", "")

        self.assertGreater(related_score, unrelated_score)
        self.assertGreaterEqual(related_score, 30)

    def test_matching_penalizes_missing_required_skills(self):
        cv_document = {
            "raw_text": "Alex built React dashboards with Git and Tailwind CSS.",
            "filename": "alex.pdf",
            "extracted_data": {
                "candidate_name": "Alex Nguyen",
                "email": "alex@example.com",
                "phone": "+84 900 111 222",
                "skills": ["React", "Git", "Tailwind CSS"],
                "projects": ["Built React dashboards with Tailwind CSS."],
                "education": ["Bachelor of Software Engineering"],
            },
        }
        job = {
            "raw_text": "Frontend Intern must have React, JavaScript, CSS, Git. Preferred REST API.",
            "title": "Frontend Intern",
            "required_skills": ["React", "JavaScript", "CSS", "Git"],
            "preferred_skills": ["REST API"],
            "extracted_requirements": {
                "responsibilities": ["Build responsive user interfaces with React."],
                "education_required": ["Bachelor degree preferred"],
            },
        }

        result = calculate_match(cv_document, job)

        self.assertIn("JavaScript", result["missing_required_skills"])
        self.assertEqual(result["score_breakdown"]["penalty_score"], 0)
        self.assertLess(result["final_score"], 90)
        self.assertLessEqual(result["recruiter_priority_score"], result["final_score"])
        self.assertTrue(result["match_explanation"]["summary"])
        self.assertTrue(result["match_explanation"]["risks"])
        self.assertTrue(result["evidence"])

    def test_demo_data_has_clear_top_frontend_candidate(self):
        job_data = extract_jd_data(DEMO_JD_TEXT, title="Frontend Intern", company="Groot Studio")
        job = {
            "raw_text": DEMO_JD_TEXT,
            "title": "Frontend Intern",
            "required_skills": ["React", "JavaScript", "HTML", "CSS", "Git", "REST API"],
            "preferred_skills": ["Tailwind CSS", "TypeScript", "Testing", "English"],
            "extracted_requirements": job_data,
        }
        results = []
        for sample in DEMO_CVS:
            cv = {
                "raw_text": sample["text"],
                "filename": sample["filename"],
                "extracted_data": extract_cv_data(sample["text"]),
            }
            results.append((sample["filename"], calculate_match(cv, job)["final_score"]))

        results.sort(key=lambda item: item[1], reverse=True)

        self.assertEqual(results[0][0], "mai-anh-frontend.pdf")
        self.assertGreater(results[0][1], results[-1][1])


class FairnessTests(unittest.TestCase):
    def test_gap_year_flag_does_not_penalize_score(self):
        from app.services.matching_service import assess_fairness_risk

        clean = assess_fairness_risk({"summary": "Frontend developer", "experience": ["Built React apps"]})
        gap = assess_fairness_risk({"summary": "Took a career break in 2021", "experience": ["Built React apps"]})
        self.assertEqual(clean["score"], 0)
        self.assertGreater(gap["score"], 0)
        self.assertTrue(any(f["signal"] == "gap_year" for f in gap["flags"]))

    def test_final_score_excludes_fairness_term(self):
        # fairness_risk_score is computed but must NOT be subtracted from final_score.
        cv = {
            "raw_text": "Career break in 2020. Built React dashboards with Git and Tailwind CSS.",
            "extracted_data": {
                "skills": ["React", "Git", "Tailwind CSS"],
                "summary": "Career break in 2020, then returned to frontend work.",
                "projects": ["Built React dashboards."],
            },
        }
        job = {"raw_text": "Frontend needs React, Git.", "required_skills": ["React", "Git"], "extracted_requirements": {}}
        result = calculate_match(cv, job)
        b = result["score_breakdown"]
        self.assertGreater(b["fairness_risk_score"], 0)  # gap-year flag fired
        weights = b["weight_profile"]
        reconstructed = round(
            weights["rule_score"] * b["rule_score"]
            + weights["semantic_similarity"] * b["semantic_score"]
            + weights["ml_rank"] * b["ml_rank_score"]
            + weights["confidence"] * b["confidence_score"]
        )
        if not result["is_knockout_failed"]:
            self.assertEqual(result["final_score"], max(0, min(100, reconstructed)))

    def test_fairness_attributes_have_no_gender(self):
        from app.services.fairness_service import infer_fairness_attributes

        attrs = infer_fairness_attributes(
            {"candidate_name": "Tran Thi Mai", "education": ["Bachelor, Bach Khoa University"]},
            "Based in Ho Chi Minh City.",
        )
        self.assertNotIn("inferred_gender", attrs)
        self.assertEqual(attrs["school_tier"], "top")
        self.assertEqual(attrs["region"], "south")


class RankingModelTests(unittest.TestCase):
    def test_ndcg_rewards_correct_order(self):
        from app.services.ranking_model import ndcg_at_k

        self.assertEqual(ndcg_at_k([2, 1, 0], 3), 1.0)  # already ideal
        self.assertLess(ndcg_at_k([0, 1, 2], 3), 1.0)  # worst order scores lower

    def test_logistic_learns_separable_labels(self):
        from app.services.ranking_model import FEATURE_KEYS, FEATURE_SCHEMA_VERSION, predict_ml_rank, train_logistic

        # High experience/project evidence -> relevant, low -> not under the v2 schema.
        X = [[0.9, 0.5, 0.5, 0.5], [0.1, 0.5, 0.5, 0.5]] * 6
        y = [1, 0] * 6
        weights, bias = train_logistic(X, y)
        model = {
            "weights": weights,
            "bias": bias,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_keys": FEATURE_KEYS,
        }
        high = predict_ml_rank(model, {"experience_project_score": 95, "confidence_score": 50})
        low = predict_ml_rank(model, {"experience_project_score": 5, "confidence_score": 50})
        self.assertGreater(high, low)

    def test_calculate_match_uses_learned_model_source(self):
        from app.services.ranking_model import FEATURE_KEYS, FEATURE_SCHEMA_VERSION

        cv = {"raw_text": "React Git", "extracted_data": {"skills": ["React", "Git"]}}
        job = {"raw_text": "React Git", "required_skills": ["React", "Git"], "extracted_requirements": {}}
        model = {
            "version": "lr-test",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_keys": FEATURE_KEYS,
            "weights": [1, 0, 0, 0],
            "bias": -0.5,
        }
        result = calculate_match(cv, job, ranking_model=model)
        self.assertEqual(result["score_breakdown"]["ml_rank_source"], "learned:lr-test")
        self.assertTrue(result["score_breakdown"]["ml_rank_used_in_final_score"])

    def test_incompatible_ranker_model_falls_back_without_zeroing_score(self):
        cv = {"raw_text": "React Git", "extracted_data": {"skills": ["React", "Git"]}}
        job = {"raw_text": "React Git", "required_skills": ["React", "Git"], "extracted_requirements": {}}
        old_model = {"version": "old", "weights": [1, 0, 0, 0, 0, 0], "bias": -0.5}
        result = calculate_match(cv, job, ranking_model=old_model)
        breakdown = result["score_breakdown"]
        self.assertEqual(breakdown["ml_rank_source"], "heuristic_proxy")
        self.assertGreater(breakdown["ml_rank_score"], 0)
        self.assertFalse(breakdown["ml_rank_used_in_final_score"])

    def test_feedback_schema_can_record_display_position_for_bias_audit(self):
        feedback = MatchFeedbackCreate(verdict="good_match", displayed_rank=3)
        self.assertEqual(feedback.displayed_rank, 3)
        self.assertEqual(feedback.label_source, "explicit_feedback")


class AuthEnforcementTests(unittest.TestCase):
    def test_no_credentials_rejected_when_demo_off(self):
        import asyncio

        from fastapi import HTTPException

        import app.routes.cvmatch_common as common

        original = common.ENABLE_DEMO_MODE
        common.ENABLE_DEMO_MODE = False
        try:
            with self.assertRaises(HTTPException) as ctx:
                class MockRequest:
                    headers = {}
                asyncio.run(common.get_optional_user(MockRequest(), None))
            self.assertEqual(ctx.exception.status_code, 401)
        finally:
            common.ENABLE_DEMO_MODE = original

    def test_demo_user_returned_when_demo_on(self):
        import asyncio

        import app.routes.cvmatch_common as common

        original = common.ENABLE_DEMO_MODE
        common.ENABLE_DEMO_MODE = True
        try:
            class MockRequest:
                headers = {}
            user = asyncio.run(common.get_optional_user(MockRequest(), None))
            self.assertEqual(user["id"], "demo-user")
        finally:
            common.ENABLE_DEMO_MODE = original

    def test_invalid_token_is_rejected_even_when_demo_on(self):
        import asyncio

        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        import app.routes.cvmatch_common as common

        original = common.ENABLE_DEMO_MODE
        common.ENABLE_DEMO_MODE = True
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not-a-valid-jwt")
        try:
            class MockRequest:
                headers = {}
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(common.get_optional_user(MockRequest(), credentials))
            self.assertEqual(ctx.exception.status_code, 401)
        finally:
            common.ENABLE_DEMO_MODE = original


class RuntimeConfigTests(unittest.TestCase):
    def _reload_config(self, env: dict[str, str]):
        saved = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            import app.config as config

            return importlib.reload(config)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_development_allows_insecure_defaults(self):
        config = self._reload_config(
            {"ENVIRONMENT": "development", "JWT_SECRET_KEY": "dev-secret-key-change-in-production", "ENABLE_DEMO_MODE": "true"}
        )
        try:
            self.assertEqual(config.validate_runtime_config(), [])
        finally:
            importlib.reload(config)

    def test_production_rejects_default_secret_and_demo_mode(self):
        config = self._reload_config(
            {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "dev-secret-key-change-in-production", "ENABLE_DEMO_MODE": "true"}
        )
        try:
            problems = config.validate_runtime_config()
            self.assertEqual(len(problems), 2)
        finally:
            importlib.reload(config)

    def test_production_accepts_strong_secret_without_demo(self):
        config = self._reload_config(
            {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "a-very-long-random-production-secret", "ENABLE_DEMO_MODE": "false"}
        )
        try:
            self.assertEqual(config.validate_runtime_config(), [])
        finally:
            importlib.reload(config)

    def test_production_rejects_short_secret(self):
        config = self._reload_config(
            {"ENVIRONMENT": "production", "JWT_SECRET_KEY": "short", "ENABLE_DEMO_MODE": "false"}
        )
        try:
            problems = config.validate_runtime_config()
            self.assertTrue(any("too short" in p for p in problems))
        finally:
            importlib.reload(config)

    def test_frontend_origins_are_read_only_from_environment(self):
        config = self._reload_config(
            {
                "FRONTEND_URLS": "http://localhost:5173/, https://recruit.example.com",
                "CORS_ALLOW_ORIGIN_REGEX": "^https://preview-[a-z]+\\.example\\.com$",
            }
        )
        try:
            self.assertEqual(
                config.FRONTEND_URLS,
                ["http://localhost:5173", "https://recruit.example.com"],
            )
            self.assertEqual(
                config.CORS_ALLOW_ORIGIN_REGEX,
                "^https://preview-[a-z]+\\.example\\.com$",
            )
        finally:
            importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
