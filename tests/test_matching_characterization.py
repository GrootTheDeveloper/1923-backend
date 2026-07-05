from __future__ import annotations

import unittest

from app.services.matching_service import calculate_match, score_requirements
from app.services.requirement_service import build_requirement_config, normalize_requirement_config
from app.services.ranking_model import FEATURE_KEYS, FEATURE_SCHEMA_VERSION


def cv(skills, *, summary="", experience=None, projects=None, education=None, raw_text=None):
    data = {
        "skills": skills,
        "summary": summary,
        "experience": experience or [],
        "projects": projects or [],
        "education": education or [],
        "certifications": [],
        "languages": [],
    }
    return {
        "raw_text": raw_text or "\n".join([summary, *skills, *(experience or []), *(projects or []), *(education or [])]),
        "extracted_data": data,
    }


def job(required=None, preferred=None, *, level="Junior", responsibilities=None, config=None, raw_text=None):
    required = required or []
    preferred = preferred or []
    extracted = {
        "required_skills": required,
        "preferred_skills": preferred,
        "responsibilities": responsibilities or [],
        "job_level": level,
    }
    value = {
        "raw_text": raw_text or " ".join([*required, *preferred, *(responsibilities or [])]),
        "required_skills": required,
        "preferred_skills": preferred,
        "level": level,
        "extracted_requirements": extracted,
    }
    if config is not None:
        value["requirements_config"] = config
    return value


SCENARIOS = {
    "junior_full_match": (
        cv(["React", "JavaScript", "Git"], summary="Frontend developer", projects=["Built React products with JavaScript and Git."], education=["Bachelor degree"]),
        job(["React", "JavaScript", "Git"], responsibilities=["Build React products with JavaScript"]),
        {"semantic_override": 88},
    ),
    "junior_missing_required": (
        cv(["React"], summary="Frontend developer", projects=["Built a React page."]),
        job(["React", "TypeScript"], ["Testing"], responsibilities=["Build frontend applications with TypeScript"]),
        {"semantic_override": 35},
    ),
    "explicit_knockout_miss": (
        cv(["Python"], summary="Backend developer", experience=["Built Python APIs."]),
        job(["Python", "Kubernetes"], config=[
            {"name": "Python", "type": "skill", "priority": "required", "weight": 15, "is_knockout": False},
            {"name": "Kubernetes", "type": "skill", "priority": "required", "weight": 15, "is_knockout": True},
        ]),
        {"semantic_override": 60},
    ),
    "empty_jd": (
        cv(["Python"], summary="Developer", experience=["Built services."], education=["Bachelor degree"]),
        job(),
        {"semantic_override": 45},
    ),
    "intern_supporting_profile": (
        cv(["React", "Git"], summary="Frontend intern", projects=["Built React UI with Git."], education=["Bachelor of Engineering"]),
        job(["React", "Git"], level="Intern", responsibilities=["Build React UI"]),
        {"semantic_override": 75},
    ),
    "senior_experience_gap": (
        cv(["Python", "FastAPI"], summary="Python developer", projects=["Built a FastAPI service."]),
        job(["Python", "FastAPI"], level="Senior", responsibilities=["Lead distributed systems architecture", "Mentor engineering teams"]),
        {"semantic_override": 52},
    ),
    "learned_ranker": (
        cv(["React", "Git"], summary="Frontend developer", projects=["Built React apps."]),
        job(["React", "Git"]),
        {"semantic_override": 70, "ranking_model": {"version": "golden-v1", "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_keys": FEATURE_KEYS, "weights": [1, 0, 0, 0], "bias": -0.5}},
    ),
    "semantic_override_clamped": (
        cv(["SQL"], summary="Data analyst"),
        job(["SQL"]),
        {"semantic_override": 140},
    ),
}


EXPECTED = {
    "junior_full_match": {"final_score": 91, "rule_score": 94, "semantic_score": 91, "ml_rank_score": 100, "confidence_score": 82, "match_level": "Strong", "is_knockout_failed": False, "missing_required_skills": []},
    "junior_missing_required": {"final_score": 28, "rule_score": 34, "semantic_score": 16, "ml_rank_score": 27, "confidence_score": 42, "match_level": "Weak", "is_knockout_failed": False, "missing_required_skills": ["TypeScript"]},
    "explicit_knockout_miss": {"final_score": 30, "rule_score": 32, "semantic_score": 35, "ml_rank_score": 2, "confidence_score": 50, "match_level": "Weak", "is_knockout_failed": True, "missing_required_skills": ["Kubernetes"]},
    "empty_jd": {"final_score": 39, "rule_score": 45, "semantic_score": 20, "ml_rank_score": 50, "confidence_score": 67, "match_level": "Weak", "is_knockout_failed": False, "missing_required_skills": []},
    "intern_supporting_profile": {"final_score": 71, "rule_score": 73, "semantic_score": 65, "ml_rank_score": 82, "confidence_score": 82, "match_level": "Good", "is_knockout_failed": False, "missing_required_skills": []},
    "senior_experience_gap": {"final_score": 43, "rule_score": 50, "semantic_score": 24, "ml_rank_score": 55, "confidence_score": 72, "match_level": "Weak", "is_knockout_failed": False, "missing_required_skills": []},
    "learned_ranker": {"final_score": 59, "rule_score": 74, "semantic_score": 55, "ml_rank_score": 44, "confidence_score": 72, "match_level": "Partial", "is_knockout_failed": False, "missing_required_skills": []},
    "semantic_override_clamped": {"final_score": 82, "rule_score": 73, "semantic_score": 100, "ml_rank_score": 82, "confidence_score": 63, "match_level": "Strong", "is_knockout_failed": False, "missing_required_skills": []},
}


def snapshot(result):
    return {
        "final_score": result["final_score"],
        "rule_score": result["rule_score"],
        "semantic_score": result["semantic_score"],
        "ml_rank_score": result["ml_rank_score"],
        "confidence_score": result["confidence_score"],
        "match_level": result["match_level"],
        "is_knockout_failed": result["is_knockout_failed"],
        "missing_required_skills": result["missing_required_skills"],
    }


class MatchingCharacterizationTests(unittest.TestCase):
    def test_representative_match_outputs_are_stable(self):
        actual = {
            name: snapshot(calculate_match(candidate, vacancy, **options))
            for name, (candidate, vacancy, options) in SCENARIOS.items()
        }
        self.assertEqual(actual, EXPECTED)

    def test_all_public_scores_stay_in_range(self):
        for name, (candidate, vacancy, options) in SCENARIOS.items():
            result = calculate_match(candidate, vacancy, **options)
            for key in ("final_score", "rule_score", "semantic_score", "ml_rank_score", "confidence_score", "fairness_risk_score"):
                with self.subTest(scenario=name, score=key):
                    self.assertGreaterEqual(result[key], 0)
                    self.assertLessEqual(result[key], 100)

    def test_prepared_job_context_preserves_scores(self):
        from app.services.matching_service import prepare_match_job_context

        for name, (candidate, vacancy, options) in SCENARIOS.items():
            direct = snapshot(calculate_match(candidate, vacancy, **options))
            context = prepare_match_job_context(vacancy)
            with_context = snapshot(calculate_match(candidate, vacancy, **options, job_context=context))
            with self.subTest(scenario=name):
                self.assertEqual(with_context, direct)

    def test_final_blend_weights_sum_to_one(self):
        result = calculate_match(*SCENARIOS["junior_full_match"][:2], **SCENARIOS["junior_full_match"][2])
        weights = result["score_breakdown"]["weight_profile"]
        self.assertAlmostEqual(weights["rule_score"] + weights["semantic_similarity"] + weights["ml_rank"] + weights["confidence"], 1.0)

    def test_fallback_required_skills_are_not_automatically_knockouts(self):
        requirements = build_requirement_config(["React"], [])
        self.assertFalse(requirements[0]["is_knockout"])

    def test_explicit_knockout_survives_normalization(self):
        requirements = normalize_requirement_config([
            {"name": "React", "type": "skill", "priority": "required", "weight": 10, "is_knockout": True}
        ])
        self.assertTrue(requirements[0]["is_knockout"])

    def test_empty_requirements_are_not_scored_as_perfect(self):
        self.assertNotEqual(score_requirements([], {}, "")["requirement_score"], 100)


if __name__ == "__main__":
    unittest.main()
