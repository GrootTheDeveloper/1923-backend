import unittest

from bson import ObjectId

from app.routes.jobs import serialize_job
from app.routes.matches import serialize_match
from app.services.extraction_service import extract_cv_data, extract_jd_data
from app.services.matching_service import calculate_match, score_semantic_similarity
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
                "matched_requirements_version": 2,
                "job_requirements_version": 3,
                "cv_snapshot": {"candidate_name": "Alex Nguyen"},
                "job_snapshot": {"title": "Frontend Intern"},
            }
        )

        self.assertEqual(match["matched_requirements_version"], 2)
        self.assertEqual(match["job_requirements_version"], 3)
        self.assertTrue(match["is_outdated"])

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
        self.assertGreater(result["score_breakdown"]["penalty_score"], 0)
        self.assertLess(result["final_score"], 90)
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


if __name__ == "__main__":
    unittest.main()
