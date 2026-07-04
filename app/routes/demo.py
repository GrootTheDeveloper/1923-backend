from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.database import (
    candidates_collection,
    cv_documents_collection,
    jobs_collection,
    match_results_collection,
)
from app.routes.cvmatch_common import get_optional_user
from app.routes.cvs import serialize_cv
from app.routes.jobs import serialize_job
from app.routes.matches import serialize_match
from app.services.extraction_service import extract_cv_data, extract_jd_data
from app.services.matching_service import calculate_match
from app.services.requirement_service import normalize_requirement_config

router = APIRouter()


DEMO_JD_TEXT = """
Frontend Intern
Company: Groot Studio

Requirements
Must have React, JavaScript, HTML, CSS, Git.
Must be able to build responsive user interfaces and consume REST API endpoints.

Preferred
Tailwind CSS, TypeScript, Testing, English communication.

Responsibilities
Build responsive user interfaces with React.
Integrate REST API data into dashboard screens.
Collaborate with backend teams and review UI bugs.
Write clean components and basic unit tests.

Education
Bachelor degree or current student in Software Engineering, Computer Science, or related field.
"""


DEMO_CVS = [
    {
        "filename": "mai-anh-frontend.pdf",
        "text": """
Mai Anh Tran
mai.anh@example.com
+84 901 111 222
https://github.com/maianh

Summary
Frontend intern candidate focused on React dashboards, clean UI components, and API integrations.

Skills
React, JavaScript, HTML, CSS, Tailwind, Git, REST API, Testing, English

Projects
Built a responsive recruitment dashboard using React and Tailwind CSS.
Integrated REST API endpoints for candidate lists, score filters, and status updates.
Wrote Jest unit tests for reusable table and form components.

Education
Bachelor of Software Engineering, University of Information Technology

Languages
English
""",
    },
    {
        "filename": "quang-minh-fullstack.pdf",
        "text": """
Quang Minh Le
minh.le@example.com
+84 902 333 444
https://github.com/minhle

Summary
Junior full-stack developer with backend API and React experience.

Skills
React, JavaScript, Node.js, Express, MongoDB, Git, REST API, Docker

Experience
Developed admin screens in React and connected them with RESTful APIs.
Built Node.js services for authentication and reporting.

Projects
Inventory management web app with React, Express, MongoDB, and Docker deployment.

Education
Bachelor of Computer Science
""",
    },
    {
        "filename": "linh-vu-ui.pdf",
        "text": """
Linh Vu
linh.vu@example.com
+84 903 555 666

Summary
UI designer learning frontend implementation.

Skills
HTML, CSS, Tailwind CSS, Figma, Git, English

Projects
Designed mobile-friendly landing pages and converted layouts into responsive HTML and CSS.
Created a design system for buttons, colors, and form states.

Education
Bachelor of Multimedia Design

Languages
English
""",
    },
    {
        "filename": "tuan-data.pdf",
        "text": """
Tuan Pham
tuan.pham@example.com
+84 904 777 888

Summary
Data intern candidate interested in analytics and Python automation.

Skills
Python, SQL, pandas, Excel, Git

Projects
Created sales analytics reports using Python and SQL.
Built ETL scripts and charts for business dashboards.

Education
Bachelor of Information Systems
""",
    },
    {
        "filename": "hieu-backend.pdf",
        "text": """
Hieu Nguyen
hieu.nguyen@example.com
+84 905 999 000
https://github.com/hieunguyen

Summary
Backend intern candidate with API and database experience.

Skills
Python, FastAPI, SQL, PostgreSQL, Docker, Git, REST API

Experience
Implemented REST API endpoints for task management and file upload.
Worked with PostgreSQL queries and Docker compose for local development.

Projects
FastAPI project management service with JWT authentication.

Education
Computer Science student
""",
    },
]


@router.post("/seed")
async def seed_demo_data(current_user: dict = Depends(get_optional_user)):
    owner_id = current_user["id"]
    await clear_demo_data(owner_id)

    now = datetime.now(timezone.utc)
    extracted_jd = extract_jd_data(DEMO_JD_TEXT, title="Frontend Intern", company="Groot Studio")
    requirements_config = normalize_requirement_config([], ["React", "JavaScript", "HTML", "CSS", "Git", "REST API"], ["Tailwind CSS", "TypeScript", "Testing", "English"])
    job = {
        "title": "Frontend Intern",
        "company": "Groot Studio",
        "level": "Intern",
        "raw_text": DEMO_JD_TEXT.strip(),
        "extracted_requirements": extracted_jd,
        "required_skills": ["React", "JavaScript", "HTML", "CSS", "Git", "REST API"],
        "preferred_skills": ["Tailwind CSS", "TypeScript", "Testing", "English"],
        "requirements_config": requirements_config,
        "requirements_version": 1,
        "status": "Ready",
        "owner_id": owner_id,
        "demo_seed": True,
        "created_at": now,
        "updated_at": now,
    }
    job_result = await jobs_collection.insert_one(job)
    job["_id"] = job_result.inserted_id

    cv_documents = []
    candidates = []
    for index, sample in enumerate(DEMO_CVS, start=1):
        extracted_cv = extract_cv_data(sample["text"])
        candidate = {
            "name": extracted_cv["candidate_name"],
            "email": extracted_cv["email"],
            "phone": extracted_cv["phone"],
            "links": extracted_cv["links"],
            "owner_id": owner_id,
            "demo_seed": True,
            "created_at": now,
            "updated_at": now,
        }
        candidate_result = await candidates_collection.insert_one(candidate)
        candidate["_id"] = candidate_result.inserted_id
        candidates.append(candidate)

        cv_document = {
            "candidate_id": candidate["_id"],
            "filename": sample["filename"],
            "raw_text": sample["text"].strip(),
            "pages": [
                {
                    "page_number": 1,
                    "text": sample["text"].strip(),
                    "char_count": len(sample["text"].strip()),
                }
            ],
            "page_count": 1,
            "char_count": len(sample["text"].strip()),
            "library_used": "Demo Seed",
            "extracted_data": extracted_cv,
            "status": "Ready",
            "owner_id": owner_id,
            "demo_seed": True,
            "created_at": now,
            "updated_at": now,
        }
        cv_result = await cv_documents_collection.insert_one(cv_document)
        cv_document["_id"] = cv_result.inserted_id
        cv_documents.append(cv_document)

    matches = []
    for cv_document in cv_documents:
        match_payload = calculate_match(cv_document, job)
        match_document = {
            **match_payload,
            "job_id": job["_id"],
            "cv_id": cv_document["_id"],
            "owner_id": owner_id,
            "pipeline_status": "New",
            "note": "",
            "matched_requirements_version": job.get("requirements_version", 1),
            "job_requirements_version": job.get("requirements_version", 1),
            "is_outdated": False,
            "outdated_reason": "",
            "cv_snapshot": {
                "candidate_name": cv_document["extracted_data"].get("candidate_name", "Unnamed Candidate"),
                "email": cv_document["extracted_data"].get("email", ""),
                "filename": cv_document["filename"],
            },
            "job_snapshot": {
                "title": job["title"],
                "company": job["company"],
                "requirements_version": job.get("requirements_version", 1),
            },
            "demo_seed": True,
            "created_at": now,
            "updated_at": now,
        }
        match_result = await match_results_collection.insert_one(match_document)
        match_document["_id"] = match_result.inserted_id
        matches.append(match_document)

    candidate_by_id = {candidate["_id"]: candidate for candidate in candidates}
    return {
        "job": serialize_job(job),
        "cvs": [serialize_cv(document, candidate_by_id.get(document["candidate_id"]), include_full_text=False) for document in cv_documents],
        "matches": sorted([serialize_match(document) for document in matches], key=lambda item: item["final_score"], reverse=True),
    }


async def clear_demo_data(owner_id: str) -> None:
    await match_results_collection.delete_many({"owner_id": owner_id, "demo_seed": True})
    await cv_documents_collection.delete_many({"owner_id": owner_id, "demo_seed": True})
    await candidates_collection.delete_many({"owner_id": owner_id, "demo_seed": True})
    await jobs_collection.delete_many({"owner_id": owner_id, "demo_seed": True})
