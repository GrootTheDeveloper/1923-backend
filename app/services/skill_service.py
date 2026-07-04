from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


DEFAULT_SKILL_ALIASES = [
    {
        "canonical": "JavaScript",
        "aliases": ["javascript", "js", "ecmascript", "es6"],
        "category": "Programming Language",
    },
    {
        "canonical": "TypeScript",
        "aliases": ["typescript", "ts"],
        "category": "Programming Language",
    },
    {
        "canonical": "React",
        "aliases": ["react", "reactjs", "react.js"],
        "category": "Frontend",
    },
    {
        "canonical": "Node.js",
        "aliases": ["node", "nodejs", "node.js", "express", "express.js"],
        "category": "Backend",
    },
    {
        "canonical": "Python",
        "aliases": ["python", "py"],
        "category": "Programming Language",
    },
    {
        "canonical": "SQL",
        "aliases": ["sql", "mysql", "postgres", "postgresql", "sql server", "sqlite"],
        "category": "Database",
    },
    {
        "canonical": "MongoDB",
        "aliases": ["mongodb", "mongo"],
        "category": "Database",
    },
    {
        "canonical": "Git",
        "aliases": ["git", "github", "gitlab"],
        "category": "Tooling",
    },
    {
        "canonical": "Docker",
        "aliases": ["docker", "container", "containers"],
        "category": "DevOps",
    },
    {
        "canonical": "REST API",
        "aliases": ["rest", "rest api", "restful", "api", "apis"],
        "category": "Backend",
    },
    {
        "canonical": "HTML",
        "aliases": ["html", "html5"],
        "category": "Frontend",
    },
    {
        "canonical": "CSS",
        "aliases": ["css", "css3", "scss", "sass"],
        "category": "Frontend",
    },
    {
        "canonical": "Tailwind CSS",
        "aliases": ["tailwind", "tailwindcss", "tailwind css"],
        "category": "Frontend",
    },
    {
        "canonical": "FastAPI",
        "aliases": ["fastapi", "fast api"],
        "category": "Backend",
    },
    {
        "canonical": "Django",
        "aliases": ["django", "django rest framework", "drf"],
        "category": "Backend",
    },
    {
        "canonical": "Flask",
        "aliases": ["flask"],
        "category": "Backend",
    },
    {
        "canonical": "Testing",
        "aliases": ["testing", "unit test", "unit tests", "jest", "pytest", "cypress", "selenium", "mocha"],
        "category": "Quality",
    },
    {
        "canonical": "AWS",
        "aliases": ["aws", "amazon web services", "ec2", "s3", "rds", "lambda"],
        "category": "Cloud",
    },
    {
        "canonical": "Azure",
        "aliases": ["azure", "azure app service", "azure static web apps", "azure devops"],
        "category": "Cloud",
    },
    {
        "canonical": "English",
        "aliases": ["english", "ielts", "toeic", "toefl"],
        "category": "Language",
    },
    {
        "canonical": "Spring Boot",
        "aliases": ["spring", "spring boot", "springboot"],
        "category": "Backend",
    },
    {
        "canonical": "Angular",
        "aliases": ["angular", "angularjs", "angular.js"],
        "category": "Frontend",
    },
    {
        "canonical": "Vue.js",
        "aliases": ["vue", "vuejs", "vue.js"],
        "category": "Frontend",
    },
    {
        "canonical": "Kubernetes",
        "aliases": ["kubernetes", "k8s"],
        "category": "DevOps",
    },
    {
        "canonical": "Terraform",
        "aliases": ["terraform"],
        "category": "DevOps",
    },
    {
        "canonical": "Figma",
        "aliases": ["figma"],
        "category": "Design",
    },
    {
        "canonical": "Jira",
        "aliases": ["jira"],
        "category": "Tooling",
    },
    {
        "canonical": "Agile/Scrum",
        "aliases": ["agile", "scrum"],
        "category": "Methodology",
    },
    {
        "canonical": "Machine Learning",
        "aliases": ["machine learning", "ml", "deep learning", "dl", "pytorch", "tensorflow"],
        "category": "Data Science",
    },
    {
        "canonical": "Data Analysis",
        "aliases": ["data analysis", "data science", "pandas", "numpy", "powerbi", "tableau"],
        "category": "Data Science",
    },
    {
        "canonical": "C++",
        "aliases": ["c++", "cpp"],
        "category": "Programming Language",
    },
    {
        "canonical": "Java",
        "aliases": ["java"],
        "category": "Programming Language",
    },
    {
        "canonical": "C#",
        "aliases": ["c#", "csharp"],
        "category": "Programming Language",
    },
    {
        "canonical": "Go",
        "aliases": ["go", "golang"],
        "category": "Programming Language",
    },
    {
        "canonical": "Rust",
        "aliases": ["rust"],
        "category": "Programming Language",
    },
    {
        "canonical": "Swift",
        "aliases": ["swift"],
        "category": "Programming Language",
    },
    {
        "canonical": "Kotlin",
        "aliases": ["kotlin"],
        "category": "Programming Language",
    },
    {
        "canonical": "Flutter",
        "aliases": ["flutter"],
        "category": "Mobile",
    },
    {
        "canonical": "Redux",
        "aliases": ["redux", "redux-toolkit"],
        "category": "Frontend",
    },
    {
        "canonical": "Next.js",
        "aliases": ["nextjs", "next.js", "next"],
        "category": "Frontend",
    },
    {
        "canonical": "GraphQL",
        "aliases": ["graphql", "gql"],
        "category": "Backend",
    },
    {
        "canonical": ".NET",
        "aliases": [".net", "dotnet", ".net core"],
        "category": "Backend",
    },
]

SKILL_ALIAS_CACHE: Dict[str, str] = {}



async def load_skill_aliases_from_db() -> None:
    global SKILL_ALIAS_CACHE
    try:
        from app.database import skill_aliases_collection
        cursor = skill_aliases_collection.find()
        rows = await cursor.to_list(length=1000)
        if rows:
            new_lookup = {}
            for row in rows:
                canonical = row["canonical"]
                new_lookup[_clean_skill(canonical)] = canonical
                for alias in row.get("aliases", []):
                    new_lookup[_clean_skill(alias)] = canonical
            SKILL_ALIAS_CACHE = new_lookup
    except Exception:
        pass


def alias_lookup(alias_rows: Iterable[dict] | None = None) -> Dict[str, str]:
    if alias_rows is None and SKILL_ALIAS_CACHE:
        return SKILL_ALIAS_CACHE
    rows = list(alias_rows or DEFAULT_SKILL_ALIASES)
    lookup: Dict[str, str] = {}
    for row in rows:
        canonical = row["canonical"]
        lookup[_clean_skill(canonical)] = canonical
        for alias in row.get("aliases", []):
            lookup[_clean_skill(alias)] = canonical
    return lookup


def normalize_skill(skill: str, lookup: Dict[str, str] | None = None) -> str:
    if not skill:
        return ""
    table = lookup or alias_lookup()
    cleaned = _clean_skill(skill)
    return table.get(cleaned, _title_skill(skill))


def normalize_skills(skills: Iterable[str], lookup: Dict[str, str] | None = None) -> List[str]:
    normalized = []
    seen = set()
    for skill in skills:
        value = normalize_skill(skill, lookup)
        key = value.casefold()
        if value and key not in seen:
            normalized.append(value)
            seen.add(key)
    return normalized


def extract_skills_from_text(text: str, alias_rows: Iterable[dict] | None = None) -> List[str]:
    lookup = alias_lookup(alias_rows)
    matches: List[Tuple[int, str]] = []
    haystack = f" {text.casefold()} "

    for alias, canonical in lookup.items():
        pattern = r"(?<![a-z0-9+#.])" + re.escape(alias) + r"(?![a-z0-9+#.])"
        match = re.search(pattern, haystack)
        if match:
            matches.append((match.start(), canonical))

    ordered = [canonical for _, canonical in sorted(matches, key=lambda item: item[0])]
    return normalize_skills(ordered, lookup)


def skill_alias_documents() -> List[dict]:
    return [dict(row) for row in DEFAULT_SKILL_ALIASES]


def _clean_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().casefold())


def _title_skill(skill: str) -> str:
    stripped = skill.strip()
    if len(stripped) <= 3 and stripped.isupper():
        return stripped
    return " ".join(part.capitalize() for part in stripped.split())
