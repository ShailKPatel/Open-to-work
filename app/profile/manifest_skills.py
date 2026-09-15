"""Deterministic skill claims from parsed manifest dependencies: no LLM
call, no cost, no ambiguity: we already know for a fact these packages are
declared, `app/ingest/github/manifests.py` parsed them at ingestion time.
Confidence is always 1.0 (existence is certain); the LLM-derived signal
(`readme_described`, in `extract.py`) is what carries genuine uncertainty.

A package name is not a skill. A `pip freeze` requirements.txt lists every
transitive dependency (`blinker`, `attrs`, `asttokens`), and a package.json
lists type stubs and build plugins (`@types/react`, `@vitejs/plugin-react`)
that say nothing a resume reader would care about. So only packages in the
curated map below produce a claim, under the name a person would put on a
resume, and several packages that all mean the same thing (`react`,
`react-dom`, `@types/react`) fold into one claim per repo. Anything not in
the map is dropped here; if it matters, the README pass can still name it.
"""

from __future__ import annotations

from app.profile.claims import SkillClaim

# Keys are normalized package names, see _normalize().
_PACKAGE_SKILLS: dict[str, str] = {
    # Python: web, data stores, APIs
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "djangorestframework": "Django REST Framework",
    "sqlalchemy": "SQLAlchemy",
    "pydantic": "Pydantic",
    "celery": "Celery",
    "redis": "Redis",
    "psycopg": "PostgreSQL",
    "psycopg2": "PostgreSQL",
    "psycopg2-binary": "PostgreSQL",
    "asyncpg": "PostgreSQL",
    "pymongo": "MongoDB",
    "motor": "MongoDB",
    "mysqlclient": "MySQL",
    "pymysql": "MySQL",
    "mysql-connector-python": "MySQL",
    "graphene": "GraphQL",
    "strawberry-graphql": "GraphQL",
    "boto3": "AWS",
    "beautifulsoup4": "Beautiful Soup",
    "scrapy": "Scrapy",
    "selenium": "Selenium",
    "playwright": "Playwright",
    "pytest": "pytest",
    # Python: data, ML, AI
    "numpy": "NumPy",
    "pandas": "Pandas",
    "polars": "Polars",
    "scipy": "SciPy",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "statsmodels": "Statsmodels",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "plotly": "Plotly",
    "streamlit": "Streamlit",
    "gradio": "Gradio",
    "dash": "Dash",
    "jupyter": "Jupyter",
    "notebook": "Jupyter",
    "jupyterlab": "Jupyter",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "optuna": "Optuna",
    "shap": "SHAP",
    "torch": "PyTorch",
    "torchvision": "PyTorch",
    "torchaudio": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "transformers": "Hugging Face Transformers",
    "sentence-transformers": "Sentence Transformers",
    "nltk": "NLTK",
    "spacy": "spaCy",
    "gensim": "Gensim",
    "opencv-python": "OpenCV",
    "opencv-python-headless": "OpenCV",
    "opencv-contrib-python": "OpenCV",
    "pyspark": "Apache Spark",
    "networkx": "NetworkX",
    "geopandas": "GeoPandas",
    "shapely": "Shapely",
    "openai": "OpenAI API",
    "anthropic": "Anthropic API",
    "litellm": "LiteLLM",
    "langgraph": "LangGraph",
    "qdrant-client": "Qdrant",
    "chromadb": "ChromaDB",
    "faiss-cpu": "FAISS",
    "faiss-gpu": "FAISS",
    "pinecone": "Pinecone",
    "pinecone-client": "Pinecone",
    # JavaScript / TypeScript
    "react": "React",
    "react-dom": "React",
    "@types/react": "React",
    "@types/react-dom": "React",
    "@vitejs/plugin-react": "React",
    "@vitejs/plugin-react-swc": "React",
    "react-native": "React Native",
    "next": "Next.js",
    "vue": "Vue.js",
    "nuxt": "Nuxt",
    "svelte": "Svelte",
    "@sveltejs/kit": "SvelteKit",
    "typescript": "TypeScript",
    "vite": "Vite",
    "webpack": "webpack",
    "tailwindcss": "Tailwind CSS",
    "express": "Express.js",
    "@nestjs/core": "NestJS",
    "fastify": "Fastify",
    "mongoose": "MongoDB",
    "mongodb": "MongoDB",
    "pg": "PostgreSQL",
    "mysql2": "MySQL",
    "ioredis": "Redis",
    "prisma": "Prisma",
    "@prisma/client": "Prisma",
    "sequelize": "Sequelize",
    "graphql": "GraphQL",
    "@apollo/client": "Apollo GraphQL",
    "@apollo/server": "Apollo GraphQL",
    "redux": "Redux",
    "@reduxjs/toolkit": "Redux",
    "@tanstack/react-query": "TanStack Query",
    "socket.io": "Socket.IO",
    "socket.io-client": "Socket.IO",
    "jsonwebtoken": "JWT",
    "electron": "Electron",
    "electron-builder": "Electron",
    "vite-plugin-electron": "Electron",
    "vite-plugin-electron-renderer": "Electron",
    "chart.js": "Chart.js",
    "react-chartjs-2": "Chart.js",
    "d3": "D3.js",
    "three": "Three.js",
    "framer-motion": "Framer Motion",
    "firebase": "Firebase",
    "@supabase/supabase-js": "Supabase",
    "stripe": "Stripe",
    "@tensorflow/tfjs": "TensorFlow.js",
    "jest": "Jest",
    "vitest": "Vitest",
    "cypress": "Cypress",
    "@playwright/test": "Playwright",
    # Go
    "github.com/gin-gonic/gin": "Gin",
    "github.com/gofiber/fiber/v2": "Fiber",
    "github.com/labstack/echo/v4": "Echo",
    "gorm.io/gorm": "GORM",
    "google.golang.org/grpc": "gRPC",
    # Rust
    "tokio": "Tokio",
    "actix-web": "Actix Web",
    "axum": "Axum",
    "diesel": "Diesel",
    "tonic": "gRPC",
    # Ruby
    "rails": "Ruby on Rails",
    "sinatra": "Sinatra",
    "rspec": "RSpec",
    # Java / Kotlin
    "hibernate-core": "Hibernate",
    "junit": "JUnit",
    "junit-jupiter": "JUnit",
    "lombok": "Lombok",
}

# Families of packages that all mean the same skill, matched by prefix so
# every sub-package (langchain-openai, spring-boot-starter-web, ...) folds in
# without listing each one.
_PREFIX_SKILLS: tuple[tuple[str, str], ...] = (
    ("langchain", "LangChain"),
    ("llama-index", "LlamaIndex"),
    ("spring-boot", "Spring Boot"),
    ("@angular/", "Angular"),
    ("@aws-sdk/", "AWS"),
    ("google-cloud-", "Google Cloud"),
)


def _normalize(dependency: str) -> str:
    name = dependency.strip().lower()
    # Gradle deps arrive as "group:artifact"; the artifact is the useful part.
    if ":" in name and "/" not in name:
        name = name.rsplit(":", 1)[1]
    # pip treats "_" and "-" (and ".") as the same name; package.json and
    # Go module paths use "." meaningfully, so only fold "_".
    return name.replace("_", "-")


def skill_for_dependency(dependency: str) -> str | None:
    """Resume-facing skill name for one declared dependency, or None if the
    package isn't one worth claiming on its own."""
    name = _normalize(dependency)
    if name in _PACKAGE_SKILLS:
        return _PACKAGE_SKILLS[name]
    for prefix, skill in _PREFIX_SKILLS:
        if name.startswith(prefix):
            return skill
    return None


def skills_from_manifests(manifests_json: dict) -> list[SkillClaim]:
    by_skill: dict[str, SkillClaim] = {}
    for filename, manifest in manifests_json.items():
        for dependency in manifest.get("dependencies", []):
            skill = skill_for_dependency(dependency)
            if skill is None:
                continue
            claim = by_skill.get(skill)
            if claim is None:
                by_skill[skill] = SkillClaim(
                    skill=skill,
                    evidence_type="declared_dependency",
                    confidence=1.0,
                    source_files=[filename],
                )
            elif filename not in claim.source_files:
                claim.source_files.append(filename)
    return list(by_skill.values())
