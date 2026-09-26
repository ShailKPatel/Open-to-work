from app.profile.manifest_skills import skill_for_dependency, skills_from_manifests


def test_empty_manifests_returns_no_claims():
    assert skills_from_manifests({}) == []


def test_known_dependencies_become_named_claims():
    manifests = {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["fastapi", "sqlalchemy"]},
        "package.json": {"ecosystem": "npm", "dependencies": ["react"]},
    }
    claims = skills_from_manifests(manifests)

    assert len(claims) == 3
    assert all(c.evidence_type == "declared_dependency" for c in claims)
    assert all(c.confidence == 1.0 for c in claims)
    assert {c.skill for c in claims} == {"FastAPI", "SQLAlchemy", "React"}


def test_transitive_and_tooling_packages_are_dropped():
    manifests = {
        "requirements.txt": {
            "ecosystem": "pip",
            "dependencies": ["blinker", "attrs", "asttokens", "certifi", "pandas"],
        },
        "package.json": {
            "ecosystem": "npm",
            "dependencies": ["@types/uuid", "@types/node", "autoprefixer", "@fontsource/inter"],
        },
    }
    assert [c.skill for c in skills_from_manifests(manifests)] == ["Pandas"]


def test_related_packages_fold_into_one_claim():
    manifests = {
        "package.json": {
            "ecosystem": "npm",
            "dependencies": [
                "@types/react",
                "@types/react-dom",
                "@vitejs/plugin-react",
                "react",
                "react-dom",
            ],
        }
    }
    claims = skills_from_manifests(manifests)
    assert [c.skill for c in claims] == ["React"]
    assert claims[0].source_files == ["package.json"]


def test_same_skill_across_manifests_tracks_every_source_file():
    manifests = {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["torch"]},
        "pyproject.toml": {"ecosystem": "python", "dependencies": ["torchvision"]},
    }
    claims = skills_from_manifests(manifests)
    assert len(claims) == 1
    assert claims[0].skill == "PyTorch"
    assert claims[0].source_files == ["requirements.txt", "pyproject.toml"]


def test_name_normalization_and_prefix_families():
    assert skill_for_dependency("Scikit_Learn") == "Scikit-learn"
    assert skill_for_dependency("langchain-openai") == "LangChain"
    assert skill_for_dependency("org.springframework.boot:spring-boot-starter-web") == "Spring Boot"
    assert skill_for_dependency("github.com/gin-gonic/gin") == "Gin"
    assert skill_for_dependency("left-pad") is None
