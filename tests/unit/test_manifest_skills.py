from app.profile.manifest_skills import skills_from_manifests


def test_empty_manifests_returns_no_claims():
    assert skills_from_manifests({}) == []


def test_each_dependency_becomes_a_claim():
    manifests = {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["fastapi", "sqlalchemy"]},
        "package.json": {"ecosystem": "npm", "dependencies": ["react"]},
    }
    claims = skills_from_manifests(manifests)

    assert len(claims) == 3
    assert all(c.evidence_type == "declared_dependency" for c in claims)
    assert all(c.confidence == 1.0 for c in claims)
    skills = {c.skill for c in claims}
    assert skills == {"fastapi", "sqlalchemy", "react"}


def test_claim_source_file_is_tracked():
    manifests = {"go.mod": {"ecosystem": "go", "dependencies": ["github.com/gin-gonic/gin"]}}
    claims = skills_from_manifests(manifests)
    assert claims[0].source_files == ["go.mod"]
