from app.ingest.github.manifests import parse_dependencies


def test_parse_requirements_txt():
    content = "requests==2.31.0\n# comment\n-e .\nfastapi>=0.100\n\npandas\n"
    assert parse_dependencies("requirements.txt", content) == [
        "fastapi",
        "pandas",
        "requests",
    ]


def test_parse_package_json():
    content = '{"dependencies": {"react": "^18.0.0"}, "devDependencies": {"vite": "^5.0.0"}}'
    assert parse_dependencies("package.json", content) == ["react", "vite"]


def test_parse_pyproject_toml_pep621():
    content = """
[project]
dependencies = ["fastapi>=0.100", "sqlalchemy>=2.0"]
"""
    assert parse_dependencies("pyproject.toml", content) == ["fastapi", "sqlalchemy"]


def test_parse_pyproject_toml_poetry():
    content = """
[tool.poetry.dependencies]
python = "^3.12"
requests = "^2.31"
"""
    assert parse_dependencies("pyproject.toml", content) == ["requests"]


def test_parse_go_mod():
    content = """
module example.com/foo

require (
	github.com/gin-gonic/gin v1.9.1
	github.com/stretchr/testify v1.8.4
)
"""
    assert parse_dependencies("go.mod", content) == [
        "github.com/gin-gonic/gin",
        "github.com/stretchr/testify",
    ]


def test_parse_cargo_toml():
    content = """
[dependencies]
serde = "1.0"
tokio = { version = "1", features = ["full"] }
"""
    assert parse_dependencies("Cargo.toml", content) == ["serde", "tokio"]


def test_parse_gemfile():
    content = """
gem 'rails', '~> 7.0'
gem "pg"
"""
    assert parse_dependencies("Gemfile", content) == ["pg", "rails"]


def test_parse_pom_xml():
    content = """
<project>
  <dependencies>
    <dependency><artifactId>spring-core</artifactId></dependency>
  </dependencies>
</project>
"""
    assert parse_dependencies("pom.xml", content) == ["spring-core"]


def test_parse_gradle():
    content = """
dependencies {
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    testImplementation "junit:junit:4.13.2"
}
"""
    assert parse_dependencies("build.gradle", content) == [
        "com.squareup.okhttp3:okhttp",
        "junit:junit",
    ]


def test_parse_dependencies_malformed_never_raises():
    assert parse_dependencies("package.json", "{not valid json") == []
    assert parse_dependencies("pyproject.toml", "not = [valid") == []
