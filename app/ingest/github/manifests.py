"""Manifest detection and lightweight dependency-name extraction.

Pure functions, no network. Structured deps here; skill extraction
(app/profile/) decides what counts as evidence and how it's weighted. This module only
parses what a manifest file declares.
"""

from __future__ import annotations

import re

MANIFEST_FILENAMES: dict[str, str] = {
    "requirements.txt": "pip",
    "pyproject.toml": "python",
    "package.json": "npm",
    "go.mod": "go",
    "Cargo.toml": "cargo",
    "Gemfile": "bundler",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "build.gradle.kts": "gradle",
}

_PIP_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_TOML_DEP_LINE = re.compile(r'^\s*"?([A-Za-z0-9][A-Za-z0-9._-]*)"?\s*=')
_GO_REQUIRE_LINE = re.compile(r"^\s*([A-Za-z0-9./_-]+)\s+v[\d.]")
_CARGO_DEP_LINE = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*=')
_GEMFILE_LINE = re.compile(r"""^\s*gem\s+['"]([A-Za-z0-9._-]+)['"]""")
_MAVEN_ARTIFACT = re.compile(r"<artifactId>([^<]+)</artifactId>")
_GRADLE_DEP_LINE = re.compile(
    r"""(?:implementation|api|compile|testImplementation)\s*[('"]+([A-Za-z0-9._-]+:[A-Za-z0-9._-]+)"""
)


def parse_dependencies(filename: str, content: str) -> list[str]:
    """Best-effort dependency name extraction. Never raises on malformed input."""
    try:
        if filename == "requirements.txt":
            return _parse_requirements_txt(content)
        if filename == "package.json":
            return _parse_package_json(content)
        if filename == "pyproject.toml":
            return _parse_pyproject_toml(content)
        if filename == "go.mod":
            return _parse_go_mod(content)
        if filename == "Cargo.toml":
            return _parse_cargo_toml(content)
        if filename == "Gemfile":
            return _parse_gemfile(content)
        if filename == "pom.xml":
            return _parse_pom_xml(content)
        if filename in ("build.gradle", "build.gradle.kts"):
            return _parse_gradle(content)
    except Exception:
        return []
    return []


def _parse_requirements_txt(content: str) -> list[str]:
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        m = _PIP_LINE.match(line)
        if m:
            deps.append(m.group(1).lower())
    return sorted(set(deps))


def _parse_package_json(content: str) -> list[str]:
    import json

    data = json.loads(content)
    deps: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        deps.update((data.get(key) or {}).keys())
    return sorted(deps)


def _parse_pyproject_toml(content: str) -> list[str]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return []
    data = tomllib.loads(content)
    deps: set[str] = set()
    project_deps = (data.get("project") or {}).get("dependencies") or []
    for entry in project_deps:
        m = _PIP_LINE.match(entry)
        if m:
            deps.add(m.group(1).lower())
    poetry_deps = ((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {}
    deps.update(k.lower() for k in poetry_deps if k.lower() != "python")
    return sorted(deps)


def _parse_go_mod(content: str) -> list[str]:
    deps = []
    in_require = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("require ("):
            in_require = True
            continue
        if in_require and stripped == ")":
            in_require = False
            continue
        if in_require or stripped.startswith("require "):
            candidate = stripped.removeprefix("require ").strip()
            m = _GO_REQUIRE_LINE.match(candidate)
            if m:
                deps.append(m.group(1))
    return sorted(set(deps))


def _parse_cargo_toml(content: str) -> list[str]:
    deps: set[str] = set()
    in_deps_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") :
            in_deps_section = "dependencies" in stripped
            continue
        if in_deps_section:
            m = _CARGO_DEP_LINE.match(stripped)
            if m:
                deps.add(m.group(1))
    return sorted(deps)


def _parse_gemfile(content: str) -> list[str]:
    deps = []
    for line in content.splitlines():
        m = _GEMFILE_LINE.match(line)
        if m:
            deps.append(m.group(1))
    return sorted(set(deps))


def _parse_pom_xml(content: str) -> list[str]:
    return sorted(set(_MAVEN_ARTIFACT.findall(content)))


def _parse_gradle(content: str) -> list[str]:
    return sorted(set(_GRADLE_DEP_LINE.findall(content)))
