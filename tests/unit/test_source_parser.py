import pytest

from app.ingest.github.source_parser import InvalidSourceError, parse_source


def test_bare_username():
    result = parse_source("octocat")
    assert result.kind == "user"
    assert result.username == "octocat"
    assert result.repo_full_name is None


def test_bare_username_with_at_prefix():
    result = parse_source("@octocat")
    assert result.kind == "user"
    assert result.username == "octocat"


def test_bare_username_strips_surrounding_whitespace():
    result = parse_source("  octocat  ")
    assert result.username == "octocat"


def test_profile_url_without_scheme():
    result = parse_source("github.com/octocat")
    assert result.kind == "user"
    assert result.username == "octocat"


def test_profile_url_with_scheme():
    result = parse_source("https://github.com/octocat")
    assert result.kind == "user"
    assert result.username == "octocat"


def test_profile_url_with_trailing_slash():
    result = parse_source("https://github.com/octocat/")
    assert result.kind == "user"
    assert result.username == "octocat"


def test_profile_url_www_variant():
    result = parse_source("https://www.github.com/octocat")
    assert result.kind == "user"
    assert result.username == "octocat"


def test_repo_url():
    result = parse_source("https://github.com/octocat/Hello-World")
    assert result.kind == "repo"
    assert result.username == "octocat"
    assert result.repo_full_name == "octocat/Hello-World"


def test_repo_url_without_scheme():
    result = parse_source("github.com/octocat/Hello-World")
    assert result.kind == "repo"
    assert result.repo_full_name == "octocat/Hello-World"


def test_repo_url_strips_git_suffix():
    result = parse_source("https://github.com/octocat/Hello-World.git")
    assert result.repo_full_name == "octocat/Hello-World"


def test_repo_url_with_trailing_slash():
    result = parse_source("https://github.com/octocat/Hello-World/")
    assert result.repo_full_name == "octocat/Hello-World"


def test_repo_url_ignores_query_string_and_fragment():
    result = parse_source("https://github.com/octocat/Hello-World?tab=readme#section")
    assert result.repo_full_name == "octocat/Hello-World"


def test_repo_url_ignores_extra_path_segments():
    # .../owner/repo/tree/main/src -> still just owner/repo
    result = parse_source("https://github.com/octocat/Hello-World/tree/main/src")
    assert result.repo_full_name == "octocat/Hello-World"


def test_empty_input_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("")


def test_whitespace_only_input_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("   ")


def test_non_github_host_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("https://gitlab.com/octocat")


def test_invalid_username_characters_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("not a username with spaces")


def test_username_leading_hyphen_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("-octocat")


def test_profile_url_trailing_hyphen_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("https://github.com/octocat-")


def test_username_consecutive_hyphens_rejected():
    # matches GitHub's username rule
    with pytest.raises(InvalidSourceError):
        parse_source("oct--ocat")


def test_username_dot_rejected():
    # unlike repo names, GitHub usernames don't allow dots
    with pytest.raises(InvalidSourceError):
        parse_source("oct.ocat")


def test_username_underscore_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("oct_ocat")


def test_username_max_length_39_accepted():
    username = "a" * 39
    result = parse_source(username)
    assert result.username == username


def test_username_over_max_length_rejected():
    with pytest.raises(InvalidSourceError):
        parse_source("a" * 40)


def test_username_single_char_accepted():
    result = parse_source("a")
    assert result.username == "a"
