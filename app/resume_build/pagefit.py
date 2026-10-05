"""Page-fit loop: lands a resume on exactly the page count asked for, one
or two pages, never over and never under. Four levers, cheapest first:

1. Typography. layout.py exposes a ladder of density rungs (margins,
   spacing, type size). A rung costs one Tectonic run, no LLM call, and
   loses nothing. Usually the only lever needed.
2. Reserve content. The orchestrator passes along what the model did not
   pick: other candidate projects, points and skills. When the resume is
   too short even at the loosest rung, these are added back, largest
   first. All of it is the account's own data, so no model call.
3. Rewording, once, when the tightest rung still overflows by a line or
   two. The model reads the PDF and proposes shorter wordings, each
   checked by _is_faithful_shortening (shorter, keeps at least two fifths
   of the original, no new numbers or words, no em dash) and by
   grounding.py (no technology the original does not name), and dropped
   if it fails.
4. Cutting, last and one item at a time: a skill, then a project point,
   then an experience point if the role keeps at least one. Roles and
   their company, title and dates are never cut. One model call returns
   an ordered plan, applied entry by entry with a recompile between each.

The model sees the compiled PDF rather than a character count because
overflow is a layout fact: one long word can push a section onto a new
page while a longer summary does not. The loosest rung that still fits
is chosen, so pages come out full rather than half empty.
"""

from __future__ import annotations

import copy
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMDispatchError,
    complete,
    system_message,
    user_message,
)
from app.resume_build.compile import CompileError, compile_tex
from app.resume_build.grounding import TechVocabulary, introduces_technology
from app.resume_build.latex import render_resume
from app.resume_build.layout import BASE_DENSITY_INDEX, DENSITY_LADDER, layout_for

# What "the model is out of reach" looks like here, after app/core/llm.py
# has already tried every stored key for the provider. Not a trimming bug
# and not something a longer wait inside this loop would fix, so the loop
# stops asking and keeps the resume it has.
_LLM_UNREACHABLE = (ApiKeyMissingError, BudgetExceededError, LLMDispatchError)

_DEFAULT_MAX_ITERATIONS = 6
_DEFAULT_MAX_ADDITIONS = 12

# Cuts asked for per look at the PDF. One call that ranks the four weakest
# items costs barely more than one that names the single weakest, and the
# resume that needs three cuts then costs one call instead of three. Kept
# short because each applied cut changes the document the rest of the plan
# was written against, so a long plan's tail is guesswork.
_MAX_PLANNED_CUTS = 4

# Hard ceiling on Tectonic runs for one call, so a pathological document
# (page count that refuses to move with density, a reserve that keeps
# adding content worth zero height) cannot turn into an unbounded
# compile storm behind a request.
_DEFAULT_MAX_COMPILES = 28

_CUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cut_type": {
            "type": "string",
            "enum": ["skill", "project_point", "experience_point", "none"],
        },
        "skill": {"type": "string"},
        "project_name": {"type": "string"},
        "point_text": {"type": "string"},
    },
    "required": ["cut_type"],
}

_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["summary", "project_point", "experience_point"],
                    },
                    "owner": {"type": "string"},
                    "original": {"type": "string"},
                    "shorter": {"type": "string"},
                },
                "required": ["target", "original", "shorter"],
            },
        }
    },
    "required": ["rewrites"],
}

_MAX_REWRITES = 6

# Shortest a rewording may be, as a share of the original. Below this it
# is no longer the same bullet said more tightly, it is a different and
# thinner one, which is a cut by another name.
_MIN_REWRITE_RATIO = 0.4

_REWRITE_SYSTEM_PROMPT = (
    "You are shown a compiled resume PDF that runs past its target page "
    "count by a small amount. Its margins, spacing and type size are "
    "already as tight as they go. Find the lines whose shortening would "
    "pull the overflow back: a bullet whose last few words wrap onto a "
    "line of their own, a summary sentence that wraps by a word or two. "
    "For each, give a shorter wording that says the same thing: drop "
    "filler, use a tighter verb, remove repetition. Never add a fact, "
    "number, tool, or claim that is not in the original, never change a "
    "number, and never use an em dash. Copy the original text exactly as "
    "listed. For a bullet, set owner to the project name or the "
    "experience company it sits under; for the summary, leave owner empty. "
    "Only reword lines where it actually saves a line in the PDF, not "
    "every line. If nothing can be shortened safely, return an empty "
    "list. Return JSON matching the given schema, nothing else."
)

_TRIM_SCHEMA = {
    "type": "object",
    "properties": {"cuts": {"type": "array", "items": _CUT_SCHEMA}},
    "required": ["cuts"],
}

_SYSTEM_PROMPT = (
    "You are shown a compiled resume PDF that is longer than its target "
    "page count. Its margins, spacing and type size have already been "
    "tightened as far as they go, so cutting content is the only option "
    "left. Return an ordered plan of small cuts, weakest item first, to "
    "be applied one at a time until the resume fits: prefer dropping a "
    "skill from the skills list, then a bullet point under a project, "
    "then, only if nothing else is available, a bullet point under an "
    "experience role (never suggest removing an entire experience role, "
    "its company, title, or dates, those are compulsory). Only suggest "
    "cutting a project bullet point from a project that has more than one "
    "point left, and an experience bullet point from a role that has more "
    "than one point left, counting the earlier cuts in your own plan. "
    "Never list the same item twice. Give as many cuts as you are asked "
    "for, ordered so that applying only the first few is still the right "
    "choice: most of the time only the first one or two get used, and the "
    "rest are there so that a resume needing more cuts does not need "
    "another look at it. If nothing safe is left to cut, return an empty "
    "list. Base the order on what you actually see causing the overflow "
    "in the PDF (a long line, a near-empty last page, a widow line), not "
    "a guess. Return JSON matching the given schema, nothing else."
)


class PageFitNotAchievedError(RuntimeError):
    """Ran out of safe cuts (or iterations) before reaching the target
    page count, or lost the ability to ask for more. Carries the best
    result actually reached, in best_tex/best_pdf_bytes/best_page_count,
    so a caller can still offer that rather than nothing.

    Raised for overflow only. Falling *short* of the target is not an
    error: the reserve can legitimately run dry on an account that simply
    does not have two pages of real material yet, and a truthful short
    resume beats a padded one. That case comes back as a FitResult with
    fit_exact False, for the caller to surface however it likes.
    """

    def __init__(
        self,
        message: str,
        best_tex: str,
        best_pdf_bytes: bytes,
        best_page_count: int,
        best_data: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.best_tex = best_tex
        self.best_pdf_bytes = best_pdf_bytes
        self.best_page_count = best_page_count
        self.best_data = best_data


@dataclass
class FitResult:
    tex: str
    pdf_bytes: bytes
    page_count: int
    cuts_made: int
    additions_made: int = 0
    rewrites_made: int = 0
    density: float = 1.0
    target_pages: int = 0
    fit_exact: bool = True
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Attempt:
    density: float
    tex: str
    pdf_bytes: bytes
    pages: int


def page_count(pdf_bytes: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def _fingerprint(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, default=str)


class _Typesetter:
    """Renders and compiles one content dict at one density rung, and
    remembers what it already compiled. The cache matters because the
    ladder walk and the cut/add loops revisit the same (content,
    density) pair often enough that recompiling it would double the
    Tectonic runs for no new information.
    """

    def __init__(self, template: str, max_compiles: int = _DEFAULT_MAX_COMPILES):
        self._template = template
        self._template_file = f"{template}.tex.j2"
        self._cache: dict[tuple[float, str], _Attempt] = {}
        self._unusable: set[float] = set()
        self._budget = max_compiles

    @property
    def budget_left(self) -> int:
        return self._budget

    def attempt(self, data: dict[str, Any], density: float) -> _Attempt:
        key = (density, _fingerprint(data))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        tex = render_resume(
            self._template_file, {**data, "layout": layout_for(self._template, density)}
        )
        pdf_bytes = compile_tex(tex)
        self._budget -= 1
        result = _Attempt(density, tex, pdf_bytes, page_count(pdf_bytes))
        self._cache[key] = result
        return result

    def try_attempt(self, data: dict[str, Any], density: float) -> _Attempt | None:
        """attempt(), but a rung that will not compile at all is reported
        as unavailable rather than failing the whole build. The tightest
        rungs switch the document class to extarticle for a sub-10pt
        body, which needs a package Tectonic has to fetch; losing that
        rung should cost the resume one notch of range, not the whole
        PDF. A rung that fails once is not retried.
        """
        if density in self._unusable or self._budget <= 0:
            return None
        try:
            return self.attempt(data, density)
        except CompileError:
            self._unusable.add(density)
            return None


def _describe_data(data: dict[str, Any]) -> str:
    lines = ["Skills: " + (", ".join(data.get("skills", [])) or "(none)")]
    for p in data.get("projects", []):
        lines.append(f"Project {p['name']!r} points: " + " | ".join(p.get("points", [])))
    for role in data.get("experience", []):
        lines.append(
            f"Experience {role['company']!r} points: " + " | ".join(role.get("points", []))
        )
    return "\n".join(lines)


def _role_holding(data: dict[str, Any], company: str, point_text: str) -> dict[str, Any] | None:
    """The role a suggestion naming `company` and quoting `point_text`
    means. The model only sees company names (see _describe_data), so two
    roles at the same company are told apart by which one holds the
    quoted point; the first in resume order wins if both do."""
    for role in data.get("experience", []):
        if role["company"] == company and point_text in role.get("points", []):
            return role
    return None


def _apply_cut(data: dict[str, Any], suggestion: dict[str, Any]) -> bool:
    """Mutates data in place. Returns True if something was actually
    removed, False if the suggestion didn't match anything current
    (stale reference, hallucinated value) so the caller can treat that
    the same as "none" rather than looping forever on a no-op.
    """
    cut_type = suggestion.get("cut_type")

    if cut_type == "skill":
        skill = str(suggestion.get("skill", "")).strip()
        skills = data.get("skills", [])
        for i, s in enumerate(skills):
            if s.casefold() == skill.casefold():
                del skills[i]
                return True
        return False

    if cut_type == "project_point":
        project_name = str(suggestion.get("project_name", "")).strip()
        point_text = str(suggestion.get("point_text", "")).strip()
        for p in data.get("projects", []):
            if p["name"] != project_name:
                continue
            points = p.get("points", [])
            if point_text in points and len(points) > 1:
                points.remove(point_text)
                return True
        return False

    if cut_type == "experience_point":
        # project_name doubles as the role's company here, keeps the
        # suggestion shape uniform rather than adding a third field name.
        company = str(suggestion.get("project_name", "")).strip()
        point_text = str(suggestion.get("point_text", "")).strip()
        role = _role_holding(data, company, point_text)
        if role is None or len(role["points"]) <= 1:
            return False
        index = role["points"].index(point_text)
        del role["points"][index]
        sources = role.get("source_points")
        if isinstance(sources, list) and index < len(sources):
            del sources[index]
        return True

    return False


_WORD_RE = re.compile(r"[a-z][a-z+#]*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _is_faithful_shortening(original: str, shorter: str) -> bool:
    """The grounding check on a rewording: shorter, not gutted, and built
    only from what the original already says. Words are compared on
    their first four letters so "built" may become "building" and
    "optimized" may become "optimizing", while a word the original never
    used, the way a new tool or a new claim would arrive, fails it.
    Short words (a, to, and, via) are free, they carry no facts.
    """
    shorter = shorter.strip()
    if not shorter or len(shorter) >= len(original):
        return False
    if len(shorter) < _MIN_REWRITE_RATIO * len(original):
        return False
    if "\u2014" in shorter or "\u2013" in shorter:
        return False
    if not set(_NUMBER_RE.findall(shorter)) <= set(_NUMBER_RE.findall(original)):
        return False
    stems = {w[:4] for w in _WORD_RE.findall(original.casefold())}
    return all(
        w[:4] in stems for w in _WORD_RE.findall(shorter.casefold()) if len(w) > 3
    )


def _apply_rewrite(
    data: dict[str, Any], rewrite: dict[str, Any], vocabulary: TechVocabulary | None = None
) -> bool:
    """Mutates data in place. True when the rewording matched a current
    line exactly, passed _is_faithful_shortening() and names no technology
    in `vocabulary` the original does not; anything else (a misquoted
    original, a wording that adds something) is dropped. Short names like
    "Go" or "AWS" slip past the word check, which ignores words of three
    letters or fewer, so the technology check is what catches them."""
    target = rewrite.get("target")
    original = str(rewrite.get("original", "")).strip()
    shorter = str(rewrite.get("shorter", "")).strip()
    owner = str(rewrite.get("owner", "")).strip()
    if not original or not _is_faithful_shortening(original, shorter):
        return False
    if vocabulary is not None and introduces_technology(original, shorter, vocabulary):
        return False

    if target == "summary":
        if (data.get("summary") or "").strip() == original:
            data["summary"] = shorter
            return True
        return False

    if target == "project_point":
        entry = next(
            (
                p
                for p in data.get("projects", [])
                if p.get("name") == owner and original in p.get("points", [])
            ),
            None,
        )
    elif target == "experience_point":
        entry = _role_holding(data, owner, original)
    else:
        return False
    if entry is None:
        return False
    points = entry["points"]
    points[points.index(original)] = shorter
    return True


def _plan_rewrites(
    data: dict[str, Any], overflowing: _Attempt, target_pages: int, account_id: int | None
) -> list[dict[str, Any]]:
    """One look at the overflowing PDF, a list of shorter wordings in the
    shape _apply_rewrite() takes. Quality tier, unlike the cut plan: this
    one writes text that ends up on the resume."""
    response = complete(
        "quality",
        [
            system_message(_REWRITE_SYSTEM_PROMPT),
            user_message(
                f"Target: {target_pages} page(s). "
                f"Current: {overflowing.pages} page(s). "
                f"Give up to {_MAX_REWRITES} rewording(s).\n\n"
                + "Summary: " + (data.get("summary") or "(none)") + "\n"
                + _describe_data(data),
                files=[overflowing.pdf_bytes],
            ),
        ],
        schema=_REWRITE_SCHEMA,
        account_id=account_id,
        purpose="pagefit_rewrite",
    )
    planned = (response.parsed or {}).get("rewrites")
    if not isinstance(planned, list):
        return []
    return [r for r in planned if isinstance(r, dict)][:_MAX_REWRITES]


def _apply_addition(data: dict[str, Any], reserve: dict[str, Any]) -> bool:
    """Moves one held-back item from the reserve into the resume.
    Mutates both in place, returns False once the reserve is empty (or
    holds only things already present, a stale reserve on a resume that
    was edited since it was built).

    Largest first, deliberately: a whole reserve project is worth several
    lines, an experience point worth about one, a skill worth almost
    nothing on its own. Overshooting the target by adding a project is
    fine and in fact useful, since the density ladder then tightens back
    down onto the target exactly, whereas creeping up one skill at a time
    would burn a compile per skill and might never arrive.
    """
    projects_held: list[dict[str, Any]] = reserve.get("projects") or []
    while projects_held:
        project = projects_held.pop(0)
        shown = {p.get("name") for p in data.get("projects", [])}
        if project.get("name") in shown:
            continue
        data.setdefault("projects", []).append(project)
        return True

    # Keyed by role id (see orchestrator.py's _build_reserve). A build
    # checkpoint saved before that was keyed by company, so a key that is
    # no current role's id still matches by company.
    held_points: dict[str, list[str]] = reserve.get("experience_points") or {}
    roles = data.get("experience", [])
    role_ids = {str(role.get("id")) for role in roles}
    for key, points in held_points.items():
        match_on = "id" if key in role_ids else "company"
        while points:
            point = points.pop(0)
            for role in roles:
                if str(role.get(match_on)) != key:
                    continue
                shown = role.setdefault("points", [])
                sources = role.get("source_points")
                if point in (sources if isinstance(sources, list) else shown):
                    continue
                shown.append(point)
                if isinstance(sources, list):
                    sources.append(point)
                return True

    skills_held: list[str] = reserve.get("skills") or []
    while skills_held:
        skill = skills_held.pop(0)
        current = data.setdefault("skills", [])
        if skill.casefold() not in {s.casefold() for s in current}:
            current.append(skill)
            return True

    return False


def _best_fitting_rung(
    typesetter: _Typesetter, data: dict[str, Any], target_pages: int
) -> _Attempt | None:
    """Walks the density ladder for the loosest rung whose page count is
    still at or under the target, which is the rung that fills the
    requested pages most completely. Returns None when even the tightest
    rung overflows.

    A walk rather than a binary search: page count against density is
    only *nearly* monotonic (a rung can land a widow line differently
    than its neighbours), and the walk is short in practice because the
    orchestrator already sizes content for the chosen template, so the
    base rung is normally within a rung or two of the answer.
    """
    base = typesetter.attempt(data, DENSITY_LADDER[BASE_DENSITY_INDEX])

    if base.pages > target_pages:
        for i in range(BASE_DENSITY_INDEX - 1, -1, -1):
            tighter = typesetter.try_attempt(data, DENSITY_LADDER[i])
            if tighter is not None and tighter.pages <= target_pages:
                return tighter
        return None

    best = base
    for i in range(BASE_DENSITY_INDEX + 1, len(DENSITY_LADDER)):
        looser = typesetter.try_attempt(data, DENSITY_LADDER[i])
        if looser is None:
            continue
        if looser.pages > target_pages:
            break
        best = looser
    return best


def _grow_to_target(
    typesetter: _Typesetter,
    working: dict[str, Any],
    reserve: dict[str, Any],
    target_pages: int,
    max_additions: int,
    additions_so_far: int,
) -> tuple[int, dict[str, Any] | None]:
    """Adds reserve content at the loosest rung until the resume reaches
    the target, or overshoots it, which the density ladder then tightens
    back down onto the target exactly. One compile per addition, not a
    whole ladder walk, since only the count at the loosest rung decides
    whether there is still room to fill.

    Returns the new total addition count and a copy of the content as it
    stood just before the final addition. That snapshot is the caller's
    undo: the addition that reaches the target is the only one that can
    overshoot so far that even the tightest layout overflows, and
    rolling back just that one keeps every earlier addition.
    """
    loosest = DENSITY_LADDER[-1]
    additions = additions_so_far
    before_last: dict[str, Any] | None = None

    while additions < max_additions and typesetter.budget_left > 0:
        snapshot = copy.deepcopy(working)
        if not _apply_addition(working, reserve):
            break
        additions += 1
        before_last = snapshot
        attempt = typesetter.try_attempt(working, loosest)
        if attempt is not None and attempt.pages >= target_pages:
            break

    return additions, before_last


def _plan_cuts(
    data: dict[str, Any],
    overflowing: _Attempt,
    target_pages: int,
    remaining: int,
    account_id: int | None,
) -> list[dict[str, Any]]:
    """One look at the overflowing PDF, an ordered list of cuts back, each
    in the shape _apply_cut() takes. Empty means the model found nothing
    safe left to cut.

    The PDF goes with it because overflow is a layout fact (see this
    module's docstring), and the PDF is also the expensive part of this
    prompt, which is the whole reason for asking about several cuts at
    once rather than re-sending it per cut.

    Bulk tier, not quality: the model is ranking items it was handed by
    how little the resume loses without them, not writing anything. The
    grounding check in _apply_cut() is what keeps a weaker model's answer
    safe, and an entry it gets wrong costs the next entry in the plan, not
    a bad resume.
    """
    wanted = max(1, min(remaining, _MAX_PLANNED_CUTS))
    response = complete(
        "bulk",
        [
            system_message(_SYSTEM_PROMPT),
            user_message(
                f"Target: {target_pages} page(s). "
                f"Current: {overflowing.pages} page(s). "
                f"Give up to {wanted} cut(s), weakest first.\n\n"
                + _describe_data(data),
                files=[overflowing.pdf_bytes],
            ),
        ],
        schema=_TRIM_SCHEMA,
        account_id=account_id,
        purpose="pagefit_trim",
    )
    planned = (response.parsed or {}).get("cuts")
    if not isinstance(planned, list):
        return []
    return [cut for cut in planned if isinstance(cut, dict)][:wanted]


def fit_to_page_limit(
    data: dict[str, Any],
    template: str,
    max_pages: int,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    account_id: int | None = None,
    max_additions: int = _DEFAULT_MAX_ADDITIONS,
    max_compiles: int = _DEFAULT_MAX_COMPILES,
    reworded: bool = False,
    cuts_made: int = 0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> FitResult:
    """Renders `data` on `template` at exactly `max_pages` pages.

    `data` may carry a "reserve" key (see the module comment and
    app/resume_build/orchestrator.py); it is consumed here and never
    reaches the template or the returned data, so a caller can hand the
    orchestrator's output straight in and persist FitResult.data as the
    resume's real content.

    Raises PageFitNotAchievedError when the resume is still too long
    after the tightest layout and every safe cut, or when the model
    needed to pick the next cut is out of reach (every key spent, budget
    gone). Either way the error carries the best compiled result, so the
    caller can still hand back a real resume. Coming up short returns
    normally with fit_exact False.

    on_progress hears the content as it stands (reserve included) each
    time a model call has changed it, along with `reworded` and
    `cuts_made`. A fit that stops partway, a Tectonic timeout say, can
    then be started again from that state by passing all three back in,
    rather than paying for the same model calls a second time.
    """
    target_pages = max(1, max_pages)
    working = copy.deepcopy(data)
    reserve = working.pop("reserve", None) or {}
    vocabulary = TechVocabulary(reserve.get("known_technologies") or [])
    typesetter = _Typesetter(template, max_compiles=max_compiles)

    cuts = cuts_made
    additions = 0
    rewrites = 0
    undo_last_addition: dict[str, Any] | None = None
    # Cuts the model has already picked but that have not been applied yet.
    # Refilled by one call whenever it runs dry, so a resume needing three
    # cuts costs one look at the PDF rather than three.
    planned_cuts: list[dict[str, Any]] = []

    def _report() -> None:
        if on_progress is not None:
            on_progress({
                "data": {**copy.deepcopy(working), "reserve": copy.deepcopy(reserve)},
                "reworded": reworded,
                "cuts_made": cuts,
            })

    def _result(attempt: _Attempt) -> FitResult:
        return FitResult(
            tex=attempt.tex,
            pdf_bytes=attempt.pdf_bytes,
            page_count=attempt.pages,
            cuts_made=cuts,
            additions_made=additions,
            rewrites_made=rewrites,
            density=attempt.density,
            target_pages=target_pages,
            fit_exact=attempt.pages == target_pages,
            data=working,
        )

    while True:
        fitting = _best_fitting_rung(typesetter, working, target_pages)

        if fitting is None and undo_last_addition is not None:
            # The addition that finally reached the target overshot far
            # enough that even the tightest layout runs over. Roll back
            # that one addition, keep the earlier ones, and take
            # whatever that lands on.
            working = undo_last_addition
            undo_last_addition = None
            additions -= 1
            reverted = _best_fitting_rung(typesetter, working, target_pages)
            if reverted is not None:
                return _result(reverted)
            fitting = None

        if fitting is None:
            overflowing = typesetter.attempt(working, DENSITY_LADDER[0])
            if cuts >= max_iterations:
                raise PageFitNotAchievedError(
                    f"still {overflowing.pages} pages after {cuts} cuts "
                    f"(limit {max_iterations}), target was {target_pages}",
                    overflowing.tex, overflowing.pdf_bytes, overflowing.pages, working,
                )

            if not reworded:
                # Rewording loses nothing, so it goes before any cut. A
                # model out of reach here is not the end of the fit: the
                # cut step below asks again and reports it properly.
                reworded = True
                try:
                    planned = _plan_rewrites(working, overflowing, target_pages, account_id)
                except _LLM_UNREACHABLE:
                    planned = []
                applied_rewrites = sum(_apply_rewrite(working, r, vocabulary) for r in planned)
                if applied_rewrites:
                    rewrites += applied_rewrites
                    _report()
                    continue

            if not planned_cuts:
                try:
                    planned_cuts = _plan_cuts(
                        working,
                        overflowing,
                        target_pages,
                        remaining=max_iterations - cuts,
                        account_id=account_id,
                    )
                except _LLM_UNREACHABLE as e:
                    # Every key for the provider is spent, or there is no
                    # budget left. The resume itself is already written
                    # and compiled; only the trimming pass is missing, so
                    # hand back what exists rather than throwing away the
                    # whole build. The caller reports it as a resume that
                    # did not reach its page target, with this reason.
                    raise PageFitNotAchievedError(
                        f"the trimming step could not run, so the resume is still "
                        f"{overflowing.pages} page(s) against a target of {target_pages}: {e}",
                        overflowing.tex, overflowing.pdf_bytes, overflowing.pages, working,
                    ) from e

            applied = False
            while planned_cuts:
                suggestion = planned_cuts.pop(0)
                if suggestion.get("cut_type") == "none":
                    continue
                if _apply_cut(working, suggestion):
                    applied = True
                    break
                # Stale or invented reference: the plan was written against
                # the data as it stood, and an earlier cut may have already
                # taken this item. Skipped, not fatal, the next entry in the
                # plan is the next-weakest item anyway.

            if not applied:
                raise PageFitNotAchievedError(
                    f"no more safe cuts available, still {overflowing.pages} pages, "
                    f"target was {target_pages}",
                    overflowing.tex, overflowing.pdf_bytes, overflowing.pages, working,
                )

            cuts += 1
            _report()
            continue

        if fitting.pages >= target_pages or typesetter.budget_left <= 0:
            return _result(fitting)

        grown, undo_last_addition = _grow_to_target(
            typesetter, working, reserve, target_pages, max_additions, additions
        )
        if grown == additions:
            # Reserve is empty and the loosest layout still leaves the
            # resume short. This account does not have `target_pages`
            # worth of real material; hand back the best honest version
            # rather than padding it out.
            return _result(fitting)
        additions = grown
