"""Job-posting analytics: skill demand across everything an account has
collected, rolled up by canonical role family (app/profile/role_family.py)
so "ML Engineer" and "Machine Learning Engineer" postings count together
instead of splitting into two unrelated buckets.

Skill-demand counting is plain SQL aggregation over
JobPosting.extracted_json, not a vector-search feature: semantic search
answers "which postings read as similar to this one" (search_job_postings
below) but cannot correctly answer "how many postings asked for Python";
that needs exact counting over the structured fields every extracted
posting already has, which this file computes directly rather than
approximating through embeddings.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.api.skills import account_skill_names
from app.core.db import JobPosting, RoleFamily
from app.profile.job_extract import parse_skills_required, skill_key
from app.profile.job_place import MODE_LABELS, REMOTE, UNKNOWN, cities, work_mode_kind

router = APIRouter(prefix="/api/job-analytics")


def _have_keys(account_id: int) -> set[str]:
    return {skill_key(name) for name in account_skill_names(account_id).values()}


def _extracted_postings(
    db: Session, account_id: int, role_family_id: int | None
) -> list[JobPosting]:
    query = select(JobPosting).where(
        JobPosting.account_id == account_id, JobPosting.extraction_status == "extracted"
    )
    if role_family_id is not None:
        query = query.where(JobPosting.role_family_id == role_family_id)
    return list(db.execute(query).scalars())


class SkillDemand(BaseModel):
    skill: str
    posting_count: int
    dominant_level: str
    level_counts: dict[str, int]
    have_it: bool


@router.get("/skills-demand", response_model=list[SkillDemand])
def skills_demand(
    account_id: int,
    role_family_id: int | None = None,
    *,
    db: DbSession,
) -> list[SkillDemand]:
    """Every skill mentioned across this account's extracted postings
    (optionally narrowed to one role family), most-requested first, each
    flagged with whether the account's own portfolio already demonstrates
    it: "these are your skills, these are missing, these are what's most
    asked for," in one list.
    """
    postings = _extracted_postings(db, account_id, role_family_id)
    have = _have_keys(account_id)

    counts: Counter[str] = Counter()
    level_counts: dict[str, Counter[str]] = {}
    display: dict[str, str] = {}
    for posting in postings:
        extracted = posting.extracted_json or {}
        for item in parse_skills_required(extracted.get("skills_required", [])):
            key = skill_key(item["skill"])
            display.setdefault(key, item["skill"])
            counts[key] += 1
            if item["level"]:
                level_counts.setdefault(key, Counter())[item["level"]] += 1

    results = []
    for key, count in counts.items():
        levels = level_counts.get(key, Counter())
        dominant = levels.most_common(1)[0][0] if levels else ""
        results.append(
            SkillDemand(
                skill=display[key],
                posting_count=count,
                dominant_level=dominant,
                level_counts=dict(levels),
                have_it=key in have,
            )
        )
    results.sort(key=lambda r: (-r.posting_count, r.skill.casefold()))
    return results


class RoleFamilySummary(BaseModel):
    id: int
    canonical_name: str
    posting_count: int
    top_skills: list[str]


@router.get("/role-families", response_model=list[RoleFamilySummary])
def role_families(account_id: int, *, db: DbSession) -> list[RoleFamilySummary]:
    """Every canonical role this account has collected postings under,
    with each one's most-requested skills, the per-role view: "for
    this role, these are the skills that come up most.\""""
    postings = _extracted_postings(db, account_id, None)
    by_family: dict[int, list[JobPosting]] = {}
    for p in postings:
        if p.role_family_id is not None:
            by_family.setdefault(p.role_family_id, []).append(p)

    families = {f.id: f for f in db.execute(select(RoleFamily)).scalars()}
    results = []
    for family_id, rows in by_family.items():
        family = families.get(family_id)
        if family is None:
            continue
        counts: Counter[str] = Counter()
        display: dict[str, str] = {}
        for p in rows:
            extracted = p.extracted_json or {}
            for item in parse_skills_required(extracted.get("skills_required", [])):
                key = skill_key(item["skill"])
                display.setdefault(key, item["skill"])
                counts[key] += 1
        top = [display[k] for k, _ in counts.most_common(8)]
        results.append(
            RoleFamilySummary(
                id=family.id,
                canonical_name=family.canonical_name,
                posting_count=len(rows),
                top_skills=top,
            )
        )
    results.sort(key=lambda r: -r.posting_count)
    return results


class SkillGap(BaseModel):
    matched: list[str]
    missing: list[str]


@router.get("/gap/{posting_id}", response_model=SkillGap)
def posting_gap(posting_id: int, account_id: int, *, db: DbSession) -> SkillGap:
    """This one posting's required skills, split into what the account's
    portfolio already demonstrates and what it doesn't: the per-job
    "you have this / you're missing this" view."""
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    have = _have_keys(account_id)
    extracted = posting.extracted_json or {}
    matched: list[str] = []
    missing: list[str] = []
    for item in parse_skills_required(extracted.get("skills_required", [])):
        target = matched if skill_key(item["skill"]) in have else missing
        target.append(item["skill"])
    return SkillGap(matched=matched, missing=missing)


class SimilarPosting(BaseModel):
    id: int
    title: str
    company: str
    score: float


@router.get("/similar/{posting_id}", response_model=list[SimilarPosting])
def similar_postings(
    posting_id: int,
    account_id: int,
    top_k: int = 5,
    *,
    db: DbSession,
) -> list[SimilarPosting]:
    """Semantic-search companion to the exact-counting endpoints above:
    other postings this account has collected that read as similar to
    this one (app/retrieval/search.py's search_job_postings). The only
    retrieval-based view in this file; everything above is exact
    aggregation."""
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")

    from app.retrieval.index import job_posting_text
    from app.retrieval.search import search_job_postings

    query_text = job_posting_text(posting)
    if not query_text.strip():
        return []
    hits = search_job_postings(query_text, account_id, top_k=top_k + 1)
    results = []
    for hit in hits:
        if hit.id == posting_id:
            continue
        other = db.get(JobPosting, hit.id)
        if other is None:
            continue
        results.append(
            SimilarPosting(
                id=other.id, title=other.title, company=other.company, score=hit.score
            )
        )
    return results[:top_k]


# A posting where the account already has this share of the asked skills
# counts as a strong match: close enough to apply now.
_STRONG = 0.7
# Below this share a posting is a long way off, not a stretch.
_STRETCH_FLOOR = 0.4


@dataclass
class _Fit:
    posting: JobPosting
    role: RoleFamily | None
    skills: list[dict]
    matched: list[str] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    pay: int | None = None

    @property
    def share(self) -> float | None:
        total = len(self.matched) + len(self.missing)
        return len(self.matched) / total if total else None


def _pct(value: float | None) -> int:
    return round((value or 0) * 100)


def _median(values: list[int]) -> int | None:
    return round(statistics.median(values)) if values else None


_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def _money(amount: int | None, currency: str | None) -> str:
    """Compact annual pay for a sentence: "₹18.5L", "$145k"."""
    if amount is None:
        return ""
    symbol = _SYMBOLS.get(currency or "", "")
    suffix = "" if symbol else f" {currency}" if currency else ""
    if currency == "INR":
        if amount >= 10_000_000:
            return f"{symbol}{amount / 10_000_000:.1f}Cr{suffix}"
        return f"{symbol}{amount / 100_000:.1f}L{suffix}".replace(".0L", "L")
    if amount >= 1_000_000:
        return f"{symbol}{amount / 1_000_000:.1f}M{suffix}"
    return f"{symbol}{round(amount / 1000)}k{suffix}"


class InsightPosting(BaseModel):
    id: int
    title: str
    company: str
    role: str | None
    match_pct: int
    skill_count: int
    matched: list[str]
    missing: list[str]
    # Best stated annual pay in this posting's own currency, which may
    # differ from the dashboard's; rankings use only same-currency pay.
    pay: int | None
    pay_currency: str | None
    applied: bool


class SkillShare(BaseModel):
    skill: str
    posting_count: int
    share_pct: int
    have_it: bool
    median_pay: int | None
    pay_premium_pct: int | None


class RoleInsight(BaseModel):
    id: int | None
    name: str
    posting_count: int
    avg_match_pct: int
    best_match_pct: int
    median_pay: int | None
    strong_count: int
    top_skills: list[SkillShare]
    missing_top: list[str]


class SkillToLearn(BaseModel):
    skill: str
    posting_count: int
    share_pct: int
    unlocks: int
    roles: list[str]
    median_pay: int | None
    pay_premium_pct: int | None
    level: str


class WorkModeShare(BaseModel):
    kind: str  # remote | hybrid | on_site | unknown
    label: str
    posting_count: int
    share_pct: int


class PlaceInsight(BaseModel):
    """One place the postings are: a city, or "Remote" for postings that
    can be done from anywhere, or "Not stated". A posting naming two
    cities counts under both."""

    name: str
    kind: str  # city | remote | unknown
    posting_count: int
    share_pct: int
    on_site_count: int
    hybrid_count: int
    unstated_count: int
    avg_match_pct: int | None
    strong_count: int
    median_pay: int | None


class Advice(BaseModel):
    kind: str
    headline: str
    detail: str
    href: str | None = None


class Insights(BaseModel):
    posting_count: int
    applied_count: int
    strong_count: int
    coverage_pct: int
    currency: str | None
    median_pay: int | None
    paid_count: int
    advice: list[Advice]
    best_role: RoleInsight | None
    roles: list[RoleInsight]
    best_matches: list[InsightPosting]
    top_paying_close: list[InsightPosting]
    near_misses: list[InsightPosting]
    skills_to_learn: list[SkillToLearn]
    demand: list[SkillShare]
    strengths: list[SkillShare]
    work_modes: list[WorkModeShare]
    places: list[PlaceInsight]


def _posting_out(fit: _Fit) -> InsightPosting:
    p = fit.posting
    return InsightPosting(
        id=p.id,
        title=p.title,
        company=p.company,
        role=fit.role.canonical_name if fit.role is not None else None,
        match_pct=_pct(fit.share),
        skill_count=len(fit.matched) + len(fit.missing),
        matched=fit.matched,
        missing=[m["skill"] for m in fit.missing],
        pay=p.salary_max_annual or p.salary_min_annual,
        pay_currency=p.salary_currency,
        applied=p.applied,
    )


def _shares(
    fits: list[_Fit], have: set[str], overall_median: int | None
) -> dict[str, SkillShare]:
    """Every skill asked across fits: how many ask, whether the account
    has it, and what those postings pay against the overall median."""
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    pays: dict[str, list[int]] = {}
    for fit in fits:
        for item in fit.skills:
            key = skill_key(item["skill"])
            display.setdefault(key, item["skill"])
            counts[key] += 1
            if fit.pay is not None:
                pays.setdefault(key, []).append(fit.pay)
    total = max(1, len(fits))
    out: dict[str, SkillShare] = {}
    for key, count in counts.items():
        paid = pays.get(key, [])
        median = _median(paid)
        premium = None
        # Two paid postings at least, or one outlier would make the claim.
        if median is not None and overall_median and len(paid) >= 2:
            premium = round((median - overall_median) / overall_median * 100)
        out[key] = SkillShare(
            skill=display[key],
            posting_count=count,
            share_pct=round(count / total * 100),
            have_it=key in have,
            median_pay=median,
            pay_premium_pct=premium,
        )
    return out


def _sorted_shares(shares: dict[str, SkillShare]) -> list[SkillShare]:
    return sorted(shares.values(), key=lambda s: (-s.posting_count, s.skill.casefold()))


def _fit_mode(fit: _Fit) -> str:
    p = fit.posting
    return work_mode_kind((p.extracted_json or {}).get("work_mode", ""), p.location)


def _work_modes(fits: list[_Fit]) -> list[WorkModeShare]:
    counts = Counter(_fit_mode(f) for f in fits)
    total = max(1, len(fits))
    return [
        WorkModeShare(
            kind=kind, label=MODE_LABELS[kind], posting_count=n, share_pct=round(n / total * 100)
        )
        for kind, n in sorted(counts.items(), key=lambda kv: (kv[0] == UNKNOWN, -kv[1]))
    ]


def _places(fits: list[_Fit]) -> list[PlaceInsight]:
    """Postings grouped by where they are. Remote ones are a place of
    their own whatever city they mention, since they can be done from
    anywhere. Cities first by posting count, then Remote's position by
    its own count, with "Not stated" always last."""
    buckets: dict[str, tuple[str, str, list[tuple[_Fit, str]]]] = {}
    for f in fits:
        mode = _fit_mode(f)
        if mode == REMOTE:
            keys = [("remote", "Remote", "remote")]
        else:
            named = cities(f.posting.location)
            keys = [(c.casefold(), c, "city") for c in named] or [("", "Not stated", "unknown")]
        for key, name, kind in keys:
            buckets.setdefault(key, (name, kind, []))[2].append((f, mode))
    total = max(1, len(fits))
    out: list[PlaceInsight] = []
    for name, kind, rows in buckets.values():
        rates = [f.share for f, _ in rows if f.share is not None]
        modes = Counter(mode for _, mode in rows)
        out.append(
            PlaceInsight(
                name=name,
                kind=kind,
                posting_count=len(rows),
                share_pct=round(len(rows) / total * 100),
                on_site_count=modes["on_site"],
                hybrid_count=modes["hybrid"],
                unstated_count=modes[UNKNOWN],
                avg_match_pct=_pct(sum(rates) / len(rates)) if rates else None,
                strong_count=sum(1 for r in rates if r >= _STRONG),
                median_pay=_median([f.pay for f, _ in rows if f.pay is not None]),
            )
        )
    out.sort(key=lambda p: (p.kind == "unknown", -p.posting_count, p.name.casefold()))
    return out


@router.get("/insights", response_model=Insights)
def insights(account_id: int, *, db: DbSession) -> Insights:
    """The job-search dashboard: every extracted posting scored against
    the account's skills, then rolled up into which roles fit best, which
    jobs to apply to now, which are worth stretching for, and which
    missing skills would open up the most (and best-paid) postings.

    Pay comparisons use only postings in the most common salary currency,
    so a rupee figure is never ranked against a dollar one.
    """
    postings = _extracted_postings(db, account_id, None)
    have = _have_keys(account_id)
    families = {f.id: f for f in db.execute(select(RoleFamily)).scalars()}

    currencies = Counter(
        p.salary_currency
        for p in postings
        if p.salary_currency and (p.salary_max_annual or p.salary_min_annual)
    )
    currency = currencies.most_common(1)[0][0] if currencies else None

    fits: list[_Fit] = []
    for p in postings:
        skills = parse_skills_required((p.extracted_json or {}).get("skills_required", []))
        fit = _Fit(
            posting=p,
            role=families.get(p.role_family_id) if p.role_family_id is not None else None,
            skills=skills,
        )
        for item in skills:
            if skill_key(item["skill"]) in have:
                fit.matched.append(item["skill"])
            else:
                fit.missing.append(item)
        best = p.salary_max_annual or p.salary_min_annual
        if best and p.salary_currency == currency:
            fit.pay = best
        fits.append(fit)

    scored = [f for f in fits if f.share is not None]
    paid = [f.pay for f in fits if f.pay is not None]
    overall_median = _median(paid)
    shares = _shares(fits, have, overall_median)

    mentions = sum(len(f.skills) for f in fits)
    covered = sum(len(f.matched) for f in fits)

    # Roles, with postings that never got a family grouped by title.
    groups: dict[tuple[int | None, str], list[_Fit]] = {}
    for f in scored:
        group = (f.role.id, f.role.canonical_name) if f.role else (None, f.posting.title)
        groups.setdefault(group, []).append(f)
    roles: list[RoleInsight] = []
    for (role_id, name), rows in groups.items():
        role_shares = _shares(rows, have, overall_median)
        missing_counts: Counter[str] = Counter()
        for f in rows:
            for m in f.missing:
                missing_counts[m["skill"]] += 1
        rates = [f.share or 0 for f in rows]
        roles.append(
            RoleInsight(
                id=role_id,
                name=name,
                posting_count=len(rows),
                avg_match_pct=_pct(sum(rates) / len(rates)),
                best_match_pct=_pct(max(rates)),
                median_pay=_median([f.pay for f in rows if f.pay is not None]),
                strong_count=sum(1 for r in rates if r >= _STRONG),
                top_skills=_sorted_shares(role_shares)[:8],
                missing_top=[skill for skill, _ in missing_counts.most_common(5)],
            )
        )
    # Fit first; more postings breaks ties, since a role seen once says less.
    roles.sort(key=lambda r: (-r.avg_match_pct, -r.posting_count, r.name.casefold()))
    best_role = next((r for r in roles if r.posting_count >= 2), roles[0] if roles else None)

    open_fits = [f for f in scored if not f.posting.applied]
    best_matches = sorted(open_fits, key=lambda f: (-(f.share or 0), -(f.pay or 0)))[:6]
    close = [f for f in open_fits if (f.share or 0) >= _STRETCH_FLOOR and f.pay is not None]
    top_paying_close = sorted(close, key=lambda f: (-(f.pay or 0), -(f.share or 0)))[:4]
    shown = {f.posting.id for f in best_matches}
    near_misses = sorted(
        (f for f in open_fits if 1 <= len(f.missing) <= 2 and f.posting.id not in shown),
        key=lambda f: (len(f.missing), -(f.pay or 0), -(f.share or 0)),
    )[:6]

    # What each missing skill would open up.
    learn: dict[str, dict] = {}
    for f in scored:
        total = len(f.matched) + len(f.missing)
        lifts = (f.share or 0) < _STRONG <= (len(f.matched) + 1) / total
        for m in f.missing:
            key = skill_key(m["skill"])
            entry = learn.setdefault(key, {"unlocks": 0, "roles": Counter(), "levels": Counter()})
            if lifts:
                entry["unlocks"] += 1
            entry["roles"][f.role.canonical_name if f.role else f.posting.title] += 1
            if m["level"]:
                entry["levels"][m["level"]] += 1
    skills_to_learn: list[SkillToLearn] = []
    for key, entry in learn.items():
        share = shares[key]
        skills_to_learn.append(
            SkillToLearn(
                skill=share.skill,
                posting_count=share.posting_count,
                share_pct=share.share_pct,
                unlocks=entry["unlocks"],
                roles=[r for r, _ in entry["roles"].most_common(3)],
                median_pay=share.median_pay,
                pay_premium_pct=share.pay_premium_pct,
                level=entry["levels"].most_common(1)[0][0] if entry["levels"] else "",
            )
        )

    def _learn_score(s: SkillToLearn) -> float:
        premium = max(0, s.pay_premium_pct or 0) / 20
        return s.posting_count + 2 * s.unlocks + premium

    skills_to_learn.sort(key=lambda s: (-_learn_score(s), s.skill.casefold()))
    skills_to_learn = skills_to_learn[:10]

    ranked = _sorted_shares(shares)
    demand = ranked[:20]
    strengths = [s for s in ranked if s.have_it][:12]
    places = _places(fits)

    advice = _advice(
        fits=fits,
        scored=scored,
        roles=roles,
        best_role=best_role,
        best_matches=best_matches,
        near_misses=near_misses,
        skills_to_learn=skills_to_learn,
        currency=currency,
        overall_median=overall_median,
        places=places,
    )

    return Insights(
        posting_count=len(fits),
        applied_count=sum(1 for f in fits if f.posting.applied),
        strong_count=sum(1 for f in scored if (f.share or 0) >= _STRONG),
        coverage_pct=_pct(covered / mentions) if mentions else 0,
        currency=currency,
        median_pay=overall_median,
        paid_count=len(paid),
        advice=advice,
        best_role=best_role,
        roles=roles,
        best_matches=[_posting_out(f) for f in best_matches],
        top_paying_close=[_posting_out(f) for f in top_paying_close],
        near_misses=[_posting_out(f) for f in near_misses],
        skills_to_learn=skills_to_learn,
        demand=demand,
        strengths=strengths,
        work_modes=_work_modes(fits),
        places=places,
    )


def _advice(
    *,
    fits: list[_Fit],
    scored: list[_Fit],
    roles: list[RoleInsight],
    best_role: RoleInsight | None,
    best_matches: list[_Fit],
    near_misses: list[_Fit],
    skills_to_learn: list[SkillToLearn],
    currency: str | None,
    overall_median: int | None,
    places: list[PlaceInsight],
) -> list[Advice]:
    """Plain-language next steps, most useful first. Each one is only
    said when the numbers behind it hold."""
    out: list[Advice] = []

    if best_role is not None:
        out.append(
            Advice(
                kind="role",
                headline=f"Your closest fit is {best_role.name}",
                detail=(
                    f"Across {best_role.posting_count} posting"
                    f"{'s' if best_role.posting_count != 1 else ''} you already have "
                    f"{best_role.avg_match_pct}% of the skills asked for."
                    + (
                        f" Most often missing: {', '.join(best_role.missing_top[:3])}."
                        if best_role.missing_top
                        else ""
                    )
                ),
            )
        )

    strong_open = [f for f in best_matches if (f.share or 0) >= _STRONG]
    if strong_open:
        top = strong_open[0].posting
        out.append(
            Advice(
                kind="apply",
                headline=(
                    f"Apply now: {len(strong_open)} strong match"
                    f"{'es' if len(strong_open) != 1 else ''} not applied to yet"
                ),
                detail=(
                    f"Start with {top.title} at {top.company} "
                    f"({_pct(strong_open[0].share)}% match)."
                ),
                href=f"/jobs/{top.id}",
            )
        )

    # The best-paid role that is still within reach, when it is not the
    # best-fit one: the "wait and build skills" path.
    reachable = [
        r for r in roles if r.median_pay is not None and r.avg_match_pct >= _pct(_STRETCH_FLOOR)
    ]
    if reachable:
        richest = max(reachable, key=lambda r: r.median_pay or 0)
        if best_role is None or richest.name != best_role.name:
            gap = ", ".join(richest.missing_top[:3])
            out.append(
                Advice(
                    kind="pay",
                    headline=f"{richest.name} pays the most among roles close to you",
                    detail=(
                        f"Median {_money(richest.median_pay, currency)} a year and you are at "
                        f"{richest.avg_match_pct}%."
                        + (f" Closing the gap mostly means {gap}." if gap else "")
                    ),
                )
            )

    place = _place_advice(places, len(fits))
    if place is not None:
        out.append(place)

    if skills_to_learn:
        s = skills_to_learn[0]
        unlock = (
            f" and it would lift {s.unlocks} more to a strong match" if s.unlocks else ""
        )
        out.append(
            Advice(
                kind="learn",
                headline=f"Learn {s.skill} next",
                detail=f"{s.posting_count} of your postings ask for it{unlock}.",
            )
        )

    premium = max(
        (s for s in skills_to_learn if (s.pay_premium_pct or 0) >= 10),
        key=lambda s: s.pay_premium_pct or 0,
        default=None,
    )
    if premium is not None:
        out.append(
            Advice(
                kind="premium",
                headline=f"{premium.skill} comes with higher pay",
                detail=(
                    f"Postings asking for it pay a median {_money(premium.median_pay, currency)}, "
                    f"{premium.pay_premium_pct}% above your overall median of "
                    f"{_money(overall_median, currency)}."
                ),
            )
        )

    one_away = [f for f in near_misses if len(f.missing) == 1]
    if one_away:
        f = one_away[0]
        out.append(
            Advice(
                kind="near",
                headline=(
                    f"{len(one_away)} job{'s are' if len(one_away) != 1 else ' is'} "
                    "one skill away"
                ),
                detail=(
                    f"{f.posting.title} at {f.posting.company} only lacks "
                    f"{f.missing[0]['skill']}."
                ),
                href=f"/jobs/{f.posting.id}",
            )
        )

    if len(fits) < 5:
        out.append(
            Advice(
                kind="more",
                headline="Add more postings for sharper advice",
                detail=(
                    "With fewer than five, one posting swings every number. "
                    "Aim for five to ten for each role you want."
                ),
                href="/jobs",
            )
        )
    elif len(scored) < len(fits):
        out.append(
            Advice(
                kind="more",
                headline=f"{len(fits) - len(scored)} postings list no skills",
                detail="They are left out of match scores. Reprocessing them may help.",
            )
        )
    return out


def _place_advice(places: list[PlaceInsight], total: int) -> Advice | None:
    """Where the jobs are: the place with the most postings, how they are
    worked there, how many need no move at all, and where the fit is
    best when that is somewhere else."""
    known = [p for p in places if p.kind != "unknown"]
    if total < 2 or not known:
        return None
    top = known[0]
    remote = next((p for p in known if p.kind == "remote"), None)
    cities_only = [p for p in known if p.kind == "city"]

    if top.kind == "remote":
        headline = "Most of your jobs are remote"
        detail = f"{top.posting_count} of {total} can be done from anywhere."
        if cities_only:
            c = cities_only[0]
            detail += f" Of the rest, {c.name} has the most, with {c.posting_count}."
    else:
        headline = f"Most of your jobs are in {top.name}"
        how = [
            f"{n} {label}"
            for n, label in ((top.on_site_count, "on-site"), (top.hybrid_count, "hybrid"))
            if n
        ]
        detail = f"{top.posting_count} of {total} postings"
        detail += f", {' and '.join(how)}." if how else "."
        if remote is not None:
            detail += (
                f" Another {remote.posting_count} "
                f"{'are' if remote.posting_count != 1 else 'is'} remote, open from any city."
            )
        else:
            detail += " None are remote, so expect to be in an office at least part of the week."

    fits = [p for p in known if p.posting_count >= 2 and p.avg_match_pct is not None]
    best = max(fits, key=lambda p: (p.avg_match_pct or 0, p.posting_count), default=None)
    if best is not None and best.name != top.name:
        where = "remote roles" if best.kind == "remote" else best.name
        detail += f" Your best fit is in {where}, at {best.avg_match_pct}% average match."
    return Advice(kind="place", headline=headline, detail=detail, href="/jobs")
