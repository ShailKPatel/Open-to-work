# Retrieval improvements

A log of changes to how the app searches a profile's skill evidence for a
job posting: what was measured, why it was low, what changed, and what it
did on held-out data. Each entry is written before its held-out run.

Metric throughout is precision@5 (p@5) unless stated: of the top five
evidence rows returned, the share a reviewer labeled relevant. Intervals
are 95 percent bootstrap intervals.

## Summary

Every change to retrieval, in order: what was wrong, the reasoning, and
what the change did on data it was never tuned on. Details are in the
numbered entries below.

| # | Change | Why | Did it help? (held-out) |
| --- | --- | --- | --- |
| 1 | Skill sub-queries keyword-only, not dense | The first hybrid lost to BM25 on held-out postings (0.519 vs 0.642): a dense sub-query for a skill the profile lacks still ranks its nearest wrong row first, and dev queries had been built from the relevance labels | Yes. Fresh sample (50): 0.624 vs BM25 0.536, paired +0.088 [+0.048, +0.136] |
| 2 | Dense run scores each row by its best similarity to any named skill | One vector for a whole posting sat near none of its skills (74% of them absent from the profile), and fusing per-skill dense runs by rank repeated entry 1's failure | Yes. Dense alone 0.172 to 0.580 on fresh, 0.279 to 0.672 on fresh-2 (+0.393 [+0.337, +0.447]); hybrid 0.624 to 0.684 on fresh |
| 3 | Skill cap 12 to 40, candidate pool filled to 25 distinct skills, posting text scanned for missed skills at half weight | An oracle run showed the retriever near its ceiling when asked for the right skills, so the loss was in the query; the pool was also giving the writer about 18 skills, not 25 | Yes, smaller than dev said. Pool recall 0.823 to 0.850 on fresh-2, paired +0.026 [+0.010, +0.043] (dev: +0.069); row p@5 -0.020 [-0.042, +0.001] |
| 4 | A 150-posting held-out set labeled by a validated LLM annotator, and SkillSpan as an external check | 50 pairs confirm only large effects; hand labeling 150 more was the bottleneck; no set yet had labels made outside this project | Annotator passed every pre-set bar (kappa 0.849, system scores within 0.005). Fresh-2: hybrid 0.712 [0.67, 0.76] vs BM25 0.559, +0.153 [+0.118, +0.188]. SkillSpan (125 pairs, its annotators' spans): 0.547 vs 0.426, +0.122 [+0.080, +0.165] |
| 5 | Match skills under other names (StackOverflow tag synonyms) | Several remaining misses were "CI/CD" for Continuous Integration and similar | No. +0.007 pool recall and -0.019 p@5 on dev; removed, never scored on held-out data |
| 6 | SkillSpan label rule: read the parts of compound spans | 21 percent of SkillSpan's "wrong" top-five rows were skills the annotators had marked inside a longer span ("core-java/spring/spring-boot") | A measurement fix, not a search change: SkillSpan hybrid 0.547 to 0.625 (ceiling 0.73 to 0.805), 78 percent of ceiling against 79 on LinkedIn |
| 8 | Keyword runs use a tokenizer that keeps C, C++ and C# apart and drops the shared "js" token | The plain tokenizer turned "C++" and "C#" into "c", and "Node.js" into "node" + "js", which matched NextAuth.js and Next.js | A little. Re-scored on both held-out sets: fresh-2 p@5 0.712 to 0.724 (+0.012 [-0.001, +0.026]), SkillSpan 0.625 to 0.634 (+0.009 [+0.002, +0.020]) |
| 9 | Full-query keyword run on the skill list only, or dropped | Intruders like Data Quality Testing and Cloud Networking match ordinary words ("data", "cloud") in the posting's title and summary | No. Real-text candidate p@5 -0.062 [-0.123, -0.008] and pool recall down on three sets: the prose names skills the extraction missed. Reverted |
| 7 | Down-weight the dense run for asked-for skills the profile lacks | The remaining avoidable errors were nearest-name intruders (NextAuth.js for Node.js) | No. Worse on LinkedIn dev and SkillSpan; similarity cannot separate real synonyms from near misses (both medians about 0.54) |

Where it stands (current design, re-scored after entry 8): on 137
held-out postings the hybrid reaches p@5 0.724 [0.68, 0.77] against a
ceiling of 0.905 and BM25's 0.559 (+0.165 [+0.127, +0.204]), and gives
the writer 84% of the relevant skills. On SkillSpan it reaches 0.634
against BM25's 0.487 (+0.147 [+0.103, +0.191]). With the right skills asked for, the retriever is within 0.01 of
the ceiling on dev, so what is left is mostly skills a reviewer infers
rather than reads.

## Data and discipline

- **Dev sets**, used freely while designing: the synthetic set (10
  profiles, 68 pairs), the real-text set (26 pairs) and the LinkedIn dev
  split (21 pairs).
- **Held-out sets**, scored once per design and never tuned against: the
  LinkedIn test split (spent on 2026-10-09 by the first hybrid design,
  which lost to BM25) and the LinkedIn fresh sample (60 postings, 50
  scored pairs).

## 1. Hybrid search loses to BM25 on held-out data (2026-10-09)

The first hybrid design ran a dense and a BM25 query for every skill a
posting names, fused by reciprocal rank. On the LinkedIn test split it
scored 0.519 against BM25's 0.642. Two causes, found on dev:

- The real-text dev queries had been written from the same hand labels
  that defined relevance, so they named exactly the skills the profile
  had. Real extracted queries do not.
- Real postings name many skills a profile lacks. A dense query for one of
  those still ranks every document, and rank fusion counts its top match
  the same as an exact match on a skill the profile has.

Fix: per-skill queries became BM25 only (a skill with no keyword match
contributes nothing). Scored once on the fresh sample: hybrid 0.624
[0.55, 0.70], BM25 0.536, paired difference +0.088 [+0.048, +0.136]
(evals/results/public-fresh-20261009T013435Z.md).

## 2. Why dense search alone scored 0.17 (2026-10-09)

The same fresh run reported the old single-vector dense search at 0.172.
Breaking it down on the LinkedIn dev split (21 pairs,
`python -m scripts.compare_retrieval --public dev`) found three things, none of them the embedding model (six models had already
been compared and landed within about 0.05 of each other under hybrid
search: evals/results/embedding-retrieval-20261008.md).

**One vector for a whole posting.** The query is the posting's title,
company, summary and skill list in one embedding. On LinkedIn dev, 211 of
the 286 skills the extraction names (74 percent) are not in the profile at
all. One vector pulled toward all of them sits near none of the ones that
matter.

**Short, templated documents.** Each evidence row is embedded as
`Skill: Pandas. Evidence: declared_dependency.` Every document shares the
template, so it takes up a large share of a five-word text and compresses
the similarities between them. Embedding the bare skill name instead
lifted single-vector dense p@5 from 0.295 to 0.476 on LinkedIn dev.

**Rank fusion discards similarity.** Splitting the query into one dense
query per skill and fusing the rankings by rank was worse than one vector
(0.238): a skill the profile lacks still puts its nearest, wrong document
at rank one. Scoring each document by its maximum cosine similarity to any
named skill keeps that near miss weak, since an exact match on another
skill scores far higher.

Dense-only variants on LinkedIn dev (21 pairs, skill evidence):

| dense variant | p@5 | nDCG@10 |
| --- | --- | --- |
| one vector, whole posting (previous) | 0.295 | 0.342 |
| one query per skill, rank fusion | 0.238 | 0.284 |
| one query per skill, max similarity | 0.648 | 0.743 |
| max similarity, documents as bare skill names | 0.752 | 0.818 |

### What changed

The hybrid search's dense run is now max similarity per skill
(`_dense_scores` in app/retrieval/search.py), falling back to the full
query when a posting names no skills. It stays one run in the fusion, so
skills the profile lacks still cannot add rankings of their own.

Not shipped: embedding documents as bare skill names. It helps dense
search alone, but inside the hybrid it did not beat the current text
(LinkedIn dev 0.743 against 0.752), and it would mean re-embedding every
stored vector.

The eval now reports the dense run alone as its own system, next to the
old single-vector search, so the embedding side can be judged on its own.

### Dev results (p@5)

Before is the shipped design from entry 1; after is this change.

| set | dense alone, before | dense alone, after | hybrid, before | hybrid, after |
| --- | --- | --- | --- | --- |
| LinkedIn dev (21) | 0.295 | 0.676 | 0.686 | 0.752 |
| real-text (26) | 0.515 | 0.669 | 0.808 | 0.823 |
| synthetic (68) | 0.582 | 0.697 | 0.682 | 0.700 |

Paired differences, dense after minus before: +0.381 [+0.200, +0.552]
on LinkedIn dev, +0.154 [+0.054, +0.262] on real-text, +0.115 [+0.076,
+0.153] on synthetic. The hybrid gains are small and within noise on two
of the three sets; the main effect is on dense search alone.

Reports: evals/results/public-dev-20261009T020306Z.md,
real-text-20261009T020223Z.md, synthetic-20261009T020147Z.md.

### Held-out result

Scored once on the fresh sample (50 pairs) after the design above was
fixed (evals/results/public-fresh-20261009T020614Z.md):

| system | p@5 | nDCG@10 |
| --- | --- | --- |
| hybrid, this change | 0.684 [0.60, 0.77] | 0.777 [0.71, 0.84] |
| hybrid, entry 1 | 0.624 [0.55, 0.70] | 0.733 [0.66, 0.80] |
| dense alone, max similarity per skill | 0.580 [0.50, 0.66] | 0.683 [0.60, 0.76] |
| dense alone, single query (previous) | 0.172 [0.12, 0.24] | 0.234 [0.17, 0.31] |
| BM25 | 0.536 [0.47, 0.61] | 0.631 [0.56, 0.70] |

- Dense alone, after minus before: +0.408 [+0.308, +0.500]. Matches
  what dev predicted.
- Hybrid minus BM25: +0.148 [+0.092, +0.208], up from +0.088.
- Hybrid against entry 1: +0.060 in the mean. There is no paired interval
  for this one, since entry 1's per-pair scores were not kept, and the two
  unpaired intervals overlap, so treat it as a likely but unconfirmed gain.
- Dense alone against BM25: +0.044 [-0.020, +0.108]. Dense search now
  roughly ties keyword matching on its own instead of trailing it by 0.36.

**Ceiling.** Some postings have fewer than five relevant evidence rows, so
a perfect ranking cannot reach p@5 1.0. The best achievable mean on this
sample is 0.852 (0.905 on LinkedIn dev), so the hybrid's 0.684 is about 80
percent of it.

**Test set status.** The fresh sample has now scored two designs (entry 1
and this one), both fixed before their run. It is still unseen by any
tuning, but the next change should be scored on a newly drawn sample
(`python -m scripts.fetch_public_eval_data --add-fresh`, labeled before
the first run).

## 3. Finding where the remaining loss is (2026-10-09)

Entry 2 fixed the retriever. This entry asks what is left, in the order a
retrieval problem is usually worked: check the metric matches the product,
find the bottleneck with oracle runs, sort the misses by cause, then fix
the largest cause first. Dev sets only.

### The metric that matches the product

The resume builder does not show the model the top five rows. It hands it
up to 25 candidate skills, deduplicated by name
(`_candidate_skills` in app/resume_build/orchestrator.py), and the model
picks from those. A relevant skill at rank 20 still reaches the model; one
outside the pool never can. So the product metric is **candidate pool
recall**: the share of labeled skills present in what the model is given.
The harness now reports it (`recall_returned` on candidate skills, printed
by `scripts/run_synthetic_eval.py`). p@5 stays as a guardrail.

### Oracle: is the retriever or the query the bottleneck?

Rerun the search with the labeled skills as the query's skill list, so the
query asks for exactly the right things (LinkedIn dev, 21 pairs):

| query | p@5 | skill recall in top 25 |
| --- | --- | --- |
| app's LLM extraction | 0.752 | 0.856 |
| labeled skills (oracle) | 0.895 | 0.997 |
| p@5 ceiling | 0.905 | 1.0 |

Given the right skills, the retriever is within 0.01 of the ceiling. The
remaining loss is almost all in the query: which skills get asked for.

### Misses by cause

Every labeled skill missing from the top 10 candidate skills on LinkedIn
dev (43 of 137), sorted by an automatic check against the query and the
posting text:

| cause | misses | example |
| --- | --- | --- |
| Inference: the posting never names it; the labeler judged it implied | 19 | Continuous Integration for a posting naming Jenkins |
| Extraction: named in the posting text, not in the extracted skill list | 13 | "continuous integration", "microservices" written in the text |
| Ranking: named in the query, ranked low | 8 | Docker, React: past the 12-skill query cap |
| Alias: a near-match name in the query | 3 | "Relational Databases (PostgreSQL, ...)" for PostgreSQL |

### Three fixes, each ablated

- **F1, skill cap 12 to 40.** 9 of 21 dev queries named more than 12
  skills, and labeled skills past the cap were dropped unseen. The cap
  dated from when every skill added a noisy dense run; since entry 2 a
  skill the profile lacks adds almost nothing.
- **F2, scan the posting text.** The profile's own skill names found as
  whole words in the posting text but missing from the extraction are
  added as extra runs at half weight (`_mentioned_skills`,
  `_MENTIONED_WEIGHT` in app/retrieval/search.py). Names of three letters
  or fewer match case-sensitively, so "go" in prose is not Go and "C" is
  never read out of C++. No LLM call. Half weight because at full weight
  real-text p@5 fell 0.831 to 0.762: a skill named once in passing
  outranked one the extraction listed. 0.5 was the first round value
  tried, not tuned further on 47 pairs.
- **F3, fill the pool.** Measuring pool recall exposed that the candidate
  step searched 25 evidence rows and deduplicated by name, so a skill
  with three rows used three slots: the model got about 18 skills, not
  25. It now searches three rows per slot and stops at 25 distinct skills.

Candidate pool recall, through the eval harness:

| step | LinkedIn dev (21) | real-text (26) | synthetic (35) |
| --- | --- | --- | --- |
| before | 0.825 | 0.655 | 1.000 |
| F1 | 0.838 | 0.655 | 1.000 |
| F1 + F3 | 0.862 | 0.693 | 1.000 |
| F1 + F3 + F2 | **0.894** | **0.720** | 1.000 |

Guardrail: row-level p@5 moved 0.752 to 0.724 on LinkedIn dev, 0.823 to
0.815 on real-text and 0.700 to 0.697 on synthetic. Candidate-skill p@5
(the first five skills the model sees) went 0.629 to 0.648 on LinkedIn dev
and 0.631 to 0.654 on real-text. All three ship.

Reports: evals/results/public-dev-20261009T024527Z.md,
real-text-20261009T024437Z.md, synthetic-20261009T024357Z.md.

### Not fixed here

- **Inference**, the largest cause. Postings rarely write "Continuous
  Integration" when they name Jenkins, or "Java" when they name Spring
  Boot. The fix is a skill taxonomy (aliases plus tool-to-practice and
  framework-to-language links, e.g. from ESCO or Lightcast Open Skills),
  applied to the query and the documents. It needs care: some of these
  misses are the labeler's judgment that a related skill counts (Vue.js
  for a React posting), and widening the query to match a single labeler
  would fit the labels, not the task.
- **Label reliability.** One labeler, no agreement figure. A second
  labeler on 20 postings with Cohen's kappa would say how much of the
  remaining gap is label noise.
- **Sample size.** On the fresh sample the per-pair p@5 difference has a
  standard deviation of about 0.21. Detecting a +0.05 change at 80 percent
  power needs about 140 pairs; +0.03 needs about 380. At 50 pairs only
  effects near +0.08 or larger can be confirmed.

### Held-out result

Scored on a newly drawn 150-posting sample in entry 4: candidate pool
recall +0.026 [+0.010, +0.043], about a third of the dev estimate.

## 4. A larger held-out set with validated labels (protocol, 2026-10-09)

Written before any of the steps below were run. Nothing in it is changed
after a result is seen.

Entry 3 needs a held-out number, and the power calculation says a 50-pair
set confirms only large effects. Labeling 150 new postings by hand is the
bottleneck, so the labels come from an LLM annotator that is checked
against the existing human labels before it is trusted, the same way the
groundedness judge was validated.

### Annotator

- A separate agent, given only: the labeling criterion from
  evals/public/retrieval_labels.yaml (which portfolio skills a reviewer
  would accept as evidence for the posting), the portfolio's 90 skill
  names, and the posting text. It is not shown the code, the search, any
  system output, any eval result, or any human label.
- Postings are passed in batches; the prompt is fixed before validation
  and reused unchanged for the test sample.

### Validation (gate)

The annotator labels the 80 original LinkedIn software postings, whose
human labels already exist (all of them already spent as test data, so
nothing new is consumed). Agreement is measured over every (posting,
skill) decision, 80 x 90.

It passes only if all three hold:

1. Cohen's kappa against the human labels is at least 0.60.
2. Micro F1 of its label sets against the human ones is at least 0.70.
3. **The labels rank systems the same way.** The app's hybrid and BM25,
   scored on these 80 postings with the annotator's labels, differ from
   their scores with human labels by less than 0.05 p@5 each, and the
   hybrid-minus-BM25 difference keeps its sign.

If it fails, the annotator is not used and no held-out claim is made from
it. The prompt is not revised against these 80 postings and retried.

### Test sample

- 150 software postings drawn under a new salt (group `software-fresh-2`,
  `open-to-work-eval-2026-fresh-2`), skipping every posting already in
  sources.yaml.
- Queries are the app's own LLM extraction of each posting, as for the
  other LinkedIn sets.
- The annotator's labels are written to
  evals/public/retrieval_labels_llm.yaml before the search is run on
  these postings, and kept apart from the human labels.
- Scored once: the design as of entry 3, with entry 2's design (no skill
  cap change, no text scan, 25-row candidate pool) alongside it on the
  same pairs, so the difference gets a paired interval.

### External check: SkillSpan

SkillSpan (Zhang et al., NAACL 2022, CC BY 4.0) has StackOverflow job
postings with knowledge spans marked by trained annotators. For each
posting in its test split from the tech source, a portfolio skill is
relevant when an annotated knowledge span names it (case-insensitive
exact match after trimming). This tests only explicitly named skills and
favors keyword matching, including entry 3's text scan; it is reported as
a check that nothing regressed on independent human labels, not as the
headline.

### Validation result

The annotator labeled the 80 original software postings blind, in eight
batches of ten, with the prompt in evals/public/annotator_prompt.md.

| criterion | bar | result |
| --- | --- | --- |
| Cohen's kappa, 80 postings x 90 skills | >= 0.60 | **0.849** |
| micro F1 against the human labels | >= 0.70 | **0.859** (precision 0.810, recall 0.914) |
| hybrid p@5, human vs annotator labels | < 0.05 apart | 0.756 vs 0.751 |
| BM25 p@5, human vs annotator labels | < 0.05 apart | 0.630 vs 0.636 |
| hybrid minus BM25 keeps its sign | yes | +0.126 vs +0.115 |

Passed on every criterion. The annotator is somewhat more generous than
the human labeler (500 labels against 443). More labels per posting
raise the p@5 ceiling a little; system scores barely move.

### Test labels

The 150 `software-fresh-2` postings were labeled by the same prompt, in
15 batches of ten, and written to evals/public/retrieval_labels_llm.yaml
(sha256 7e34486559df74a17f8ff20d527a136493371ff098618f4f1cd8b80e6d6dec66)
before the search ran on any of them: 831 labels, 13 postings with none.

### Change to the SkillSpan plan

The SkillSpan test split has only 31 tech postings. Since nothing in this
project is trained on SkillSpan, all three splits are used (140 tech
postings). Decided before any system was run on SkillSpan.

### Held-out result (LinkedIn fresh-2)

Extraction finished after the quota resets and extra keys (131 postings
by gemini-flash-latest, 19 by the flash-lite fallback when the first was
overloaded, the same mix as the earlier sets). Scored once, on all 150
postings, 137 of which have at least one relevant skill
(evals/results/designs-public-fresh2-20261009T050344Z.md,
public-fresh2-20261009T050627Z.md). Labels unchanged since they were
frozen (sha256 7e344865...).

Entry 3 against entry 2, paired over the same 137 pairs:

| metric | entry 2 | entry 3 | paired difference |
| --- | --- | --- | --- |
| candidate pool recall | 0.823 | **0.850** | **+0.026 [+0.010, +0.043]** |
| hits p@5 | 0.733 | 0.712 | -0.020 [-0.042, +0.001] |
| hits nDCG@10 | 0.797 | 0.789 | -0.008 [-0.025, +0.010] |
| candidate skills p@5 | 0.623 | 0.618 | -0.006 [-0.026, +0.013] |

- The pool recall gain holds on held-out data, with an interval above
  zero, but it is less than half what dev showed (+0.069). Dev, with 21
  pairs and the choices made on it, overstated it, as dev sets do.
- Row-level p@5 leans down by about the same amount as on dev; the
  interval just reaches zero. Entry 3 trades a little precision in the
  top five rows for more relevant skills in what the model is given.
- The first five candidate skills the model sees are unchanged.

All systems on the same pairs (p@5 ceiling 0.905):

| system | p@5 | nDCG@10 |
| --- | --- | --- |
| hybrid (current) | **0.712 [0.67, 0.76]** | 0.789 [0.75, 0.82] |
| dense alone, max similarity per skill | 0.672 [0.62, 0.72] | 0.747 [0.71, 0.78] |
| BM25 | 0.559 [0.52, 0.60] | 0.658 [0.62, 0.69] |
| dense alone, single query (original) | 0.279 [0.23, 0.32] | 0.299 [0.26, 0.34] |

- Hybrid minus BM25, p@5: **+0.153 [+0.118, +0.188]**.
- Dense alone (entry 2's change) minus the original single query:
  +0.393 [+0.337, +0.447]. Dense search alone now beats BM25, +0.112
  [+0.073, +0.150], where on the first test split it trailed by 0.35.
- With 137 pairs the intervals are about half as wide as on the 50-pair
  fresh sample, and the entry 2 results reproduce on a sample nearly
  three times the size, labeled independently of the first.

### SkillSpan result (independent human labels)

Extraction finished over several quota days; a Google project shut off
mid-run surfaced a key-failover bug (a 403 "project denied access" was
treated as a bad request rather than a blocked key), fixed in
app/core/llm.py before the last postings. Scored once, on all 140 postings, 125 of which name
at least one portfolio skill (evals/results/designs-skillspan-
20261010T042246Z.md, skillspan-20261010T042516Z.md). Labels are exact
name matches against spans SkillSpan's annotators marked, so only named
skills count, and the p@5 ceiling is 0.73 (many postings name fewer than
five of the portfolio's skills).

| system | p@5 | nDCG@10 |
| --- | --- | --- |
| hybrid (current) | **0.547 [0.49, 0.60]** | 0.759 [0.72, 0.80] |
| dense alone, max similarity per skill | 0.560 [0.51, 0.61] | 0.781 [0.74, 0.82] |
| BM25 | 0.426 [0.38, 0.47] | 0.636 [0.59, 0.68] |
| dense alone, single query (original) | 0.155 [0.13, 0.18] | 0.253 [0.21, 0.30] |

- Hybrid minus BM25: **+0.122 [+0.080, +0.165]**. The third independent
  sample, and the first with labels nobody on this project made, agrees
  with the two LinkedIn ones.
- Dense alone ties the hybrid here (-0.013 [-0.053, +0.026]) and beats
  BM25 by +0.134. Entry 2's change reproduces: +0.405 over the original
  single query.
- Entry 3 against entry 2: candidate pool recall 0.930 to 0.993, paired
  +0.063 [+0.038, +0.092]; hits p@5 -0.019 [-0.050, +0.010]. The pool
  gain is inflated here, as planned for: the labels are exact names found
  in the text, which is what entry 3's text scan looks for.
- Relative to its ceiling, the hybrid reaches 75 percent (0.547 of 0.73);
  on fresh-2 it reaches 79 percent (0.712 of 0.905).

## 5. Matching skills under other names: tried, not shipped (2026-10-09)

After entry 3, 15 of 137 labeled skills on LinkedIn dev were still missing
from the candidate pool. Several were the same skill under another name:
"CI/CD" and "CI/CD pipelines" for Continuous Integration, "SpringBoot" for
Spring Boot. The rest were inferred by the labeler (Pandas for a Python
machine learning role, REST APIs for a posting naming JSON). For those,
the embedding similarity between the missing skill and the nearest
skill the query named was 0.60 to 0.70, the same range as unrelated pairs
(Stripe and CSS, 0.63), so no similarity threshold separates them.

**Tried.** Map each extracted skill, and each part of a compound one
("CI/CD", "Relational Databases (PostgreSQL, DB2, or Oracle)"), through
StackOverflow's tag synonyms (the 2,500 most applied, CC BY-SA 4.0: ci to
continuous-integration, k8s to kubernetes, golang to go, sklearn to
scikit-learn) and compare names with case, spaces, hyphens and dots
ignored. A profile skill reached this way joined the query. Nothing in it
was written to fit a dev case: "Java EE with SpringBoot" stayed unmatched
rather than adding a rule for "with".

**Result (dev):**

| set | pool recall without | with | hits p@5 without | with |
| --- | --- | --- | --- | --- |
| LinkedIn dev (21) | 0.894 | 0.901 | 0.724 | 0.705 |
| real-text (26) | 0.720 | 0.720 | 0.815 | 0.815 |
| synthetic (35) | 1.000 | 1.000 | 0.697 | 0.697 |

About one more skill reached the pool on LinkedIn dev, and hits p@5 fell
0.019 (paired interval [-0.057, +0.000]). The real-text and synthetic
queries already use clean names, so nothing changed there. No measured
gain, so the code and data file were removed, not kept switched off.

**What this says about what is left.** On LinkedIn dev, candidate pool
recall is 0.894 against 0.997 for the oracle query. The remaining gap is
mostly skills a reviewer infers rather than reads, which is also where a
single labeler's judgment matters most. Further gains here should be
measured against more than one labeler before they are believed.

## 6. SkillSpan's label rule, corrected (2026-10-10)

Written before the corrected rule was run.

**What was found.** After the SkillSpan run, the top five results of the
hybrid search were sorted by why the labels counted them wrong (625
slots): 21.3 percent were a portfolio skill sitting inside a longer span
the annotators marked ("core-java/spring/spring-boot" for Spring Boot,
"angular/react.js" for React), 5.0 percent were named in the posting but
not marked, and 18.9 percent were not in the posting at all. The first
group is a defect in entry 4's rule, not in the search: the annotators
marked the skill, and exact whole-span matching threw it away.

**Corrected rule (v2), fixed here and not changed after.** A portfolio
skill is relevant when any part of an annotated knowledge span names it,
where:

- a span is split into parts on `/ , ; ( ) & |` and the words "and" and
  "or", and the whole span is kept as one more part;
- each part and each skill name is mapped through StackOverflow's tag
  synonyms (the same 2,500 most-applied pairs tried in entry 5: react.js
  to reactjs, golang to go), then compared with case, spaces, hyphens,
  dots and underscores ignored;
- comparison is of whole parts, never substrings, so "javascript" does
  not label Java.

No other change: same postings, same queries, same search, scored once
under v2 next to the v1 numbers above. The search is not changed in this
entry. Because the rule was corrected after seeing results, v2 numbers
are reported as a corrected measurement of the same run, not as a new
held-out result.

**Result under v2** (same run, corrected labels; evals/results/skillspan-20261010T043801Z.md):
128 pairs score (3 more postings now name a portfolio skill), and the p@5
ceiling rises from 0.73 to 0.805.

| system | p@5, rule v1 | p@5, rule v2 |
| --- | --- | --- |
| hybrid (current) | 0.547 | **0.625 [0.58, 0.68]** |
| dense alone, max similarity per skill | 0.560 | 0.623 |
| BM25 | 0.426 | 0.487 |

Hybrid minus BM25 under v2: +0.138 [+0.095, +0.181]. Relative to its
ceiling the hybrid reaches 78 percent, against 79 percent on LinkedIn
fresh-2, so after the rule fix SkillSpan is not low; it was mismeasured.

## 7. Down-weighting asked-for skills the profile lacks: tried, not shipped (2026-10-10)

SkillSpan is used as a dev set from here on; it has been scored.

**What was found.** Under rule v2, 240 of 640 top-five slots are wrong,
but 125 of those are forced: the posting has fewer relevant rows than
five, so nothing could fill them. Of the 115 avoidable ones, the most
common intruders are rows the posting never asked for but that are the
nearest name to something it did: NextAuth.js for Node.js and React.js,
Cloud Networking for "cloud", Data Quality Testing for "performance
testing". Each comes from the dense run, which takes a skill the profile
lacks to its nearest row.

**Tried.** Split the dense run in two: skills with a keyword match in the
profile at full weight, the rest at a lower one.

| set | pool recall w=1.0 | w=0.5 | w=0.25 | hits p@5, w=0.5 minus w=1.0 |
| --- | --- | --- | --- | --- |
| LinkedIn dev | 0.894 | 0.845 | 0.846 | -0.019 [-0.048, +0.000] |
| real-text | 0.720 | 0.721 | 0.715 | +0.000 |
| synthetic | 1.000 | 1.000 | 1.000 | +0.003 |
| SkillSpan v2 | 0.954 | 0.937 | 0.947 | -0.025 [-0.045, -0.005] |

Worse everywhere it moved. Skills without a keyword match are not only
near misses: "k8s", "CI/CD" and "React.js" have no exact row either, and
the dense run is what finds Kubernetes, Continuous Integration and React
for them. Reverted.

**Why no similarity rule can fix it.** For every asked-for skill with no
keyword match, the cosine similarity of its nearest row was compared
between cases where that row is relevant and where it is not:

| set | nearest row relevant: median cosine | nearest row wrong: median cosine |
| --- | --- | --- |
| LinkedIn dev (41 / 121 skills) | 0.58 | 0.53 |
| SkillSpan v2 (108 / 706 skills) | 0.54 | 0.54 |

The distributions overlap almost completely, and none reach 0.75. With
this embedding model, a similarity threshold or weight cannot tell
"k8s to Kubernetes" from "Node.js to NextAuth.js". The next lever would
be a step that knows what the names mean rather than how close their
vectors are, such as asking the extraction model which of the profile's
own skills the posting asks for; that adds an LLM call per search and
needs its own held-out check.

## 8. A keyword tokenizer that knows technology names (2026-10-10)

**What was found.** The keyword runs shared the baseline's tokenizer,
which keeps only letters and digits. "C++" and "C#" both became "c" and
matched the skill C; "Node.js" became "node" and "js", and "js" matched
every JavaScript library in the profile (NextAuth.js, Next.js), one source
of the nearest-name intruders in entry 7.

**Change.** `keyword.tech_tokenize`, used only by the app's search
(`_TECH_TOKENS` in app/retrieval/search.py): C++, C#, F# and .NET become
single tokens (cpp, csharp, fsharp, dotnet), and "name.js" becomes the
name plus the joined form ("node", "nodejs"), never a bare "js", so
"React.js" still finds React. The BM25 baseline keeps the plain
tokenizer, so it stays the same baseline every result above was measured
against.

**Result (dev, paired against the plain tokenizer):**

| set | hits p@5 | candidate skills p@5 | pool recall |
| --- | --- | --- | --- |
| LinkedIn dev (21) | +0.000 | +0.010 [+0.000, +0.029] | +0.000 |
| real-text (26) | +0.000 | +0.000 | -0.010 [-0.029, +0.000] |
| synthetic (68) | +0.006 [+0.000, +0.015] | +0.000 | +0.000 |
| SkillSpan v2 (128) | +0.009 [+0.002, +0.020] | +0.014 [+0.002, +0.028] | +0.003 |

Small, mostly positive, no interval below zero. Kept because it fixes a
real defect at no measured cost, not because of the size of the gain.

**Held-out re-score.** With the design fixed, both held-out sets were
scored once more, entries 2, 3 and 8 side by side on the same pairs
(designs-public-fresh2-20261010T053810Z.md,
designs-skillspan-20261010T054806Z.md):

| set | metric | entry 3 | entry 8 (current) | paired difference |
| --- | --- | --- | --- | --- |
| fresh-2 (137) | hits p@5 | 0.712 | 0.724 | +0.012 [-0.001, +0.026] |
| fresh-2 (137) | candidate pool recall | 0.850 | 0.844 | -0.006 [-0.015, +0.000] |
| fresh-2 (137) | candidate skills p@5 | 0.618 | 0.626 | +0.009 [-0.004, +0.023] |
| SkillSpan v2 (128) | hits p@5 | 0.625 | 0.634 | +0.009 [+0.002, +0.020] |
| SkillSpan v2 (128) | candidate skills p@5 | 0.539 | 0.553 | +0.014 [+0.002, +0.028] |

Fresh-2 has now scored three designs (entries 3 and 8 here, plus the
original run of entry 3), each fixed before its run; the next change
needs a new sample. Current headline: fresh-2 hybrid 0.724 [0.68, 0.77]
vs BM25 0.559, +0.165 [+0.127, +0.204]; SkillSpan 0.634 [0.59, 0.68] vs
0.487, +0.147 [+0.103, +0.191].

**What this says about precision.** On fresh-2 the hybrid's p@5 of 0.712
is 79 percent of what is achievable (0.905), its first result is relevant
in 89 percent of postings (MRR 0.888), and nDCG@10 is 0.789. The remaining
errors are nearest-name rows that embedding similarity cannot tell from
real synonyms (entry 7). The levers left need a model that knows what the
names mean, at the cost of an LLM call per search: mapping the posting's
skills onto the profile's own skill list, or having the model rerank the
top candidates.

## 9. Keyword-matching the skill list instead of the whole posting: tried, not shipped (2026-10-10)

**Idea.** Several remaining intruders are multi-word skills that share an
ordinary word with the posting's title or summary (Data Quality Testing
for "data", Cloud Networking for "cloud", Embedded Systems for
"systems"). The full-query keyword run searches that prose. Two variants
were scored on the dev sets: that run over the extracted skill list only,
and no full-query keyword run at all (the per-skill runs stay).

**Result (paired against the current search):**

| set | candidate p@5, skills only | pool recall, skills only | candidate p@5, dropped |
| --- | --- | --- | --- |
| LinkedIn dev | -0.019 [-0.057, +0.000] | -0.014 [-0.041, +0.000] | -0.029 [-0.076, +0.000] |
| real-text | -0.062 [-0.123, -0.008] | -0.022 [-0.045, -0.003] | -0.062 [-0.123, -0.008] |
| synthetic | +0.000 | +0.000 | -0.006 |
| SkillSpan v2 | +0.005 [-0.005, +0.016] | -0.007 [-0.016, +0.000] | +0.000 |

Worse or flat everywhere it matters. The title and summary often name
skills the extraction left out, and that is worth more than the noise of
ordinary words. Reverted.
