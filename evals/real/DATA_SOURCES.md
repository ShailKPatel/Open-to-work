# Real-text eval set: sources, privacy and limits

This project runs locally for one person at a time. It has no users, no
production traffic and no logs, so it cannot be evaluated the way a hosted
product would be (on real queries, with real outcomes). Its evals therefore
come from two sets built for the purpose:

| Set | Data | Labels | What it is good for |
| --- | --- | --- | --- |
| Synthetic (`evals/synthetic/`) | Invented people, roles, repos and postings | Rule-based: a document is relevant when it names a skill the posting asks for | Regression across many kinds of profile; runs in CI with no network |
| Real-text (this folder) | Public job postings and public open-source repositories | Labeled by hand against the criteria below | Checking the system on text nobody wrote for this project |

Neither set contains anything from the maintainer's own profile.

## Sources

**Job postings.** 28 postings from the public Greenhouse job boards of 14
employers, read through the same public API those boards use to display
them (`boards-api.greenhouse.io`). Listed by board and posting id in
`sources.yaml`. Chosen to cover a spread of roles: Android, iOS, frontend,
backend, full stack, machine learning, data science, data analysis,
analytics and data engineering, site reliability, cloud infrastructure,
firmware, engineering management and internships, plus two roles outside
software (sales, design) that a matcher should find nothing for.

**Repositories.** 21 public open-source repositories, each pinned to a
commit in `sources.yaml` with its license (MIT, BSD, Apache-2.0 or the
PostgreSQL license). Only the README and the root dependency manifests are
read. Together they stand in for one developer's portfolio spanning web
backends, frontends, mobile, infrastructure, data, machine learning and
embedded work. They are not one person's work and are not presented as
such.

## What is stored, and where

- Committed: posting ids, repository names and commits, licenses, the
  repositories' one-line descriptions, and the labels in `labels.yaml`.
- Not committed: the posting text and README text. They are downloaded by
  `scripts/fetch_real_eval_data.py` into `evals/real/cache/`, which is
  gitignored. Posting text belongs to the employers who wrote it, so it is
  kept as a local working copy for evaluation and not redistributed.

## Privacy

The sources were chosen so that no personal data is needed at all: job
postings describe roles, not people, and the repositories are read only for
what the project is and what it is built with. On download, posting text is
scrubbed of email addresses, phone numbers and labeled contact lines
(`app/evals/real.py`, `scrub_pii`), and that function is covered by tests.
Contributor names, commit authors and issue threads are never read.

No resumes or real work histories are used. No source of real resume text
was found that is both consented for this use and openly available, and an
anonymized career history can often still be traced back to its owner. Work
history (experience points) is therefore only evaluated on the synthetic
set.

## Labeling criteria

Labels are in `labels.yaml`, made by one labeler reading each posting and
README in full.

- **Repository skills**: what the README shows the project is built with or
  is. Dependencies the app already reads from manifests are not repeated.
  Things the README only links to or mentions as alternatives are left out.
- **Posting fields**: what the posting states. A field the posting is
  ambiguous about (two pay ranges, two numbers of years, a work mode implied
  but never said) is `null` and is not scored.
- **Posting skills**: the core technologies the posting names. Scored by
  recall, so an extractor is not penalised for also listing secondary ones.
- **Relevant skills**: which skills in the portfolio a reviewer would
  accept as evidence for the posting. A judgment rather than a string
  match: an iOS posting that only names Swift still makes SwiftUI and UIKit
  evidence relevant. Retrieval is scored against these.

## Limits

- **Small.** 28 postings and 26 scored queries. A difference of a few
  points between two systems is within noise at this size.
- **One labeler.** No second labeler, so there is no agreement figure. The
  criteria above are written down so a second pass can be compared.
- **Point in time.** Postings close. Once closed, a posting can only be read
  from a local cache, so a fresh clone can rebuild the set only while the
  postings are still open.
- **Composite portfolio.** Real profiles are smaller and more uneven than 21
  well-known repositories. Numbers here say how the ranking behaves on real
  text, not how it will do for any particular person.
- **No work history.** See Privacy above.
