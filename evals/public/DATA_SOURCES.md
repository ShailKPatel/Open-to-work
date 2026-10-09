# Public-dataset eval set: sources and limits

## Where the data comes from

[LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings),
published on Kaggle by arshkon under CC BY-SA 4.0, read from its Hugging Face
mirror `datastax/linkedin_job_listings` (about 124,000 postings). Each row
carries the posting's description and the structured fields its poster filled
in on LinkedIn's form: title, company, location, salary range, work type,
remote flag.

`sources.yaml` pins 200 job ids. Only the ids are committed. The text is
downloaded into `cache/` (gitignored) by `scripts/fetch_public_eval_data.py`,
with email addresses, phone numbers and named contact lines removed before it
is written. Company names stay, since they are part of a posting.

## How the sample was drawn

Deterministically: within each group, the eligible postings with the lowest
SHA-256 of a fixed salt and the job id, at most one per company. Eligible
means a description of 600 to 8,000 characters that is almost all ASCII.

| group | postings | what it is for |
| --- | --- | --- |
| software | 80 | extraction, and retrieval against the open-source portfolio |
| general-salary | 60 | extraction on non-software roles that list a salary |
| general | 60 | extraction on non-software roles without one |

## Dev and test

Each posting is assigned to dev (about 30%) or test (about 70%) by a hash of
its key (`app/evals/splits.py`), so the split never moves. Prompts, rules and
thresholds may be changed while looking at dev. Only test is reported. Once a
change has been made after looking at test results, test has become dev, and
a new test sample is needed.

## Labels

**Extraction** labels are the poster's own form fields, not written by anyone
on this project. A field is scored only when the description gives the model
a fair chance at it (`app/evals/public.py`, `expected_fields`):

- title, company, location: always, from the form. The posting text starts
  with the header a LinkedIn page shows (title, company, location), so these
  check that the model keeps the header instead of taking a title or place
  from the body. They are expected to be near perfect.
- salary: scored when the description states both ends of the form's range
  (one number when both ends are equal). When the form has no salary and the
  description states no pay in any form (no dollar amount, no "160k", no
  hourly rate), the expected answer is an empty string, which checks that the
  model does not invent one. Other postings are not scored on salary.

  Both parts of that rule were corrected after the first test run: it had
  expected two numbers for a single rate, and treated pay written without a
  dollar sign ("160k a year", "50HR") as no pay, marking the model wrong for
  reading it. The fix changed labels only; nothing in the app changed, and
  the run was re-scored from its saved output.
- employment type: scored when the description names the form's work type
  in words.
- remote: scored when the form allows remote work and the description says
  "remote". The form has no way to say on-site or hybrid, so those are not
  scored.

**Retrieval** labels (`retrieval_labels.yaml`) are judgments: for each software
posting, which skills in the composite open-source portfolio a reviewer would
accept as evidence for it. They were made from the posting text alone, before
the search was first run on these postings, and have not been changed since.

## What these numbers cannot tell you

- One labeler made the retrieval labels. There is no second labeler and no
  agreement figure for them.
- LinkedIn's form fields are what the poster typed and can be wrong or
  disagree with the description. Rows where they disagree are scored as the
  model's error.
- The portfolio is a stand-in for one developer, built from public
  repositories; real accounts have fewer, smaller projects.
- Postings are from 2023 and 2024, mostly in the United States, in English.
