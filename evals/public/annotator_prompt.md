# Retrieval relevance annotator prompt

Fixed before validation (docs/RETRIEVAL_IMPROVEMENTS.md, entry 4) and used
unchanged for every batch. `{SKILLS_FILE}`, `{BATCH_FILE}` and `{OUT_FILE}`
are the only parts filled in per batch.

---

You are labeling job postings for a research dataset. Work only from the
two input files named below. Do not open, search or read any other file,
and do not run commands other than reading those two files and writing
your output file.

**Inputs**

- `{SKILLS_FILE}`: the skill names in one developer's portfolio, one per
  line. These are the only labels you may use, spelled exactly as listed.
- `{BATCH_FILE}`: a JSON list of job postings, each `{"key", "text"}`.

**Task**

For each posting, list which portfolio skills a hiring reviewer would
accept as evidence for that posting: if the developer showed work in that
skill, would a reviewer of this posting count it as relevant to the job?

This is a judgment, not a string match:

- A skill can be relevant without being named. An iOS posting that names
  only Swift still makes SwiftUI and UIKit relevant. A posting asking for
  "strong coding skills in any language" for a backend role still makes
  backend languages and frameworks relevant.
- A skill named in passing, or in a list of things the company uses
  elsewhere, is relevant only if a reviewer for this role would care.
- Generic practices (Unit Testing, Continuous Integration, REST APIs and
  similar) are relevant when the role clearly involves them, whether or
  not the posting uses those exact words.
- Leave the list empty when the portfolio offers nothing for the posting
  (sales, retail, course writing, non-software roles).
- Prefer the skills that matter for the role. Do not add every loosely
  related skill.

**Output**

Write `{OUT_FILE}` as one JSON object mapping each posting's `key` to a
list of skill names from the skills file, for every posting in the batch.
Then reply with only the word `done` and the number of postings labeled.
