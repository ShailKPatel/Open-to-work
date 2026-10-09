# Labeling generated bullets

`generated_bullets.yaml` holds project bullets the app's resume writer
produced (`scripts/collect_generated_bullets.py`), each with the evidence
the writer was shown. `generated_labels.yaml` holds one label per bullet.
`perturbed_test.yaml` holds test-split bullets changed to carry one
unsupported claim each.

## The rule

A bullet is **grounded** when every concrete claim in it is stated in its
evidence, or is a conservative paraphrase of it. Concrete claims are:

- technologies used, and what each was used for
- what was built, and what it does
- numbers: scale, speed, counts, percentages
- who the work was for or with: a team, a company, users, customers,
  production use
- outcomes, adoption, awards, publications, certifications

Wording with no checkable content ("robust", "modern", "scalable") does not
make a bullet ungrounded on its own. This is the same rule the judge's
prompt states (`app/evals/groundedness.py`), so the judge is measured against
the standard it is asked to apply.

A label is **borderline** when the call could reasonably go either way, for
example a bullet that links two things the evidence lists separately
("offline favourites with Firebase"). Borderline bullets are reported apart,
so agreement on clear cases is not diluted by judgment calls.

## How the labels were made

- Labeled from the bullet and its evidence alone, before the judge was run
  on any of them, and not changed after its results were read.
- The perturbed set was written from test-split bullets labeled grounded,
  one claim per bullet, five bullets for each kind of claim, and frozen
  before the judge saw it.
- Dev and test are split by posting, so every bullet from one resume is on
  the same side.

## Limits

- One labeler, and no second labeler to measure agreement between people.
  Borderline cases are where a second labeler would most likely differ.
- The writer's own error rate is low (about 4 percent), so the judge's
  ability to catch unsupported claims is measured mostly on the perturbed
  set, whose claims were written by hand rather than produced by the writer.
- Bullets for the open-source portfolio describe repositories the account
  did not write. The label judges a bullet against its evidence, not
  authorship.
