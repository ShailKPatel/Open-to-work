# Groundedness judge: prompt change for scope claims (2026-10-08)

gemini-flash-lite-latest, judging each bullet against the evidence the
resume writer sees (project name, description, skills).

The held-out subtle set (judge_bullets_hard.yaml) showed the judge passing
bullets that claim a team, a company or an organisation for a personal
project: 1 of 3 caught. The prompt was changed to treat who the work was
for or with (team, company or client, organisation, users, customers,
production use) as a concrete claim needing evidence. A second held-out set
(judge_bullets_scope.yaml, 8 scope inflations and 8 grounded controls) was
written before the change and is the clean measure of it.

| Set | Before | After |
| --- | --- | --- |
| Scope set, 16 bullets (written before the change) | 0.938, kappa 0.875, 7/8 inflations caught | 1.000, kappa 1.000, 8/8 |
| Subtle set, 30 bullets (prompted the change, so not held out for it) | 0.933, kappa 0.867 | 1.000, kappa 1.000 |
| Standard set, 45 bullets | 1.000, kappa 1.000 | 1.000, kappa 1.000 |
| Grounded controls flagged in error | 0 | 0 |
