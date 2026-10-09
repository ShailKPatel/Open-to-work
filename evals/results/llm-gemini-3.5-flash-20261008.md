# LLM evals on gemini/gemini-3.5-flash (2026-10-08)

The model the app is configured with. Run on free-tier keys, which allow 20
requests a day per project, so this covers the two parts that matter most:
extraction on the real postings and the held-out subtle judge set. The full
suite also ran on gemini-flash-lite-latest (llm-20261008T124039Z.md).

### job extraction (real postings)

28 items, 0 failed calls.

| field | accuracy | scored |
| --- | --- | --- |
| company | 1.000 | 28 |
| employment_type | 1.000 | 6 |
| experience_required | 0.944 | 18 |
| location | 0.875 | 24 |
| salary_range | 1.000 | 27 |
| seniority | 0.917 | 12 |
| title | 1.000 | 28 |
| work_mode | 1.000 | 14 |
| skills (recall) | 0.948 | 23 |

Scored before the location matcher accepted the same city with its region
written differently; under the current matcher the three location
mismatches below are matches, so location is 24/24.

<details><summary>Mismatches</summary>

- robinhood-8246088.location: expected 'Toronto, ON', got 'Toronto, Canada'
- robinhood-5319465.location: expected 'Toronto, ON', got 'Toronto, Canada'
- airbnb-8257909.experience_required: expected '10+ years', got '10+ years of industry experience with a BS/Masters and 2+ years with a PhD'
- airbnb-7532824.seniority: expected 'Manager', got ''
- brex-8502529002.location: expected 'New York, NY', got 'New York, New York, United States'

</details>

### Groundedness judge against held-out subtle cases

21 of 30 bullets judged; 3 more calls and the remaining 6 bullets were refused by the free-tier daily quota, not answered.

- Accuracy: 1.000
- Cohen's kappa: 1.000
- Ungrounded bullets caught: 1.000

| how the bullet was made | judge agrees | bullets |
| --- | --- | --- |
| adjacent_tech | 1.000 | 5 |
| generalization | 1.000 | 2 |
| grounded_inference | 1.000 | 7 |
| role_inflation | 1.000 | 1 |
| scope_shift | 1.000 | 3 |
| unstated_number | 1.000 | 3 |

Not yet run on this model: synthetic job extraction, resume extraction, the
standard judge set and extraction under prompt injection.
