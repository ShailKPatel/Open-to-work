# SkillSpan: sources and limits

## Where the data comes from

[SkillSpan](https://aclanthology.org/2022.naacl-main.366) (Zhang, Jensen,
Sonniks and Plank, NAACL 2022), CC BY 4.0, read from its Hugging Face copy
`jjzha/skillspan`. English job postings split into sentences, with skill
and knowledge spans marked by annotators trained on the paper's
guidelines. Organization names and other identifying text were already
anonymized by the authors.

Only the `tech` source (StackOverflow job postings) is used, from all
three splits: 140 postings. Nothing here is trained, so the splits have no
role. `scripts/fetch_skillspan.py` downloads the files into `cache/`
(gitignored).

## Labels

A skill in the composite open-source portfolio (app/evals/real.py) is
relevant to a posting when one of the posting's knowledge spans names it.
No one on this project labeled anything.

Rule v2 (current): each span is also read as its parts, split on
`/ , ; ( ) & |`, "and" and "or"; parts and skill names are mapped through
StackOverflow tag synonyms (`so_tag_synonyms.json`, the 2,500 most
applied, CC BY-SA 4.0) and compared whole, ignoring case, spaces,
hyphens, dots and underscores. Rule v1, the first one run, needed a whole
span to equal the name, so "angular/react.js" labeled neither Angular nor
React. The rule was corrected after the first scored run, which found 21
percent of the top five results were skills marked inside such spans;
both results are reported (docs/RETRIEVAL_IMPROVEMENTS.md entry 6).

## What these numbers cannot tell you

- Only skills a posting names count. A skill a reviewer would infer
  (SwiftUI for an iOS posting naming only Swift) is not relevant here,
  unlike the LinkedIn and real-text sets.
- Exact name matching favors keyword search, and in particular the app's
  scan of the posting text for skill names. Read it as a check that
  nothing regressed on independent labels, not as the headline.
- Variants outside the synonym list still do not match: a span "core java"
  labels Java only because "core-java" is a StackOverflow synonym of it.
- Postings are from 2020 and 2021.
