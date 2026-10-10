# Retrieval designs compared: skillspan (125 pairs)

Generated 2026-10-10T04:22:45.956432+00:00.

| metric | entry 2 | entry 3 (current) |
| --- | --- | --- |
| hits p@5 | 0.566 [0.51, 0.63] | 0.547 [0.49, 0.60] |
| hits nDCG@10 | 0.774 [0.73, 0.82] | 0.759 [0.72, 0.80] |
| candidate pool recall | 0.930 [0.90, 0.96] | 0.993 [0.99, 1.00] |
| candidate skills p@5 | 0.478 [0.43, 0.53] | 0.467 [0.42, 0.51] |

Paired differences, each design minus the one before it (95% bootstrap):

- entry 3 (current) minus entry 2, hits p@5: -0.019 [-0.050, +0.010]
- entry 3 (current) minus entry 2, hits nDCG@10: -0.015 [-0.044, +0.015]
- entry 3 (current) minus entry 2, candidate pool recall: +0.063 [+0.038, +0.092]
- entry 3 (current) minus entry 2, candidate skills p@5: -0.011 [-0.032, +0.011]

BM25 p@5 0.426; current hybrid minus BM25: +0.122 [+0.080, +0.165].
