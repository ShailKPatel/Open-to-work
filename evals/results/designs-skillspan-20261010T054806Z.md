# Retrieval designs compared: skillspan (128 pairs)

Generated 2026-10-10T05:48:06.448166+00:00.

| metric | entry 2 | entry 3 | entry 8 (current) |
| --- | --- | --- | --- |
| hits p@5 | 0.631 [0.58, 0.68] | 0.625 [0.58, 0.68] | 0.634 [0.59, 0.68] |
| hits nDCG@10 | 0.780 [0.74, 0.82] | 0.790 [0.75, 0.83] | 0.801 [0.76, 0.84] |
| candidate pool recall | 0.902 [0.87, 0.93] | 0.954 [0.93, 0.97] | 0.956 [0.93, 0.97] |
| candidate skills p@5 | 0.541 [0.50, 0.59] | 0.539 [0.50, 0.58] | 0.553 [0.51, 0.60] |

Paired differences, each design minus the one before it (95% bootstrap):

- entry 3 minus entry 2, hits p@5: -0.006 [-0.033, +0.020]
- entry 3 minus entry 2, hits nDCG@10: +0.009 [-0.014, +0.033]
- entry 3 minus entry 2, candidate pool recall: +0.052 [+0.023, +0.081]
- entry 3 minus entry 2, candidate skills p@5: -0.002 [-0.020, +0.017]
- entry 8 (current) minus entry 3, hits p@5: +0.009 [+0.002, +0.020]
- entry 8 (current) minus entry 3, hits nDCG@10: +0.012 [+0.003, +0.022]
- entry 8 (current) minus entry 3, candidate pool recall: +0.003 [+0.000, +0.007]
- entry 8 (current) minus entry 3, candidate skills p@5: +0.014 [+0.002, +0.028]

BM25 p@5 0.487; current hybrid minus BM25: +0.147 [+0.103, +0.191].
