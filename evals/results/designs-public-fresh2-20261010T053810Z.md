# Retrieval designs compared: public-fresh2 (137 pairs)

Generated 2026-10-10T05:38:10.860786+00:00.

| metric | entry 2 | entry 3 | entry 8 (current) |
| --- | --- | --- | --- |
| hits p@5 | 0.733 [0.69, 0.78] | 0.712 [0.67, 0.76] | 0.724 [0.68, 0.77] |
| hits nDCG@10 | 0.797 [0.76, 0.83] | 0.789 [0.75, 0.82] | 0.784 [0.75, 0.82] |
| candidate pool recall | 0.823 [0.79, 0.85] | 0.850 [0.82, 0.88] | 0.844 [0.81, 0.87] |
| candidate skills p@5 | 0.623 [0.58, 0.67] | 0.618 [0.58, 0.66] | 0.626 [0.58, 0.67] |

Paired differences, each design minus the one before it (95% bootstrap):

- entry 3 minus entry 2, hits p@5: -0.020 [-0.042, +0.001]
- entry 3 minus entry 2, hits nDCG@10: -0.008 [-0.025, +0.010]
- entry 3 minus entry 2, candidate pool recall: +0.026 [+0.010, +0.043]
- entry 3 minus entry 2, candidate skills p@5: -0.006 [-0.026, +0.013]
- entry 8 (current) minus entry 3, hits p@5: +0.012 [-0.001, +0.026]
- entry 8 (current) minus entry 3, hits nDCG@10: -0.005 [-0.026, +0.014]
- entry 8 (current) minus entry 3, candidate pool recall: -0.006 [-0.015, +0.000]
- entry 8 (current) minus entry 3, candidate skills p@5: +0.009 [-0.004, +0.023]

BM25 p@5 0.559; current hybrid minus BM25: +0.165 [+0.127, +0.204].
