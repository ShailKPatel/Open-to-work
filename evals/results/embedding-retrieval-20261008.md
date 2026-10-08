# Embedding models for retrieval (2026-10-08)

Skill evidence pairs only, every model with its own query and document prefixes, scored alone (dense, one query) and inside the hybrid search the app runs (scripts/compare_embedding_models.py). Embed time is the profile's documents on a laptop CPU.

## real-text set

| model | system | p@5 | nDCG@10 | MRR | embed s |
| --- | --- | --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | dense | 0.423 | 0.427 | 0.704 | 18.8 |
| BAAI/bge-small-en-v1.5 | hybrid | 0.777 | 0.726 | 0.913 | 18.8 |
| BAAI/bge-base-en-v1.5 | dense | 0.554 | 0.535 | 0.766 | 8.5 |
| BAAI/bge-base-en-v1.5 | hybrid | 0.823 | 0.746 | 0.952 | 8.5 |
| mixedbread-ai/mxbai-embed-large-v1 | dense | 0.508 | 0.464 | 0.754 | 55.6 |
| mixedbread-ai/mxbai-embed-large-v1 | hybrid | 0.808 | 0.736 | 0.962 | 55.6 |
| intfloat/e5-base-v2 | dense | 0.646 | 0.644 | 0.946 | 10.2 |
| intfloat/e5-base-v2 | hybrid | 0.808 | 0.740 | 0.974 | 10.2 |
| thenlper/gte-base | dense | 0.446 | 0.488 | 0.685 | 20.5 |
| thenlper/gte-base | hybrid | 0.792 | 0.743 | 0.948 | 20.5 |
| sentence-transformers/all-MiniLM-L6-v2 | dense | 0.554 | 0.498 | 0.708 | 6.8 |
| sentence-transformers/all-MiniLM-L6-v2 | hybrid | 0.823 | 0.742 | 0.969 | 6.8 |

## synthetic set

| model | system | p@5 | nDCG@10 | MRR | embed s |
| --- | --- | --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | dense | 0.669 | 0.810 | 0.860 | 19.1 |
| BAAI/bge-small-en-v1.5 | hybrid | 0.846 | 0.984 | 1.000 | 19.1 |
| BAAI/bge-base-en-v1.5 | dense | 0.731 | 0.852 | 0.941 | 10.6 |
| BAAI/bge-base-en-v1.5 | hybrid | 0.840 | 0.973 | 0.986 | 10.6 |
| mixedbread-ai/mxbai-embed-large-v1 | dense | 0.589 | 0.748 | 0.855 | 81.5 |
| mixedbread-ai/mxbai-embed-large-v1 | hybrid | 0.846 | 0.972 | 0.971 | 81.5 |
| intfloat/e5-base-v2 | dense | 0.720 | 0.839 | 0.950 | 11.8 |
| intfloat/e5-base-v2 | hybrid | 0.857 | 0.986 | 1.000 | 11.8 |
| thenlper/gte-base | dense | 0.651 | 0.772 | 0.874 | 28.6 |
| thenlper/gte-base | hybrid | 0.857 | 0.985 | 0.986 | 28.6 |
| sentence-transformers/all-MiniLM-L6-v2 | dense | 0.737 | 0.849 | 0.931 | 7.8 |
| sentence-transformers/all-MiniLM-L6-v2 | hybrid | 0.840 | 0.978 | 0.986 | 7.8 |

Under hybrid search every model lands within about 0.05 precision@5 of the others, inside the bootstrap interval width at this sample size, while alone they range from 0.42 to 0.65 on real text. bge-base-en-v1.5 is kept: tied best on real text, and switching would re-embed every stored vector for no measured gain.
