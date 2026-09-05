# Retrieval evaluation (n=238)

| mode | hit-rate@5 | MRR@5 | species hit-rate@5 |
| --- | --- | --- | --- |
| `sparse` | 0.433 | 0.356 | 0.626 |
| `dense` | 0.445 | 0.402 | 0.605 |
| `hybrid` | 0.445 | 0.406 | 0.634 |
| `hybrid_rerank` | 0.483 | 0.424 | 0.651 |
| `hybrid_rerank_norewrite` | 0.622 | 0.567 | 0.744 |

Best: **hybrid_rerank_norewrite**

## Hit-rate by query archetype

| mode | care_howto | comparison | name_lookup | safety | site_recommendation | trait_filter |
| --- | --- | --- | --- | --- | --- | --- |
| `sparse` | 0.540 | 0.500 | 0.633 | 0.630 | 0.231 | 0.282 |
| `dense` | 0.757 | 0.500 | 0.551 | 0.481 | 0.308 | 0.300 |
| `hybrid` | 0.622 | 0.500 | 0.612 | 0.518 | 0.308 | 0.309 |
| `hybrid_rerank` | 0.676 | 0.500 | 0.673 | 0.593 | 0.385 | 0.318 |
| `hybrid_rerank_norewrite` | 0.892 | 1.000 | 0.735 | 0.741 | 0.462 | 0.464 |
