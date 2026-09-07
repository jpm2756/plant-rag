# Retrieval evaluation (n=238)

| mode | hit-rate@5 | MRR@5 | species hit-rate@5 |
| --- | --- | --- | --- |
| `sparse` | 0.496 | 0.390 | 0.681 |
| `dense` | 0.487 | 0.422 | 0.655 |
| `hybrid` | 0.471 | 0.419 | 0.676 |
| `hybrid_rerank` | 0.517 | 0.465 | 0.698 |
| `hybrid_rerank_norewrite` | 0.626 | 0.568 | 0.744 |
| `hybrid_rerank_dual` | 0.626 | 0.567 | 0.744 |

Best: **hybrid_rerank_norewrite**

## Hit-rate by query archetype

| mode | care_howto | comparison | name_lookup | safety | site_recommendation | trait_filter |
| --- | --- | --- | --- | --- | --- | --- |
| `sparse` | 0.649 | 0.500 | 0.653 | 0.778 | 0.231 | 0.336 |
| `dense` | 0.784 | 0.500 | 0.551 | 0.593 | 0.385 | 0.345 |
| `hybrid` | 0.649 | 0.500 | 0.571 | 0.667 | 0.231 | 0.345 |
| `hybrid_rerank` | 0.757 | 0.500 | 0.612 | 0.704 | 0.462 | 0.354 |
| `hybrid_rerank_norewrite` | 0.892 | 1.000 | 0.735 | 0.778 | 0.462 | 0.464 |
| `hybrid_rerank_dual` | 0.892 | 1.000 | 0.735 | 0.778 | 0.462 | 0.464 |
