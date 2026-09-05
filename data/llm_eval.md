# LLM evaluation (n=30, judge = gpt-4o-mini)

| variant | relevance | groundedness | citation_validity | hallucination_rate | cited_target_species | p50_latency_ms | avg_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v1` | 5 | 5 | 1.767 | 0 | 0.8 | 5992 | 0.000295 |
| `v2` | 4.6 | 4.6 | 1.967 | 0.1 | 0.767 | 5933 | 0.000335 |
| `v3` | 4.733 | 4.733 | 1.167 | 0.067 | 0.767 | 6348 | 0.000372 |

Best: **v1** (set `PROMPT_VARIANT=v1`)
