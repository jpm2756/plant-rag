# LLM evaluation (n=30, judge = gpt-4o-mini)

| variant | relevance | groundedness | citation_validity | answers_with_citation | invalid_index_rate | hallucination_rate | cited_target_species | p50_latency_ms | avg_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `v1` | 5 | 4.867 | 4.867 | 0.9 | 0 | 0.033 | 0.8 | 3891 | 0.000317 |
| `v2` | 4.867 | 4.733 | 4.7 | 0.9 | 0 | 0.067 | 0.8 | 4480 | 0.000372 |
| `v3` | 4.967 | 5 | 5 | 0.967 | 0 | 0 | 0.767 | 5397 | 0.000406 |

Best: **v3** (set `PROMPT_VARIANT=v3`)
