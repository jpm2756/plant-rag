"""Streamlit chat UI over the PlantRAG API, with retrieval transparency and feedback."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODES = [
    "hybrid_rerank_dual",
    "hybrid_rerank",
    "hybrid_rerank_norewrite",
    "hybrid",
    "dense",
    "sparse",
]
VARIANTS = ["v1", "v2", "v3"]

st.set_page_config(page_title="PlantRAG — USDA plant assistant", page_icon="🌿", layout="wide")


def api(method: str, path: str, **kwargs) -> dict[str, Any]:
    response = requests.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def stats() -> dict[str, Any]:
    try:
        return api("GET", "/stats")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


st.title("🌿 PlantRAG")
st.caption(
    "Ask about plant selection, site tolerances, propagation or safety. Answers are grounded "
    "in the USDA PLANTS database and its fact-sheet PDFs."
)

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=0)
    variant = st.selectbox("Prompt variant", VARIANTS, index=0)
    use_rewrite = st.toggle("Query rewriting", value=True)
    use_tools = st.toggle("Exact-lookup tools", value=True)
    judge = st.toggle("Score answer with LLM judge", value=False)

    st.divider()
    st.header("Knowledge base")
    info = stats()
    corpus = info.get("corpus") or {}
    if corpus:
        st.metric("Species", f"{corpus.get('species', 0):,}")
        st.metric("Documents", f"{corpus.get('documents', 0):,}")
        st.metric("PDF chunks", f"{corpus.get('pdf_chunks', 0):,}")
        st.metric("Vectors", f"{info.get('vector_points', 0):,}")
    elif info.get("error"):
        st.error(f"API unreachable: {info['error']}")
    st.caption(f"Model: `{info.get('model', '?')}` · Embeddings: `{info.get('dense_model', '?')}`")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])

EXAMPLES = [
    "Shade-tolerant native shrubs under 6 feet for acidic soil in the Northeast",
    "Is Abies balsamea toxic to livestock?",
    "How do I propagate switchgrass from seed?",
    "Drought-tolerant grasses that fix nitrogen for a dry roadside",
]
if not st.session_state.messages:
    st.write("**Try one of these:**")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES, strict=False):
        if column.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No documents retrieved.")
        return
    for source in sources:
        arms = ", ".join(source.get("arms") or [])
        score = source.get("rerank_score")
        score_label = (
            f"rerank {score:.3f}" if score is not None else f"rrf {source.get('score') or 0:.4f}"
        )
        with st.expander(
            f"[{source['context_index']}] {source['title']} — {source['section']} "
            f"({arms}, {score_label})"
        ):
            st.write(source.get("text") or "")
            links = []
            if source.get("profile_url"):
                links.append(f"[USDA profile]({source['profile_url']})")
            if source.get("source_url"):
                links.append(f"[Source PDF](https://plants.usda.gov{source['source_url']})")
            if links:
                st.markdown(" · ".join(links))


def render_result(result: dict[str, Any]) -> None:
    st.markdown(result["answer"])

    columns = st.columns(4)
    columns[0].metric("Latency", f"{result['latency_ms'] / 1000:.1f}s")
    columns[1].metric("Retrieved", result["n_results"])
    columns[2].metric("Tokens", result["prompt_tokens"] + result["completion_tokens"])
    columns[3].metric("Cost", f"${result['cost_usd']:.5f}")

    with st.expander("Retrieval detail", expanded=False):
        st.write(f"**Rewritten query:** {result['rewritten_query']}")
        st.write(
            f"**Archetype:** `{result.get('archetype')}` · **mode:** `{result['retrieval_mode']}`"
        )
        st.write("**Extracted filters:**")
        st.json(result.get("filters") or {}, expanded=False)
        if result.get("filters_relaxed"):
            st.warning("Filters returned nothing and were relaxed for this answer.")
        if result.get("tools_used"):
            st.write(f"**Tools called:** {', '.join(result['tools_used'])}")
        st.write(
            "**Stage latency (ms):** "
            f"rewrite {result['latency_rewrite_ms']} · retrieve {result['latency_retrieve_ms']} · "
            f"rerank {result['latency_rerank_ms']} · generate {result['latency_generate_ms']}"
        )
        if result.get("judge"):
            st.write("**Judge:**")
            st.json(result["judge"], expanded=False)

    st.write("**Sources**")
    render_sources(result.get("sources") or [])

    conversation_id = result.get("conversation_id")
    if not conversation_id:
        return
    state_key = f"feedback::{conversation_id}"
    if st.session_state.get(state_key):
        st.success("Thanks — feedback recorded.")
        return
    with st.form(key=f"form::{conversation_id}", clear_on_submit=True):
        comment = st.text_input("Optional comment", key=f"comment::{conversation_id}")
        left, right, _ = st.columns([1, 1, 6])
        up = left.form_submit_button("👍 Helpful")
        down = right.form_submit_button("👎 Not helpful")
        if up or down:
            try:
                api(
                    "POST",
                    "/feedback",
                    json={
                        "conversation_id": conversation_id,
                        "rating": 1 if up else -1,
                        "comment": comment or None,
                    },
                )
                st.session_state[state_key] = True
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not save feedback: {exc}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_result(message["result"])

question = st.chat_input("e.g. evergreen hedge for a windy coastal site with salt spray")
if not question:
    question = st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Retrieving from USDA PLANTS…"):
        try:
            result = api(
                "POST",
                "/ask",
                json={
                    "question": question,
                    "retrieval_mode": mode,
                    "prompt_variant": variant,
                    "use_rewrite": use_rewrite,
                    "use_tools": use_tools,
                    "judge": judge,
                },
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")
            result = None
    if result:
        st.session_state.messages.append({"role": "assistant", "result": result})
        st.rerun()
