"""
PhysicsFlow — AI Reservoir Engineering Assistant
Streamlit demo for Hugging Face Spaces / hackathon showcase.

Data questions are answered directly from tool results (synthetic Norne
baseline) without an LLM — eliminating hallucination entirely.
"""

from __future__ import annotations
import os
import sys
import time

import streamlit as st
import plotly.graph_objects as go

# ── Path resolution (local dev vs HF Spaces) ─────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)

for _candidate in [
    os.path.join(_repo_root, "engine"),   # local: spaces/ is inside repo
    os.path.join(_here, "engine"),         # HF Space: engine/ copied alongside app.py
]:
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break

from physicsflow.agent.context_provider import ReservoirContextProvider
from physicsflow.agent.tools import ReservoirTools
from physicsflow.agent.reservoir_agent import ReservoirAgent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhysicsFlow — AI Reservoir Assistant",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Norne field data...")
def bootstrap():
    ctx   = ReservoirContextProvider()
    ctx._seed_norne_baseline()           # Seed synthetic Norne well profiles
    agent = ReservoirAgent(context_provider=ctx)
    tools = ReservoirTools(ctx)
    return agent, tools

agent, tools = bootstrap()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ PhysicsFlow")
    st.caption("AI-Native Reservoir Simulation Platform")
    st.divider()

    st.markdown("### 📁 Demo Project")
    st.markdown("**Norne Field** — Norwegian North Sea")
    st.markdown("**Grid:** 46 × 112 × 22 cells")

    perf       = tools.get_well_performance("all")
    mm_data    = tools.get_data_mismatch_per_well()
    hm_summary = tools.get_hm_iteration_summary()

    well_count  = len(perf.get("well_summary", {}))
    above_count = len(mm_data.get("above_expectation", []))
    below_count = len(mm_data.get("below_expectation", []))

    col1, col2 = st.columns(2)
    col1.metric("Producers", well_count)
    col2.metric("Injectors", 9)
    col1.metric("Above target", above_count)
    col2.metric("Below target", below_count)

    st.divider()
    st.markdown("### ⚡ Status")
    st.success("PINO Surrogate — Trained ✓")
    hm_iters = hm_summary.get("n_iterations", 0)
    if hm_iters:
        st.info(f"αREKI — {hm_iters} iterations")
    else:
        st.warning("αREKI — Not started")
    st.metric("Baseline RMSE", hm_summary.get("baseline_rmse", "N/A"))

    st.divider()
    st.markdown("### 💬 Try asking")

    example_qs = [
        "Which wells are above and below expectations?",
        "Show me production profiles",
        "Break down data mismatch per well",
        "Summarise history matching status",
        "Which wells are matching poorly?",
    ]
    for q in example_qs:
        if st.button(q, use_container_width=True, key=f"btn_{q[:25]}"):
            st.session_state["_pending"] = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        agent.clear_history("demo")
        st.rerun()

    st.divider()
    st.caption(
        "Built with PhysicsFlow v2.0.5 · "
        "[GitHub](https://github.com/Danny024/PhysicsFlow)"
    )

# ── Welcome message ───────────────────────────────────────────────────────────
WELCOME = """\
**Welcome to PhysicsFlow!** 🛢️

I'm the AI reservoir engineering assistant for the **Norne field** demo \
(Norwegian North Sea — 46×112×22 grid, 22 producers, 9 injectors).

I answer well-performance and history-matching questions **directly from \
simulation data** — no hallucinations, no generic UI instructions, just \
real numbers:

- *"Which wells are performing above and below expectations?"*
- *"Break down the data mismatch per well"*
- *"Summarise the history matching status"*
- *"Show me production profiles"*

Use the sidebar buttons or type below to get started.
"""

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": WELCOME}]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("PhysicsFlow — AI Reservoir Assistant")
st.caption(
    "Physics-Informed Neural Operator · αREKI Ensemble History Matching · "
    "Norne Field (Norwegian North Sea)"
)

# ── Well performance chart (always visible) ───────────────────────────────────
with st.expander("📊 Well Performance Overview", expanded=False):
    well_summary = perf.get("well_summary", {})
    above_exp    = mm_data.get("above_expectation", [])
    below_exp    = mm_data.get("below_expectation", [])

    wells  = list(well_summary.keys())
    peaks  = [well_summary[w].get("peak_wopr_stbd", 0) for w in wells]
    colors = []
    for w in wells:
        if w in above_exp:
            colors.append("#2ECC71")
        elif w in below_exp:
            colors.append("#E74C3C")
        else:
            colors.append("#3498DB")

    fig = go.Figure(go.Bar(
        x=wells,
        y=peaks,
        marker_color=colors,
        text=[f"{p:,.0f}" for p in peaks],
        textposition="outside",
    ))
    fig.update_layout(
        title="Peak Oil Rate by Well (STB/day)",
        xaxis_title="Well",
        yaxis_title="Peak WOPR (STB/day)",
        height=380,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    fig.add_annotation(
        x=0.01, y=1.08, xref="paper", yref="paper",
        text="🟢 Above expectation  🔴 Below expectation  🔵 On target",
        showarrow=False, font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input handling ────────────────────────────────────────────────────────────
pending = st.session_state.pop("_pending", None)
prompt  = pending or st.chat_input("Ask about wells, production, or history matching...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""

        for chunk in agent.chat("demo", prompt):
            tok = chunk.get("token", "")
            if tok:
                full_response += tok
                placeholder.markdown(full_response + "▌")
            if chunk.get("is_done"):
                placeholder.markdown(full_response)
                break

    st.session_state["messages"].append(
        {"role": "assistant", "content": full_response}
    )
    st.rerun()
