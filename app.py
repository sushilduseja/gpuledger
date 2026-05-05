"""
app.py -- GPULedger: AI training cost calculator.
Run: streamlit run app.py
"""

import math
import pandas as pd
import streamlit as st

from hardware import (
    DEFAULT_MFU,
    GPU_CATALOG,
    GPU_BY_NAME,
    REFERENCE_RUNS,
    RunResult,
    evaluate,
    fmt_cost,
    fmt_flops,
    fmt_hours,
    gpu_hours,
    total_cost_usd,
    wall_clock_hours,
)

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="GPULedger",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { max-width: 800px; padding-top: 2rem; }
        .stMetric label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
        .stDataFrame { border-radius: 6px; }
        code { font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Header --------------------------------------------------------------------
st.markdown("## 🖥️ GPULedger")
st.caption(
    "Convert a FLOPs compute budget into GPU-hours and USD. "
    "Makes AI training compute physically and financially concrete."
)
st.divider()

# -- FLOPs input ---------------------------------------------------------------
st.markdown("#### Compute budget")

input_mode = st.radio(
    "Input method",
    ["Load a reference training run", "Enter FLOPs manually"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_mode == "Load a reference training run":
    run_label = st.selectbox("Reference run", list(REFERENCE_RUNS.keys()))
    ref = REFERENCE_RUNS[run_label]
    flops_input = ref["flops"]
    if not ref["confirmed"]:
        st.caption(f"⚠️ {run_label} FLOPs figure is an external estimate, not confirmed by the developer.")
    else:
        st.caption(f"FLOPs: {fmt_flops(flops_input)}")
else:
    col_exp, col_coeff = st.columns([2, 3])
    with col_exp:
        exponent = st.number_input(
            "Exponent (the N in 10^N)", min_value=15, max_value=30,
            value=23, step=1,
        )
    with col_coeff:
        coefficient = st.number_input(
            "Coefficient (the X in X x 10^N)", min_value=0.1, max_value=9.99,
            value=3.15, step=0.01, format="%.2f",
        )
    flops_input = coefficient * (10 ** exponent)
    st.caption(f"= {fmt_flops(flops_input)} FLOPs")

st.divider()

# -- Training parameters -------------------------------------------------------
st.markdown("#### Training parameters")

p1, p2 = st.columns(2)

with p1:
    num_gpus = st.number_input(
        "Number of GPUs", min_value=1, max_value=100_000, value=64, step=1,
    )
    st.caption("Parallel GPUs in the training cluster. Total cost stays fixed; wall-clock time falls.")

with p2:
    mfu_pct = st.slider(
        "Model FLOPs Utilization (MFU) %", min_value=10, max_value=70,
        value=int(DEFAULT_MFU * 100), step=5,
    )
    st.caption(
        "Fraction of peak GPU throughput actually used. "
        "10-25%: basic setup. 30-45%: well-optimized. 50-70%: expert-level."
    )

mfu = mfu_pct / 100.0

st.divider()

# -- GPU selector & primary result ---------------------------------------------
st.markdown("#### GPU")

gpu_name = st.selectbox(
    "Primary GPU",
    [g.name for g in GPU_CATALOG],
    index=0,
)
selected_gpu = GPU_BY_NAME[gpu_name]
st.caption(selected_gpu.notes)

result: RunResult = evaluate(flops_input, selected_gpu, mfu, num_gpus)

# -- Primary metrics -----------------------------------------------------------
st.divider()
m1, m2, m3, m4 = st.columns(4)

m1.metric("GPU-hours", f"{result.total_gpu_hours:,.0f}")
m1.caption("Total compute time across all GPUs. This is what you pay for.")

m2.metric("Wall-clock time", fmt_hours(result.wall_hours))
m2.caption(f"Actual elapsed time with {num_gpus:,} GPUs running in parallel.")

m3.metric("Total cost", fmt_cost(result.cost_usd))
m3.caption(f"At ~${selected_gpu.spot_usd_hr:.2f}/hr spot. Prices vary by provider.")

m4.metric("Effective TFLOPS", f"{selected_gpu.bf16_tflops * mfu:.0f}")
m4.caption(f"Peak {selected_gpu.bf16_tflops:.0f} TFLOPS x {mfu_pct}% MFU.")

# -- Step-by-step math ---------------------------------------------------------
with st.expander("How these numbers were calculated", expanded=False):
    eff_flops_per_sec = selected_gpu.bf16_tflops * 1e12 * mfu
    raw_gpu_secs = flops_input / eff_flops_per_sec
    raw_gpu_hrs = raw_gpu_secs / 3600

    math_text = (
        f"Formula:  GPU-hours = FLOPs / (peak_TFLOPS x 10^12 x MFU x 3600)\n"
        f"\n"
        f"Inputs:\n"
        f"  FLOPs              = {fmt_flops(flops_input)}\n"
        f"  GPU peak BF16      = {selected_gpu.bf16_tflops:.0f} TFLOPS\n"
        f"  MFU                = {mfu_pct}%  =  {mfu:.2f}\n"
        f"  Cluster size       = {num_gpus:,} GPUs\n"
        f"\n"
        f"Step 1 -- effective throughput per GPU:\n"
        f"  {selected_gpu.bf16_tflops:.0f} x 10^12 x {mfu:.2f} = {eff_flops_per_sec:.3e} FLOPS/sec\n"
        f"\n"
        f"Step 2 -- total GPU-seconds:\n"
        f"  {fmt_flops(flops_input)} / {eff_flops_per_sec:.3e} = {raw_gpu_secs:,.0f} sec\n"
        f"\n"
        f"Step 3 -- convert to GPU-hours:\n"
        f"  {raw_gpu_secs:,.0f} / 3600 = {raw_gpu_hrs:,.1f} GPU-hours\n"
        f"\n"
        f"Step 4 -- wall-clock time with {num_gpus:,} GPUs:\n"
        f"  {raw_gpu_hrs:,.1f} / {num_gpus:,} = {result.wall_hours:,.1f} hours  ({fmt_hours(result.wall_hours)})\n"
        f"\n"
        f"Step 5 -- cost:\n"
        f"  {raw_gpu_hrs:,.1f} GPU-hours x ${selected_gpu.spot_usd_hr:.2f}/hr = {fmt_cost(result.cost_usd)}\n"
        f"\n"
        f"Note: total cost is independent of cluster size -- more GPUs means less time, same money."
    )
    st.code(math_text, language=None)

st.divider()

# -- All GPU comparison --------------------------------------------------------
st.markdown("#### All GPUs — same compute budget")

rows = []
for gpu in GPU_CATALOG:
    gh = gpu_hours(flops_input, gpu, mfu)
    wh = wall_clock_hours(gh, num_gpus)
    cost = total_cost_usd(gh, gpu)
    rows.append({
        "GPU": gpu.name,
        "Gen": gpu.generation,
        "VRAM": f"{gpu.vram_gb} GB",
        "Peak BF16": f"{gpu.bf16_tflops:.0f} TFLOPS",
        "GPU-hours": f"{gh:,.0f}",
        "Wall-clock": fmt_hours(wh),
        "Cost (spot)": fmt_cost(cost),
        "Price/hr": f"${gpu.spot_usd_hr:.2f}",
    })

df = pd.DataFrame(rows)
# Highlight selected GPU row
st.dataframe(df, width='stretch', hide_index=True)
st.caption(
    f"Wall-clock assumes {num_gpus:,} GPUs at {mfu_pct}% MFU. "
    "Spot prices are approximate mid-2025 market rates (CoreWeave, Lambda, Together AI). "
    "On-demand rates run 30-80% higher."
)

st.divider()

# -- Governance insight --------------------------------------------------------
gh_gpt3_h100 = gpu_hours(3.15e23, GPU_BY_NAME["H100 SXM5 80GB"], mfu)
cost_gpt3_h100 = total_cost_usd(gh_gpt3_h100, GPU_BY_NAME["H100 SXM5 80GB"])

gh_llama405_h100 = gpu_hours(3.65e25, GPU_BY_NAME["H100 SXM5 80GB"], mfu)
cost_llama405_h100 = total_cost_usd(gh_llama405_h100, GPU_BY_NAME["H100 SXM5 80GB"])

st.info(
    f"**Why this matters for governance.**  "
    f"Retraining GPT-3 today on H100s at {mfu_pct}% MFU costs approximately {fmt_cost(cost_gpt3_h100)}. "
    f"A LLaMA 3 405B-scale run costs roughly {fmt_cost(cost_llama405_h100)}. "
    f"Frontier models above that scale are accessible to fewer than a dozen organisations globally. "
    f"Compute cost is not just an engineering constraint -- it is the natural chokepoint that makes "
    f"hardware-level governance tractable."
)

# -- Footer --------------------------------------------------------------------
st.divider()
st.caption(
    "GPU specs: NVIDIA product briefs (BF16 dense, no sparsity). "
    "Pricing: approximate spot rates, mid-2025. "
    "MFU framing: Korthikanti et al. (2022), Chowdhery et al. (2022)."
)
