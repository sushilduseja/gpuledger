# 🖥️ GPULedger

Convert a FLOPs compute budget into GPU-hours and USD across six GPU types. Makes AI training compute physically and financially concrete.

Part of a compute governance learning project. Companion to [FLOPscope](https://github.com/sushilduseja/flopscope).

---

## What it does

- Takes a FLOPs compute budget (from FLOPscope, a reference run, or manual entry)
- Calculates GPU-hours, wall-clock time, and total cost for any cluster size
- Compares H100 SXM5, H200, H100 PCIe, A100 SXM4, A100 PCIe, L40S side by side
- Adjusts for Model FLOPs Utilization (MFU) -- the gap between peak spec and real throughput
- Shows step-by-step math for every result

## Core formula

```
GPU-hours = FLOPs / (peak_BF16_TFLOPS x 10^12 x MFU x 3600)
Wall-clock = GPU-hours / num_GPUs
Cost       = GPU-hours x spot_price_per_hour
```

Total cost is independent of cluster size. More GPUs means less time, same money.

## What is MFU?

Model FLOPs Utilization is the fraction of a GPU's peak throughput that training actually achieves.
A GPU rated at 989 TFLOPS running at 35% MFU delivers 346 effective TFLOPS.

- 10-25%: basic or poorly optimised setup
- 30-45%: well-optimised production training
- 50-70%: expert-level (e.g. PaLM, Megatron-LM at scale)

## Why this matters for governance

Frontier training runs at LLaMA 3 405B scale cost roughly $15-30M at current spot prices.
GPT-4-class runs are estimated higher still. Fewer than a dozen organisations globally can sustain
this spending. Compute cost is not just an engineering constraint -- it is the natural chokepoint
that makes hardware-level AI governance tractable.

---

## Quickstart

```bash
git clone https://github.com/sushilduseja/gpuledger.git
cd gpuledger
uv sync
streamlit run app.py
```

Deploy free to [Streamlit Community Cloud](https://streamlit.io/cloud) -- no card required.

## Use with FLOPscope

```
FLOPscope  -->  estimates training FLOPs for a model spec
GPULedger  -->  converts those FLOPs into GPU-hours and cost
```

Take the FLOPs figure from FLOPscope and enter it manually in GPULedger.

## Project structure

```
gpuledger/
├── app.py          # Streamlit UI
├── hardware.py     # GPU catalog, specs, pricing, core calculations
├── requirements.txt
└── README.md
```

`hardware.py` has no UI dependencies -- import it in notebooks or scripts directly:

```python
from hardware import GPU_BY_NAME, evaluate, fmt_cost, fmt_hours

gpu = GPU_BY_NAME["H100 SXM5 80GB"]
result = evaluate(flops=3.15e23, gpu=gpu, mfu=0.35, num_gpus=1024)

print(fmt_hours(result.wall_hours))   # e.g. "2.1 hrs"
print(fmt_cost(result.cost_usd))      # e.g. "$247"
```

---

## GPU specs reference

All throughput figures use BF16 dense (no sparsity) -- the relevant format for LLM training.
Pricing is approximate spot/reserved market rate, USD/hr, mid-2025.

| GPU              | BF16 TFLOPS | VRAM  | ~Spot/hr |
|------------------|-------------|-------|----------|
| H100 SXM5 80GB   | 989         | 80 GB | $2.80    |
| H200 SXM 141GB   | 989         | 141GB | $4.20    |
| H100 PCIe 80GB   | 756         | 80 GB | $2.20    |
| A100 SXM4 80GB   | 312         | 80 GB | $1.90    |
| A100 PCIe 80GB   | 312         | 80 GB | $1.60    |
| L40S 48GB        | 733         | 48 GB | $1.40    |

---

## References

- [NVIDIA H100 Product Brief (2023)](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf)
- [NVIDIA A100 Datasheet (2020)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)
- [Korthikanti et al. (2022). *Reducing Activation Recomputation in Large Transformer Models.*](https://arxiv.org/abs/2205.05198) — MFU framing
- [Chowdhery et al. (2022). *PaLM: Scaling Language Modeling with Pathways.*](https://arxiv.org/abs/2204.02311) — real-world MFU benchmarks
