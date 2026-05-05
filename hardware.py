"""
hardware.py -- GPU catalog, specs, and training cost calculations.

FLOPs units: raw floating-point operations (not TFLOPS).
All peak throughput figures use BF16 dense (no sparsity) -- the relevant format for LLM training.
Pricing is approximate spot/reserved market rate, USD/hr, as of mid-2025.
"""

import math
from dataclasses import dataclass


DEFAULT_MFU = 0.35  # 35% -- realistic average for production training setups


@dataclass
class GPU:
    name: str
    bf16_tflops: float        # peak BF16 dense TFLOPS (no sparsity)
    vram_gb: int
    spot_usd_hr: float        # approximate cloud spot price USD/hr
    generation: str           # "Hopper" | "Ampere" | "Ada"
    notes: str


GPU_CATALOG: list[GPU] = [
    GPU(
        name="H100 SXM5 80GB",
        bf16_tflops=989.0,
        vram_gb=80,
        spot_usd_hr=2.80,
        generation="Hopper",
        notes="Current standard for frontier model training. Highest NVLink bandwidth.",
    ),
    GPU(
        name="H200 SXM 141GB",
        bf16_tflops=989.0,
        vram_gb=141,
        spot_usd_hr=4.20,
        generation="Hopper",
        notes="Same compute as H100 SXM. HBM3e gives 1.4x memory bandwidth -- better for large models.",
    ),
    GPU(
        name="H100 PCIe 80GB",
        bf16_tflops=756.0,
        vram_gb=80,
        spot_usd_hr=2.20,
        generation="Hopper",
        notes="PCIe variant. Lower inter-GPU bandwidth than SXM. Suitable for smaller runs.",
    ),
    GPU(
        name="A100 SXM4 80GB",
        bf16_tflops=312.0,
        vram_gb=80,
        spot_usd_hr=1.90,
        generation="Ampere",
        notes="Previous generation flagship. Still the most widely deployed training GPU globally.",
    ),
    GPU(
        name="A100 PCIe 80GB",
        bf16_tflops=312.0,
        vram_gb=80,
        spot_usd_hr=1.60,
        generation="Ampere",
        notes="PCIe variant of A100. Common in cloud on-demand pools.",
    ),
    GPU(
        name="L40S 48GB",
        bf16_tflops=362.05,
        vram_gb=48,
        spot_usd_hr=1.40,
        generation="Ada",
        notes="Inference-optimized but capable for training smaller models. 48 GB VRAM limits scale.",
    ),
]

GPU_BY_NAME: dict[str, GPU] = {g.name: g for g in GPU_CATALOG}


# -- Reference training runs ---------------------------------------------------
# FLOPs computed via C = 6 x N x D. Estimates marked.

REFERENCE_RUNS: dict[str, dict] = {
    "GPT-2 (1.5B)":          {"flops": 1.97e20, "confirmed": True},
    "GPT-3 (175B)":          {"flops": 3.15e23, "confirmed": True},
    "LLaMA 3 8B":            {"flops": 1.81e23, "confirmed": True},
    "LLaMA 3 70B":           {"flops": 6.30e24, "confirmed": True},
    "LLaMA 3 405B":          {"flops": 3.65e25, "confirmed": True},
    "GPT-4 (est.)":          {"flops": 2.00e25, "confirmed": False},
    "Gemini Ultra (est.)":   {"flops": 5.00e24, "confirmed": False},
}


# -- Core calculations ---------------------------------------------------------

def gpu_hours(flops: float, gpu: GPU, mfu: float = DEFAULT_MFU) -> float:
    """
    GPU-hours = FLOPs / (peak_BF16_FLOPS_per_sec * MFU * 3600)

    bf16_tflops is in TFLOPS = 10^12 FLOPS/sec.
    MFU (Model FLOPs Utilization) is the fraction of peak throughput actually achieved.
    """
    if flops <= 0:
        raise ValueError("flops must be positive")
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in range (0, 1]")
    effective_flops_per_sec = gpu.bf16_tflops * 1e12 * mfu
    return flops / effective_flops_per_sec / 3600.0


def wall_clock_hours(total_gpu_hours: float, num_gpus: int) -> float:
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    return total_gpu_hours / num_gpus


def total_cost_usd(total_gpu_hours: float, gpu: GPU) -> float:
    return total_gpu_hours * gpu.spot_usd_hr


@dataclass
class RunResult:
    gpu: GPU
    flops: float
    mfu: float
    num_gpus: int
    total_gpu_hours: float
    wall_hours: float
    cost_usd: float

    @property
    def wall_days(self) -> float:
        return self.wall_hours / 24.0

    @property
    def wall_weeks(self) -> float:
        return self.wall_hours / 168.0


def evaluate(
    flops: float,
    gpu: GPU,
    mfu: float = DEFAULT_MFU,
    num_gpus: int = 1,
) -> RunResult:
    if flops <= 0:
        raise ValueError("flops must be positive")
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in range (0, 1]")
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    gh = gpu_hours(flops, gpu, mfu)
    wh = wall_clock_hours(gh, num_gpus)
    cost = total_cost_usd(gh, gpu)
    return RunResult(
        gpu=gpu,
        flops=flops,
        mfu=mfu,
        num_gpus=num_gpus,
        total_gpu_hours=gh,
        wall_hours=wh,
        cost_usd=cost,
    )


# -- Formatting helpers --------------------------------------------------------

def fmt_flops(f: float) -> str:
    if f <= 0:
        raise ValueError("flops must be positive")
    exp = int(math.floor(math.log10(f)))
    coeff = f / (10 ** exp)
    return f"{coeff:.2f} x 10^{exp}"


def fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h*60:.0f} min"
    if h < 24:
        return f"{h:.1f} hrs"
    if h < 24 * 14:
        return f"{h/24:.1f} days"
    return f"{h/168:.1f} weeks"


def fmt_cost(c: float) -> str:
    if c < 1_000:
        return f"${c:,.0f}"
    if c < 1_000_000:
        return f"${c/1_000:.1f}K"
    return f"${c/1_000_000:.2f}M"
