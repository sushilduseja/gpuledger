import pytest
from hardware import (
    GPU_BY_NAME,
    GPU_CATALOG,
    evaluate,
    fmt_cost,
    fmt_flops,
    fmt_hours,
    gpu_hours,
    total_cost_usd,
    wall_clock_hours,
)


class TestGpuHours:
    def test_basic_calculation(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        gh = gpu_hours(flops=3.15e23, gpu=gpu, mfu=0.35)
        assert gh == pytest.approx(252780.586, rel=1e-3)

    def test_flops_must_be_positive(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        with pytest.raises(ValueError, match="flops must be positive"):
            gpu_hours(flops=0, gpu=gpu, mfu=0.35)
        with pytest.raises(ValueError, match="flops must be positive"):
            gpu_hours(flops=-1, gpu=gpu, mfu=0.35)

    def test_mfu_must_be_positive(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        with pytest.raises(ValueError, match="mfu must be in range"):
            gpu_hours(flops=1e20, gpu=gpu, mfu=0)
        with pytest.raises(ValueError, match="mfu must be in range"):
            gpu_hours(flops=1e20, gpu=gpu, mfu=-0.1)
        with pytest.raises(ValueError, match="mfu must be in range"):
            gpu_hours(flops=1e20, gpu=gpu, mfu=1.1)


class TestWallClockHours:
    def test_basic_calculation(self):
        wh = wall_clock_hours(total_gpu_hours=252780.586, num_gpus=1024)
        assert wh == pytest.approx(246.856, rel=1e-3)

    def test_num_gpus_must_be_at_least_1(self):
        with pytest.raises(ValueError, match="num_gpus must be at least 1"):
            wall_clock_hours(total_gpu_hours=100, num_gpus=0)


class TestTotalCostUsd:
    def test_basic_calculation(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        cost = total_cost_usd(total_gpu_hours=252780.586, gpu=gpu)
        assert cost == pytest.approx(707785.64, rel=1e-3)


class TestEvaluate:
    def test_gpt3_on_h100(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        result = evaluate(flops=3.15e23, gpu=gpu, mfu=0.35, num_gpus=1024)
        assert result.wall_hours == pytest.approx(246.856, rel=1e-3)
        assert result.cost_usd == pytest.approx(707785.64, rel=1e-3)

    def test_readme_example(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        result = evaluate(flops=3.15e23, gpu=gpu, mfu=0.35, num_gpus=1024)
        assert fmt_hours(result.wall_hours) == "10.3 days"
        assert fmt_cost(result.cost_usd) == "$707.8K"


class TestFormatting:
    def test_fmt_flops(self):
        assert fmt_flops(3.15e23) == "3.15 x 10^23"

    def test_fmt_flops_invalid(self):
        with pytest.raises(ValueError):
            fmt_flops(0)
        with pytest.raises(ValueError):
            fmt_flops(-1)

    def test_fmt_hours(self):
        assert fmt_hours(246.856) == "10.3 days"
        assert fmt_hours(2.0) == "2.0 hrs"
        assert fmt_hours(0.5) == "30 min"
        assert fmt_hours(400.0) == "2.4 weeks"

    def test_fmt_cost(self):
        assert fmt_cost(707785.64) == "$707.8K"
        assert fmt_cost(500) == "$500"
        assert fmt_cost(1500000) == "$1.50M"


class TestGpuCatalog:
    def test_all_gpus_have_unique_names(self):
        names = [gpu.name for gpu in GPU_CATALOG]
        assert len(names) == len(set(names))

    def test_all_gpus_have_valid_specs(self):
        for gpu in GPU_CATALOG:
            assert gpu.bf16_tflops > 0
            assert gpu.vram_gb > 0
            assert gpu.spot_usd_hr > 0

    def test_h100_sxm5_is_fastest(self):
        h100 = GPU_BY_NAME["H100 SXM5 80GB"]
        for gpu in GPU_CATALOG:
            if gpu.name != "H100 SXM5 80GB":
                assert h100.bf16_tflops >= gpu.bf16_tflops

    def test_l40s_uses_dense_bf16(self):
        l40s = GPU_BY_NAME["L40S 48GB"]
        assert l40s.bf16_tflops == pytest.approx(362.05, rel=1e-3)

    def test_readme_spec_table_matches_code(self):
        readme_specs = {
            "H100 SXM5 80GB": 989,
            "H200 SXM 141GB": 989,
            "H100 PCIe 80GB": 756,
            "A100 SXM4 80GB": 312,
            "A100 PCIe 80GB": 312,
            "L40S 48GB": 362,
        }
        for name, expected_tflops in readme_specs.items():
            gpu = GPU_BY_NAME[name]
            actual = round(gpu.bf16_tflops)
            assert actual == expected_tflops, f"{name} mismatch"


class TestReferenceRuns:
    def test_lLaMA3_405B_cost_estimate(self):
        gpu = GPU_BY_NAME["H100 SXM5 80GB"]
        result = evaluate(flops=3.65e25, gpu=gpu, mfu=0.35)
        assert result.cost_usd > 50_000_000  # At least $50M
        assert result.cost_usd < 200_000_000  # Under $200M