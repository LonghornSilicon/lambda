import torch
import sys
sys.path.insert(0, "/home/shadeform/flash-attention-5")
from common.workloads import generate_workload, ALL_WORKLOADS


def test_all_workloads_count():
    assert len(ALL_WORKLOADS) == 12


def test_generate_workload_shapes():
    cfg = ALL_WORKLOADS[0]
    Q, K, V = generate_workload(cfg, seed=42)
    assert Q.shape == (cfg["batch"], cfg["num_heads"], cfg["seq_len"], cfg["head_dim"])
    assert Q.dtype == torch.float32


def test_outlier_workload_has_large_values():
    outlier_cfgs = [c for c in ALL_WORKLOADS if c["outliers"]]
    assert len(outlier_cfgs) > 0
    Q, K, V = generate_workload(outlier_cfgs[0], seed=42)
    assert Q.abs().max() > 10.0


def test_no_outlier_workload_is_standard_normal():
    clean_cfgs = [c for c in ALL_WORKLOADS if not c["outliers"]]
    Q, K, V = generate_workload(clean_cfgs[0], seed=42)
    assert Q.abs().max() < 10.0


def test_deterministic_with_seed():
    cfg = ALL_WORKLOADS[0]
    Q1, K1, V1 = generate_workload(cfg, seed=123)
    Q2, K2, V2 = generate_workload(cfg, seed=123)
    assert torch.equal(Q1, Q2)
