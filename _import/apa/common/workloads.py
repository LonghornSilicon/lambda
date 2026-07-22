import torch

ALL_WORKLOADS = [
    {"seq_len": 512,  "head_dim": 64,  "causal": False, "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 512,  "head_dim": 128, "causal": True,  "outliers": True,  "batch": 2, "num_heads": 8},
    {"seq_len": 2048, "head_dim": 64,  "causal": False, "outliers": True,  "batch": 2, "num_heads": 8},
    {"seq_len": 2048, "head_dim": 128, "causal": True,  "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 8192, "head_dim": 64,  "causal": True,  "outliers": False, "batch": 1, "num_heads": 4},
    {"seq_len": 8192, "head_dim": 128, "causal": False, "outliers": True,  "batch": 1, "num_heads": 4},
    {"seq_len": 2048, "head_dim": 64,  "causal": True,  "outliers": True,  "batch": 2, "num_heads": 8},
    {"seq_len": 2048, "head_dim": 128, "causal": False, "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 4096, "head_dim": 64,  "causal": False, "outliers": False, "batch": 2, "num_heads": 8},
    {"seq_len": 4096, "head_dim": 128, "causal": True,  "outliers": True,  "batch": 1, "num_heads": 8},
    {"seq_len": 8192, "head_dim": 64,  "causal": False, "outliers": False, "batch": 1, "num_heads": 4},
    {"seq_len": 8192, "head_dim": 128, "causal": True,  "outliers": False, "batch": 1, "num_heads": 4},
]


def generate_workload(config, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)

    B = config["batch"]
    H = config["num_heads"]
    N = config["seq_len"]
    d = config["head_dim"]
    shape = (B, H, N, d)

    Q = torch.randn(shape, generator=gen, dtype=torch.float32)
    K = torch.randn(shape, generator=gen, dtype=torch.float32)
    V = torch.randn(shape, generator=gen, dtype=torch.float32)

    if config["outliers"]:
        for tensor in [Q, K, V]:
            mask = torch.bernoulli(torch.full(shape, 0.001), generator=gen).bool()
            outlier_vals = torch.randn(shape, generator=gen, dtype=torch.float32) * 100.0
            tensor[mask] = outlier_vals[mask]

    return Q, K, V
