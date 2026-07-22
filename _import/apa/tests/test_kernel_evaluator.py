import sys
import os
sys.path.insert(0, "/home/shadeform/flash-attention-5")
sys.path.insert(0, "/home/shadeform/openevolve")
os.chdir("/home/shadeform/flash-attention-5")


def test_kernel_evaluator_on_initial_kernel():
    """Evaluator should return valid scores for the baseline kernel."""
    from phase2_kernel.evaluator import evaluate
    result = evaluate("phase2_kernel/initial_kernel.py")
    m = result.metrics

    assert "combined_score" in m
    assert "accuracy_score" in m
    assert "throughput_score" in m
    assert "reliability_score" in m

    assert m["accuracy_score"] > 0.8, f"Accuracy too low: {m['accuracy_score']}"
    assert m["throughput_score"] < 0.1, f"Python loops shouldn't be fast: {m['throughput_score']}"
    assert m["reliability_score"] == 1.0, f"Initial kernel should never produce NaN"
    assert 0.0 < m["combined_score"] < 1.0


def test_kernel_evaluator_on_broken_kernel():
    """Evaluator should handle a kernel that crashes gracefully."""
    import tempfile
    code = '''
def flash_attention_5(Q, K, V, causal=False):
    raise RuntimeError("broken kernel")

def get_kernel():
    return flash_attention_5
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(code)
        path = f.name

    from phase2_kernel.evaluator import evaluate
    result = evaluate(path)
    assert result.metrics["combined_score"] == 0.0
    os.unlink(path)
