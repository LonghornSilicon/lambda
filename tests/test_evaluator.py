import sys
import os
sys.path.insert(0, "/home/shadeform/flash-attention-5")
sys.path.insert(0, "/home/shadeform/openevolve")
os.chdir("/home/shadeform/flash-attention-5")


def test_evaluator_on_initial_policy():
    """Evaluator should return valid scores for the baseline FP16 policy."""
    from phase1_policy.evaluator import evaluate
    result = evaluate("phase1_policy/initial_policy.py")
    m = result.metrics

    assert "combined_score" in m
    assert "accuracy_score" in m
    assert "compression_score" in m
    assert "reliability_score" in m

    # FP16 baseline: high accuracy, zero compression
    assert m["accuracy_score"] > 0.5, f"Accuracy too low: {m['accuracy_score']}"
    assert m["compression_score"] < 0.05, f"FP16 should have near-zero compression: {m['compression_score']}"
    assert m["reliability_score"] == 1.0, f"FP16 should never produce NaN"
    assert 0.0 < m["combined_score"] < 1.0


def test_evaluator_on_broken_policy():
    """Evaluator should handle a policy that returns invalid precision gracefully."""
    import tempfile
    code = '''
def precision_policy(block_stats):
    return "bfloat3"

def get_policy():
    return precision_policy
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(code)
        path = f.name

    from phase1_policy.evaluator import evaluate
    result = evaluate(path)
    assert result.metrics["reliability_score"] == 0.0
    os.unlink(path)
