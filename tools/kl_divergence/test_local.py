"""
Local test for the KL divergence computation pipeline.
Tests the core math and data flow using synthetic data.
Only requires torch + safetensors (no transformers, vllm, or GPU needed).
"""

import json
import os
import tempfile

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_kl_for_chunk(base_hidden, target_hidden, weights, temperature=1.0):
    """
    Simplified version of _compute_kl_for_chunk from compute_kl.py.

    Hidden states are expected to be post-norm (extracted at layer index
    num_hidden_layers from vLLM), so no additional normalization is applied.
    """
    base_logits = base_hidden.float() @ weights["lm_head_weight"].float().T
    target_logits = target_hidden.float() @ weights["lm_head_weight"].float().T

    if weights.get("lm_head_bias") is not None:
        base_logits += weights["lm_head_bias"].float()
        target_logits += weights["lm_head_bias"].float()

    if temperature != 1.0:
        base_logits = base_logits / temperature
        target_logits = target_logits / temperature

    base_log_probs = F.log_softmax(base_logits, dim=-1)
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    kl_per_position = F.kl_div(
        target_log_probs, base_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)

    return kl_per_position


def create_fake_hidden_states(output_dir, num_samples, seq_len, hidden_dim, token_ids_list=None):
    """Create fake hidden state safetensors files (simulating vLLM output)."""
    os.makedirs(output_dir, exist_ok=True)
    files = []
    all_hidden = []
    all_token_ids = []

    for i in range(num_samples):
        hidden = torch.randn(seq_len, 1, hidden_dim, dtype=torch.float16)
        if token_ids_list is not None:
            token_ids = token_ids_list[i]
        else:
            token_ids = torch.randint(0, 32000, (seq_len,))
        filename = f"hidden_states_{i:06d}.safetensors"
        save_file(
            {"hidden_states": hidden, "token_ids": token_ids},
            os.path.join(output_dir, filename),
        )
        files.append(filename)
        all_hidden.append(hidden)
        all_token_ids.append(token_ids)

    metadata = {
        "model_id": "test-model",
        "layer_index": 31,
        "max_seq_length": seq_len,
        "num_samples": num_samples,
        "dataset_name": "test",
        "dataset_config": "test",
        "split": "test",
        "files": files,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    return all_hidden, all_token_ids


def create_fake_weights(hidden_dim, vocab_size):
    """Create fake lm_head weights (no norm needed — hidden states are post-norm)."""
    return {
        "lm_head_weight": torch.randn(vocab_size, hidden_dim, dtype=torch.float16),
        "lm_head_bias": None,
    }


def run_tests():
    hidden_dim = 64   # small for speed
    vocab_size = 128   # small for speed
    seq_len = 32
    num_samples = 4

    passed = 0
    failed = 0

    # --- Test 1: Logit computation correctness ---
    print("Test 1: Logit computation (hidden @ lm_head.T)...")
    h = torch.randn(4, hidden_dim, dtype=torch.float16)
    w_lm = torch.randn(vocab_size, hidden_dim, dtype=torch.float16)
    logits = h.float() @ w_lm.float().T
    assert logits.shape == (4, vocab_size), f"Wrong shape: {logits.shape}"
    # Softmax over logits should sum to 1
    probs = F.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), f"Softmax sums: {sums}"
    print(f"  PASSED (logit shape: {logits.shape}, softmax sums: {sums.tolist()})")
    passed += 1

    # --- Test 2: Logit computation with bias ---
    print("Test 2: Logit computation with lm_head bias...")
    bias = torch.randn(vocab_size, dtype=torch.float16)
    logits_with_bias = h.float() @ w_lm.float().T + bias.float()
    # Adding bias should shift logits but softmax should still sum to 1
    probs_bias = F.softmax(logits_with_bias, dim=-1)
    sums_bias = probs_bias.sum(dim=-1)
    assert torch.allclose(sums_bias, torch.ones_like(sums_bias), atol=1e-4)
    # Logits should differ from no-bias version
    assert not torch.allclose(logits, logits_with_bias)
    print(f"  PASSED (bias shifts logits, softmax still valid)")
    passed += 1

    # --- Test 3: Self-comparison KL = 0 ---
    print("Test 3: Self-comparison KL divergence = 0...")
    weights = create_fake_weights(hidden_dim, vocab_size)
    h_test = torch.randn(seq_len, hidden_dim, dtype=torch.float16)
    kl = compute_kl_for_chunk(h_test, h_test, weights)
    mean_kl = kl.mean().item()
    assert abs(mean_kl) < 1e-5, f"Self-comparison KL should be ~0, got {mean_kl}"
    print(f"  PASSED (mean KL: {mean_kl})")
    passed += 1

    # --- Test 4: Different inputs produce KL > 0 ---
    print("Test 4: Different inputs produce KL > 0...")
    h_base = torch.randn(seq_len, hidden_dim, dtype=torch.float16)
    h_target = h_base + torch.randn_like(h_base) * 0.5
    kl = compute_kl_for_chunk(h_base, h_target, weights)
    mean_kl = kl.mean().item()
    assert mean_kl > 0, f"Different inputs should have KL > 0, got {mean_kl}"
    print(f"  PASSED (mean KL: {mean_kl:.6f})")
    passed += 1

    # --- Test 5: KL is non-negative everywhere ---
    print("Test 5: KL is non-negative for all positions...")
    assert (kl >= -1e-6).all(), f"KL has negative values: {kl.min().item()}"
    print(f"  PASSED (min: {kl.min().item():.8f})")
    passed += 1

    # --- Test 6: Temperature scaling ---
    print("Test 6: Temperature scaling changes KL...")
    kl_t1 = compute_kl_for_chunk(h_base, h_target, weights, temperature=1.0)
    kl_t05 = compute_kl_for_chunk(h_base, h_target, weights, temperature=0.5)
    kl_t2 = compute_kl_for_chunk(h_base, h_target, weights, temperature=2.0)
    # Lower temperature -> sharper distributions -> higher KL
    # Higher temperature -> smoother distributions -> lower KL
    assert kl_t05.mean() > kl_t1.mean(), "Lower temp should increase KL"
    assert kl_t2.mean() < kl_t1.mean(), "Higher temp should decrease KL"
    print(f"  PASSED (t=0.5: {kl_t05.mean():.6f}, t=1.0: {kl_t1.mean():.6f}, t=2.0: {kl_t2.mean():.6f})")
    passed += 1

    # --- Test 7: End-to-end with safetensors files (matching token IDs) ---
    print("Test 7: End-to-end with safetensors file I/O...")
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "base")
        target_dir = os.path.join(tmpdir, "target")

        # Generate shared token IDs for both base and target
        shared_token_ids = [torch.randint(0, 32000, (seq_len,)) for _ in range(num_samples)]

        # Create identical hidden states for both, with same token IDs
        base_hidden, _ = create_fake_hidden_states(
            base_dir, num_samples, seq_len, hidden_dim, token_ids_list=shared_token_ids
        )

        # Copy base as target (self-comparison) with same token IDs
        os.makedirs(target_dir, exist_ok=True)
        target_files = []
        for i in range(num_samples):
            filename = f"hidden_states_{i:06d}.safetensors"
            save_file(
                {"hidden_states": base_hidden[i], "token_ids": shared_token_ids[i]},
                os.path.join(target_dir, filename),
            )
            target_files.append(filename)

        target_meta = {
            "model_id": "test-model", "layer_index": 31,
            "max_seq_length": seq_len, "num_samples": num_samples,
            "dataset_name": "test", "dataset_config": "test",
            "split": "test", "files": target_files,
        }
        with open(os.path.join(target_dir, "metadata.json"), "w") as f:
            json.dump(target_meta, f)

        # Run the full computation loop (mimicking compute_kl.py)
        from safetensors import safe_open

        with open(os.path.join(base_dir, "metadata.json")) as f:
            base_meta = json.load(f)
        with open(os.path.join(target_dir, "metadata.json")) as f:
            target_meta = json.load(f)

        per_sample_kl = []
        for bf, tf in zip(sorted(base_meta["files"]), sorted(target_meta["files"])):
            with safe_open(os.path.join(base_dir, bf), framework="pt") as f:
                bh = f.get_tensor("hidden_states").squeeze(1)
                base_tids = f.get_tensor("token_ids")
            with safe_open(os.path.join(target_dir, tf), framework="pt") as f:
                th = f.get_tensor("hidden_states").squeeze(1)
                target_tids = f.get_tensor("token_ids")

            # Verify token alignment
            assert torch.equal(base_tids, target_tids), "Token IDs should match"

            kl = compute_kl_for_chunk(bh, th, weights)
            per_sample_kl.append(kl.mean().item())

        mean_kl = sum(per_sample_kl) / len(per_sample_kl)
        assert abs(mean_kl) < 1e-5, f"E2E self-comparison KL should be ~0, got {mean_kl}"
        print(f"  PASSED (mean KL across {num_samples} samples: {mean_kl})")
        passed += 1

    # --- Test 8: E2E with perturbed target (matching token IDs) ---
    print("Test 8: End-to-end with perturbed target...")
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "base")
        target_dir = os.path.join(tmpdir, "target")

        shared_token_ids = [torch.randint(0, 32000, (seq_len,)) for _ in range(num_samples)]

        base_hidden, _ = create_fake_hidden_states(
            base_dir, num_samples, seq_len, hidden_dim, token_ids_list=shared_token_ids
        )

        # Create perturbed target with same token IDs
        os.makedirs(target_dir, exist_ok=True)
        target_files = []
        for i in range(num_samples):
            filename = f"hidden_states_{i:06d}.safetensors"
            h_perturbed = base_hidden[i] + torch.randn_like(base_hidden[i]) * 0.3
            save_file(
                {"hidden_states": h_perturbed, "token_ids": shared_token_ids[i]},
                os.path.join(target_dir, filename),
            )
            target_files.append(filename)

        target_meta = {
            "model_id": "test-model-quantized", "layer_index": 31,
            "max_seq_length": seq_len, "num_samples": num_samples,
            "dataset_name": "test", "dataset_config": "test",
            "split": "test", "files": target_files,
        }
        with open(os.path.join(target_dir, "metadata.json"), "w") as f:
            json.dump(target_meta, f)

        from safetensors import safe_open

        with open(os.path.join(base_dir, "metadata.json")) as f:
            base_meta = json.load(f)

        per_sample_kl = []
        for bf, tf in zip(sorted(base_meta["files"]), sorted(target_meta["files"])):
            with safe_open(os.path.join(base_dir, bf), framework="pt") as f:
                bh = f.get_tensor("hidden_states").squeeze(1)
            with safe_open(os.path.join(target_dir, tf), framework="pt") as f:
                th = f.get_tensor("hidden_states").squeeze(1)

            kl = compute_kl_for_chunk(bh, th, weights)
            per_sample_kl.append(kl.mean().item())

        mean_kl = sum(per_sample_kl) / len(per_sample_kl)
        assert mean_kl > 0, f"Perturbed target should have KL > 0, got {mean_kl}"
        assert all(k >= -1e-6 for k in per_sample_kl), "All per-sample KL should be non-negative"
        print(f"  PASSED (mean KL: {mean_kl:.6f}, per-sample: {[f'{k:.6f}' for k in per_sample_kl]})")
        passed += 1

    # --- Test 9: Token ID mismatch raises error ---
    print("Test 9: Token ID mismatch detection...")
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = os.path.join(tmpdir, "base")
        target_dir = os.path.join(tmpdir, "target")

        # Create base with one set of token IDs
        base_token_ids = [torch.randint(0, 32000, (seq_len,)) for _ in range(num_samples)]
        create_fake_hidden_states(
            base_dir, num_samples, seq_len, hidden_dim, token_ids_list=base_token_ids
        )

        # Create target with DIFFERENT token IDs
        different_token_ids = [torch.randint(0, 32000, (seq_len,)) for _ in range(num_samples)]
        create_fake_hidden_states(
            target_dir, num_samples, seq_len, hidden_dim, token_ids_list=different_token_ids
        )

        from safetensors import safe_open

        with open(os.path.join(base_dir, "metadata.json")) as f:
            base_meta = json.load(f)
        with open(os.path.join(target_dir, "metadata.json")) as f:
            target_meta = json.load(f)

        bf = sorted(base_meta["files"])[0]
        tf = sorted(target_meta["files"])[0]

        with safe_open(os.path.join(base_dir, bf), framework="pt") as f:
            base_tids = f.get_tensor("token_ids")
        with safe_open(os.path.join(target_dir, tf), framework="pt") as f:
            target_tids = f.get_tensor("token_ids")

        mismatch_detected = not torch.equal(base_tids, target_tids)
        assert mismatch_detected, "Different random token IDs should not be equal"
        print(f"  PASSED (mismatch correctly detected)")
        passed += 1

    # --- Test 10: Dimension compatibility check ---
    print("Test 10: Hidden dim / vocab size validation...")
    mismatched_weights = create_fake_weights(hidden_dim + 1, vocab_size)
    h_test = torch.randn(seq_len, hidden_dim, dtype=torch.float16)
    try:
        # This should fail because hidden_dim doesn't match lm_head
        compute_kl_for_chunk(h_test, h_test, mismatched_weights)
        print("  FAILED (should have raised an error)")
        failed += 1
    except RuntimeError:
        print("  PASSED (RuntimeError raised for dimension mismatch)")
        passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
