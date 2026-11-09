"""Time the kernel captured by the sample's existing correctness test."""

import argparse
import json
from pathlib import Path
import runpy
from statistics import median

import torch
import torch_npu  # noqa: F401


def measure(captured, outputs, reference, physical_cache, tokens):
    check_recovery(captured, outputs, reference, physical_cache, tokens)
    program, args = captured
    for _ in range(20):
        program(*args)
    torch.npu.synchronize()
    samples = []
    for _ in range(7):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            program(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    print(json.dumps({"tokens": tokens, "median_us": median(samples), "samples_us": samples}), flush=True)


def profile(captured, outputs, reference, physical_cache, tokens):
    check_recovery(captured, outputs, reference, physical_cache, tokens)
    program, args = captured
    for _ in range(10):
        program(*args)
    torch.npu.synchronize()
    for _ in range(30):
        program(*args)
    torch.npu.synchronize()
    print(f"tokens={tokens}: launched 30 measured invocations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    token_counts = (8, 32, 128, 2048)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tokens", type=int, choices=token_counts)
    selection.add_argument("--profile-tokens", type=int, choices=token_counts)
    options = parser.parse_args()
    root = options.repo.resolve()
    suite = runpy.run_path(str(root / "test/quant_mla_prolog/test_quant_mla_prolog.py"))
    test = suite["test_network_mxfp8_precision"]
    check_recovery = test.__globals__["_assert_c1_workspace_recovery"]
    test.__globals__["_assert_c1_workspace_recovery"] = (
        profile if options.profile_tokens is not None else measure
    )
    network_cases = tuple(zip(suite["NETWORK_CASE_IDS"], suite["EXPECTED_TOKENS"]))
    selected_tokens = options.profile_tokens or options.tokens
    cases = (*network_cases, ("gen_t2048_c1", 2048)) if selected_tokens else network_cases
    for case, tokens in cases:
        if selected_tokens in (None, tokens):
            test(case, tokens)
