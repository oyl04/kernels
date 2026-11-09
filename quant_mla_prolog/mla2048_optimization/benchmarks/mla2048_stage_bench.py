from pathlib import Path
import runpy
import sys
import torch
import torch_npu

repo, diagnostic = sys.argv[1:3]
torch.npu.set_device(0)
free, total = torch.npu.mem_get_info()
print('FREE_BYTES', free, flush=True)
if free < 3.5 * 1024**3:
    raise SystemExit('ABORT: insufficient free memory')
torch.npu.set_per_process_memory_fraction(2 * 1024**3 / total, 0)
suite = runpy.run_path(str(Path(repo) / 'test/quant_mla_prolog/test_quant_mla_prolog.py'))
test = suite['test_network_mxfp8_precision']
golden = test.__globals__['golden']
golden._NETWORK_CASE_CONFIG['custom_t2048_c1'] = dict(tokens=2048, seed=20260908, blocks=72,
    indices=list(range(128,2176)), occupied_ranges=[[0,128]], scale_groups=24, qscale=0.5, notes='Stage diagnostic')
original = test.__globals__['_assert_c1_workspace_recovery']

def measure(captured, outputs, reference, cache, tokens):
    original(captured, outputs, reference, cache, tokens)
    print('FULL_BASELINE_CORRECTNESS_PASSED', flush=True)
    _, args = captured
    module = runpy.run_path(diagnostic)
    program = module['_compile_o11_native_variant'](0, 1)
    try:
        for _ in range(10):
            program(*args)
        torch.npu.synchronize()
        for _ in range(30):
            program(*args)
        torch.npu.synchronize()
        print('DIAGNOSTIC_ONLY_COMPLETED_40_TASKS; partial outputs not validated', flush=True)
    finally:
        program.close()

test.__globals__['_assert_c1_workspace_recovery'] = measure
test('custom_t2048_c1', 2048)
