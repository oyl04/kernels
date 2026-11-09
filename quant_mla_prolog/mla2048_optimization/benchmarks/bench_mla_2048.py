from pathlib import Path
import runpy,sys
import torch
import torch_npu

root=Path(sys.argv[1])
suite=runpy.run_path(str(root/"test/quant_mla_prolog/test_quant_mla_prolog.py"))
test=suite["test_network_mxfp8_precision"]
golden=test.__globals__["golden"]
case_id="custom_t2048_c1"
golden._NETWORK_CASE_CONFIG[case_id]=dict(tokens=2048, seed=20260908, blocks=72,
    indices=list(range(128,2176)), occupied_ranges=[[0,128]], scale_groups=24,
    qscale=0.5, notes="T2048 prefill correctness and device profiling")
check=test.__globals__["_assert_c1_workspace_recovery"]
def profile(captured,outputs,reference,cache,tokens):
    check(captured,outputs,reference,cache,tokens)
    program,args=captured
    for _ in range(10): program(*args)
    torch.npu.synchronize()
    for _ in range(30): program(*args)
    torch.npu.synchronize()
    print("T2048 golden and recovery passed; 30 measured invocations",flush=True)
test.__globals__["_assert_c1_workspace_recovery"]=profile
test(case_id,2048)
