import runpy
from pathlib import Path
import torch
import torch_npu

torch.npu.set_device(0)
free, total = torch.npu.mem_get_info()
print(f'DEVICE_MEMORY free={free} total={total}', flush=True)
if free < 3.5 * 1024**3:
    raise SystemExit('ABORT: less than 3.5 GiB free')
# Limit this process's PyTorch caching allocator to 2 GiB, leaving a margin.
# Driver/runtime allocations are outside this allocator limit.
torch.npu.set_per_process_memory_fraction(2 * 1024**3 / total, 0)
runpy.run_path(str(Path(__file__).with_name('bench_mla_2048.py')), run_name='__main__')
print('PEAK_TORCH_BYTES', torch.npu.max_memory_allocated(), flush=True)
