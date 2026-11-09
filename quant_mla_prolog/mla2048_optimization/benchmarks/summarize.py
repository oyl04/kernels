"""Require a valid T=2048 run before summarizing steady-state device time."""
import csv
from pathlib import Path
import statistics
import sys

directory = Path(sys.argv[1])
variant = sys.argv[2]
log = (directory / 'run.log').read_text(errors='replace')
if 'T2048 golden and recovery passed; 30 measured invocations' not in log:
    raise SystemExit(f'{directory}: missing correctness marker')
symbol = '_MlaRuntimePrefillKernelT2048' if variant == 'gated' else '_MlaRuntimePrefillKernel___call__'
rows = []
for file in directory.rglob('op_summary*.csv'):
    with file.open() as stream:
        rows.extend(row for row in csv.DictReader(stream) if symbol in row.get('Op Name', ''))
if len(rows) != 47:
    raise SystemExit(f'{directory}: expected 47 records of {symbol}, found {len(rows)}')
rows.sort(key=lambda row: float(row['Task Start Time(us)']))
times = [float(row['Task Duration(us)']) for row in rows[-30:]]
print(f'{variant}: mean {statistics.mean(times):.3f} µs; '
      f'range {min(times):.3f}–{max(times):.3f} µs; n={len(times)}')
