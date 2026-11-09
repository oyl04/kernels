"""Generate larger Q epilogue batches for the experimental prefill path."""
import ast
from pathlib import Path
import re
import sys

s = Path(sys.argv[1]).read_text()
rows = int(sys.argv[3])
assert rows in (16, 32)
split = '--split' in sys.argv[4:]
assert not split or rows == 32
a = s.index('class _MlaQHeadTokenEpilogMxfp8Unit:')
b = s.index('\nclass ', a + 10)
epilog = re.sub(r'(?<![eE]-)\b8\b', str(rows), s[a:b])
assert 'vdups(1.0e-8, dtypes.float32, mask=full)' in epilog
if split:
    x = epilog.index('        with vf(mode="simd"):')
    y = epilog.index('        output_rows =', x)
    vec = epilog[x:y].replace('range_constexpr(32)', 'range_constexpr(16)')
    for name in ('q_nope_rows', 'q_rope_rows', 'encoded', 'scales'):
        vec = re.sub(r'\b' + name + r'\b', 'part_' + name, vec)
    epilog = epilog[:x] + '''        for half in range(2):
            part_q_nope_rows = tile_slice(q_nope_rows, (16, KV_LORA), (half, 0))
            part_q_rope_rows = tile_slice(q_rope_rows, (16, ROPE), (half, 0))
            part_encoded = tile_slice(encoded, (16, FEATURES), (half, 0))
            part_scales = tile_slice(scales, (16, 1), (half, 0))
''' + ''.join('    ' + line for line in vec.splitlines(keepends=True)) + epilog[y:]
s = s[:a] + epilog + s[b:]
a = s.index('class _MlaRuntimePrefillKernel:')
b = s.index('class _MlaRuntimePrefillLauncher:', a)
body = s[a:b]
a2 = body.index('            epilog = _MlaQHeadTokenEpilogMxfp8Unit(')
b2 = body.index('        else:\n            epilog_bf16', a2)
body = body[:a2] + re.sub(r'\b8\b', str(rows), body[a2:b2]) + body[b2:]
# The head-readiness cores>=8 condition is topology, not batch size.
body = body.replace(f'if cores >= {rows}:', 'if cores >= 8:')
s = s[:a] + body + s[b:]
ast.parse(s)
with Path(sys.argv[2]).open('x') as output:
    output.write(s)
