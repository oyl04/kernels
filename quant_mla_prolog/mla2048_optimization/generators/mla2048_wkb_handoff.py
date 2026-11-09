"""Generate the experimental two-chunk handoff for T=2048 FP8, 32 AICs."""
import ast
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
start = source.index('class _O11MlaWkbTokenMajorLargeUnit:')
end = source.index('class _MlaPrefillQbUnit:', start)
unit = source[start:end].rstrip()
unit = unit.replace('def __init__(self, cdepth=1, bdepth=1):',
                    'def __init__(self, cdepth=1, bdepth=1, early=False):\n        self.early = early')
unit = unit.replace('q_latent, head):', 'q_latent, head, early_flag):')
unit += '''
            if const_expr(self.early):
                if get_block_num() == 32:
                    if m_tile + 1 == m_tiles // 2:
                        cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                        cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                        cube_sync_block_arrive(_sync_ops.PIPE.MTE2, early_flag, mode=2)


'''
source = source[:start] + unit + source[end:]
a = source.index('class _MlaRuntimePrefillKernel:')
b = source.index('class _MlaRuntimePrefillLauncher:', a)
body = source[a:b]
old = '''                if cores >= 8:
                    vec_sync_block_wait(_sync_ops.PIPE.S, 3 + head // cores, mode=2)
                for token_batch in range(sub, capacity // 16, 2):
                    epilog.process_batch(
                        nope, rope, out, descale_q, head * (capacity // 16) + token_batch
                    )'''
new = '''                if cores == 32:
                    vec_sync_block_wait(_sync_ops.PIPE.S, 6 + head // cores, mode=2)
                    for token_batch in range(sub, capacity // 32, 2):
                        epilog.process_batch(nope, rope, out, descale_q, head * (capacity // 16) + token_batch)
                    vec_sync_block_wait(_sync_ops.PIPE.S, 3 + head // cores, mode=2)
                    for token_batch in range(capacity // 32 + sub, capacity // 16, 2):
                        epilog.process_batch(nope, rope, out, descale_q, head * (capacity // 16) + token_batch)
                else:
''' + '\n'.join('    ' + line for line in old.splitlines())
assert body.count(old) == 1
body = body.replace(old, new)
old = '            wkb_unit.process_head(q_head, wkb, q_latent_h, head)'
assert body.count(old) == 1
body = body.replace(old, '            wkb_unit.process_head(q_head, wkb, q_latent_h, head, 6 + head // cores)')
body = body.replace('_O11MlaWkbTokenMajorLargeUnit(cdepth=2, bdepth=2)',
                    '_O11MlaWkbTokenMajorLargeUnit(cdepth=2, bdepth=2, early=self.quant_mode_c == 1)')
result = source[:a] + body + source[b:]
ast.parse(result)
with Path(sys.argv[2]).open('x') as out:
    out.write(result)
