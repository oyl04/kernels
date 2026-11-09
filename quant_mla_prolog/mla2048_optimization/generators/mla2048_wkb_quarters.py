"""Extend the tested halfway handoff to 512-token chunks; T=2048/32 AIC only."""
import ast
from pathlib import Path
import sys

s = Path(sys.argv[1]).read_text()
old = '''                    if m_tile + 1 == m_tiles // 2:
                        cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                        cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                        cube_sync_block_arrive(_sync_ops.PIPE.MTE2, early_flag, mode=2)'''
new = '''                    if m_tile + 1 < m_tiles:
                        if (m_tile + 1) % (m_tiles // 4) == 0:
                            cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                            cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                            chunk = (m_tile + 1) // (m_tiles // 4) - 1
                            cube_sync_block_arrive(_sync_ops.PIPE.MTE2, early_flag + 3 * chunk, mode=2)'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''                    vec_sync_block_wait(_sync_ops.PIPE.S, 6 + head // cores, mode=2)
                    for token_batch in range(sub, capacity // 32, 2):
                        epilog.process_batch(nope, rope, out, descale_q, head * (capacity // 16) + token_batch)
                    vec_sync_block_wait(_sync_ops.PIPE.S, 3 + head // cores, mode=2)
                    for token_batch in range(capacity // 32 + sub, capacity // 16, 2):
                        epilog.process_batch(nope, rope, out, descale_q, head * (capacity // 16) + token_batch)'''
new = '''                    chunk_batches = capacity // 64
                    for chunk in range(4):
                        if chunk < 3:
                            vec_sync_block_wait(_sync_ops.PIPE.S, 6 + head // cores + 3 * chunk, mode=2)
                        else:
                            vec_sync_block_wait(_sync_ops.PIPE.S, 3 + head // cores, mode=2)
                        for token_batch in range(chunk * chunk_batches + sub, (chunk + 1) * chunk_batches, 2):
                            epilog.process_batch(nope, rope, out, descale_q, head * (capacity // 16) + token_batch)'''
assert s.count(old) == 1
s = s.replace(old, new)
ast.parse(s)
with Path(sys.argv[2]).open('x') as out:
    out.write(s)
