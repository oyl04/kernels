"""Generate experimental whole-head QB weight residency for 32-AIC prefill."""
import ast
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
bm = int(sys.argv[3])
assert bm in (64, 128)
start = source.index('class _O11QbHeadOwnedMixT128:')
end = source.index('class _O11MlaQHeadLocalEpilogUnit:', start)
body = source[start:end]
body = body[:body.index('        block = get_block_idx()')]
body = body.replace('_O11QbHeadOwnedMixT128', '_QbResidentPrefill')
body = body.replace('self.base_m = 128', f'self.base_m = {bm}')
body = body.replace('self.k_l1 = 512', 'self.k_l1 = Q_LORA')
# Only one complete head of B/scales needs to be resident.
body = body.replace('(kl1, bn),\n            dtypes.int8,\n            depth=2', '(kl1, bn),\n            dtypes.int8,\n            depth=1')
body = body.replace('(self.scale_l1_len, bn),\n            Float8E8M0,\n            depth=2', '(self.scale_l1_len, bn),\n            Float8E8M0,\n            depth=1')
body += '''        block = get_block_idx()
        capacity = qa_gm.shape[0]
        out_token_major = out_gm.view((capacity, N_HEADS * HEAD_DIM))
        for local_head in range_constexpr(self.max_heads):
            n_idx = block + local_head * self.core_num
            if n_idx < N_HEADS:
                sb = self.l1_scale_b.produce()
                mem_copy(sb, tile_slice(swqb_gm, (bn, self.scale_groups, 2), (n_idx, 0, 0)), engine=self.copy_scale_b)
                scale_b = self.l1_scale_b.consume()
                bw = self.l1_b.produce()
                mem_copy(bw, tile_slice(wqb, (Q_LORA, bn), (0, n_idx)))
                b = self.l1_b.consume().reinterpret(dtype=Float8E4M3FN)
                for m_idx in range(capacity // bm):
                    aw = self.l1_a.produce()
                    mem_copy(aw, tile_slice(qa_gm.view(dtype=dtypes.int8), (bm, Q_LORA), (m_idx, 0)), engine=self.copy_a)
                    saw = self.l1_scale_a.produce()
                    mem_copy(saw, tile_slice(sqa_gm, (bm, self.scale_groups, 2), (m_idx, 0, 0)), engine=self.copy_scale_a)
                    a = self.l1_a.consume().reinterpret(dtype=Float8E4M3FN).reinterpret(shape=(bm, Q_LORA))
                    scale_a = self.l1_scale_a.consume().reinterpret(shape=(bm, self.scale_l1_len))
                    cw = self.l0c.produce()
                    for k in range_constexpr(Q_LORA // bk):
                        a0 = self.l0a.produce()
                        mem_copy(a0, tile_slice(a, (bm, bk), (0, k)), mx_scale=tile_slice(scale_a, (bm, self.scale_l0_len), (0, k)))
                        b0 = self.l0b.produce()
                        mem_copy(b0, tile_slice(b, (bk, bn), (k, 0)), transpose=True, mx_scale=tile_slice(scale_b, (self.scale_l0_len, bn), (k, 0)))
                        final = k + 1 == Q_LORA // bk
                        matmul(cw.reinterpret(shape=(bm, bn)), self.l0a.consume().reinterpret(shape=(bm, bk)), self.l0b.consume().reinterpret(shape=(bk, bn)), init=(k == 0), unit_flag=3 if final else 2)
                    mem_copy(tile_slice(out_token_major, (bm, bn), (m_idx, n_idx)), self.l0c.consume(), engine=self.fixpipe, unit_flag=3)


'''
source = source[:end] + body + source[end:]
a = source.index('        for m_idx in range(capacity // 128):\n            qb = _O11QbHeadOwnedMixT128', source.index('class _MlaRuntimePrefillKernel:'))
b = source.index('        cube_sync_all()', a)
source = source[:a] + f'''        qb = _QbResidentPrefill({bm}, 32)
        qb.projection_kernel(q_head, qa_mx, wqb, qa_scale, swqb)
''' + source[b:]
ast.parse(source)
with Path(sys.argv[2]).open('x') as output:
    output.write(source)
