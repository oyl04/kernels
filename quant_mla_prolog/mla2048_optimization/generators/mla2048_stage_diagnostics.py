"""Create incomplete-output diagnostic kernels, not operator replacements."""
import ast
from pathlib import Path
import sys

baseline = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
out.mkdir(exist_ok=True)
start = baseline.index('class _MlaRuntimePrefillKernel:')
end = baseline.index('class _MlaRuntimePrefillLauncher:', start)
body = baseline[start:end]
cuts = {
    'pad': body.index('        front = _O11MlaRef17FrontUnit(128)'),
    'front': body.index('        if const_expr(self.quant_mode_c == 1):\n            kv ='),
    'qb': body.index('        if cores < 8:\n            vec_sync_block_wait'),
}
for stage in ['pad', 'front', 'qb', 'wkb']:
    partial = body[:cuts['qb' if stage == 'wkb' else stage]]
    if stage == 'wkb':
        partial += '''        wkb_unit = _O11MlaWkbTokenMajorLargeUnit(cdepth=2, bdepth=2)
        for head in range(block, N_HEADS, cores):
            wkb_unit.process_head(q_head, wkb, q_latent_h, head)
        cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
        cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
'''
    partial += '        global_sync_all()\n\n\n'
    name = '_MlaDiagnostic' + stage.capitalize() + 'Kernel'
    result = (baseline[:start] + partial + baseline[end:]).replace('_MlaRuntimePrefillKernel', name)
    result = result.replace('cannbotdsl_quant_mla_prolog', 'cannbotdsl_mla_diag_' + stage)
    ast.parse(result)
    with (out / (stage + '.py')).open('x') as stream:
        stream.write(result)
