from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
a=s.index("        front = _O11MlaRef17FrontUnit(128)",s.index("class _MlaRuntimePrefillKernel:"))
b=s.index("        qa_token = block * 2 + sub",a)
s=s[:a]+"""        front = _O11MlaRef17FrontUnit(128)
        for n in range(block, 32, cores):
            if n < 24:
                for m_idx in range(capacity // 128):
                    front.project_qa_owned_tile(qa_raw, x_block, wqa, sx_block, swqa, n, m_idx)
            else:
                for m_idx in range(capacity // 128):
                    front.project_kva_tile(kva_raw, x_block, wkva, sx_block, swkva, n - 24, m_idx)
        _o11_front_sync_aic_to_aiv()
        channel_rewind(reset_sync_id=True)
        if block == 0:
            for m_idx in range(capacity // 128):
                front.project_kva_tile(kva_raw, x_block, wkva, sx_block, swkva, 8, m_idx)
"""+s[b:]
compile(s,str(p),"exec");p.write_text(s)
