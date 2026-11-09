from pathlib import Path
import sys
p=Path(sys.argv[1]);v=sys.argv[2];s=p.read_text()
def edit(s,a,b):
 assert s.count(a)==1,a
 return s.replace(a,b)
if v in ("m64","all","qb-m64","all-static","all-static-fit"):
 a=s.index("class _O11MlaWkbTokenMajorLargeUnit:")
 b=s.index("class _MlaPrefillQbUnit:",a)
 s=s[:a]+edit(s[a:b],"self.bm = 32","self.bm = 64")+s[b:]
if v in ("qb-owned","all","qb-m64","all-static","all-static-fit"):
 a=s.index("class _MlaRuntimePrefillKernel:")
 b=s.index("class _MlaRuntimePrefillLauncher:",a)
 body=s[a:b]
 body=edit(body,"qb = _MlaPrefillQbUnit(128)","qb = _O11QbHeadOwnedMixT128(128, 32)")
 body=edit(body,"""        cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, 1, mode=0)
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, 1, mode=0)
""","")
 s=s[:a]+body+s[b:]
if v in ("front-pair","all","all-static","all-static-fit"):
 donor=Path(sys.argv[3]).read_text()
 a=donor.index("    @jit\n    def project_kva_pair(")
 b=donor.index("class _O11QbHeadMixNoAlias:",a)
 method=donor[a:b]
 i=s.index("class _O11QbHeadMixNoAlias:")
 s=s[:i]+method+s[i:]
 a=s.index("        front = _O11MlaRef17FrontUnit(128)",s.index("class _MlaRuntimePrefillKernel:"))
 b=s.index("        qa_token = block * 2 + sub",a)
 s=s[:a]+"""        front = _O11MlaRef17FrontUnit(128)
        if block < 24:
            for m_idx in range(capacity // 128):
                front.project_qa_owned_tile(qa_raw, x_block, wqa, sx_block, swqa, block, m_idx)
        elif block < 31:
            for m_idx in range(capacity // 128):
                front.project_kva_tile(kva_raw, x_block, wkva, sx_block, swkva, block - 24, m_idx)
        else:
            for m_idx in range(capacity // 128):
                front.project_kva_pair(kva_raw, x_block, wkva, sx_block, swkva, m_idx)
        _o11_front_sync_aic_to_aiv()
        channel_rewind(reset_sync_id=True)
""" + s[b:]
if v in ("all-static", "all-static-fit"):
 a=s.index("        front = _O11MlaRef17FrontUnit(128)",s.index("class _MlaRuntimePrefillKernel:"))
 b=s.index("        qa_token = block * 2 + sub",a)
 s=s[:a]+s[a:b].replace("range(capacity // 128)", "range_constexpr(16)")+s[b:]
if v == "all-static-fit":
 a=s.index("class _O11MlaRef17FrontUnit:")
 b=s.index("class _O11QbHeadMixNoAlias:",a)
 body=s[a:b]
 pos=body.index("        self.l1_b = Channel(")
 body=body[:pos].replace("depth=2","depth=1")+body[pos:]
 s=s[:a]+body+s[b:]
a=s.index("class _O11MlaRef17FrontUnit:")
b=s.index("class _O11QbHeadMixNoAlias:",a)
body=s[a:b].replace("(self.m, bn), dtypes.float32, depth=2", "(self.m, bn), dtypes.float32, depth=1")
s=s[:a]+body+s[b:]
compile(s,str(p),"exec");p.write_text(s)
