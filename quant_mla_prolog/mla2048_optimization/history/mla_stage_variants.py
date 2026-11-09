from pathlib import Path
import sys
p=Path(sys.argv[1]); variant=sys.argv[2]; s=p.read_text()
def edit(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
if variant.startswith("n256"):
    a=s.index("class _O11MlaWkbTokenMajorUnit:")
    b=s.index("class _MlaInputTailUnit:",a)
    body=s[a:b].replace("(QK_NOPE, 128)","(QK_NOPE, 256)").replace("(self.bm, 128)","(self.bm, 256)").replace("range_constexpr(4)","range_constexpr(2)")
    body=body.replace("depth=bdepth", "depth=1")
    if variant == "n256-c1":
        body=body.replace("depth=self.cdepth","depth=1")
    s=s[:a]+body+s[b:]
elif variant == "head-pipeline":
    a=s.index("class _O11QbHeadOwnedMixT128:")
    b=s.index("class _O11MlaQHeadLocalEpilogUnit:",a)
    body=s[a:b]
    body=edit(body,"        swqb_gm: Tensor,\n", "        swqb_gm: Tensor,\n        wkb_unit, wkb, q_latent_h, head_ready,\n")
    # Bound the simultaneous QB/WKB L1 and L0 footprints.
    body=body.replace("depth=2", "depth=1")
    body=body.rstrip()+"""
                cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                wkb_unit.process_head(out_gm, wkb, q_latent_h, n_idx)
                cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_block_arrive(_sync_ops.PIPE.MTE2, head_ready + local_head, mode=2)


"""
    s=s[:a]+body+s[b:]
    a=s.index("        wkb_unit = _O11MlaWkbTokenMajorUnit(",s.index("class _MlaRuntimeTileUnit:"))
    b=s.index("        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.tail_done",a)
    s=s[:a]+s[b:]
    s=edit(s,"        qb.projection_kernel(q_head, qa_mx, wqb, qa_scale, swqb)",
           """        wkb_unit = _O11MlaWkbTokenMajorUnit(self.padded_m, cdepth=1, bdepth=1)
        qb.projection_kernel(q_head, qa_mx, wqb, qa_scale, swqb, wkb_unit, wkb, q_latent_h, self.head_ready)""")
elif variant in ("separate-ub", "separate-ub-fit"):
    a=s.index("        # Q channels reuse the KV UB arena.")
    b=s.index("        if core_num < 8:",a)
    s=s[:a]+"        # Keep KV and Q allocations distinct to permit overlapping transfers.\n"+s[b:]
elif variant == "q-dedicated":
    a=s.index("                    batch_count = self.output_m // 8",s.index("                q_token = _MlaQHeadTokenEpilogMxfp8Unit("))
    b=s.index("        else:\n            q_bf16",a)
    s=s[:a]+"""                    if sub == 1:
                        for token_batch in range_constexpr(16):
                            batch = head * 16 + token_batch
                            q_token.process_batch(nope, combined, output, descale_q, batch)
"""+s[b:]
elif variant != "thin-control": raise ValueError(variant)
if variant in ("separate-ub-fit", "thin-control"):
    a=s.index("class _O11QbHeadOwnedMixT128:")
    b=s.index("class _O11MlaQHeadLocalEpilogUnit:",a)
    s=s[:a]+s[a:b].replace("depth=2", "depth=1")+s[b:]
    s=edit(s, "wkb_unit = _O11MlaWkbTokenMajorUnit(self.padded_m, cdepth=2, bdepth=2)",
           "wkb_unit = _O11MlaWkbTokenMajorUnit(self.padded_m, cdepth=1, bdepth=1)")
compile(s,str(p),"exec");p.write_text(s)
