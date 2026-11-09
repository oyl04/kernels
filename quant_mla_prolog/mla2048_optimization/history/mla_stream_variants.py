from pathlib import Path
import sys

path = Path(sys.argv[1])
variant = sys.argv[2]
source = path.read_text()

def edit(body, old, new):
    assert body.count(old) == 1, old
    return body.replace(old, new)

if variant in ("early-tile", "early-fixpipe"):
    start = source.index("class _O11MlaWkbTokenMajorUnit:")
    end = source.index("class _MlaInputTailUnit:", start)
    wkb = source[start:end].rstrip()
    wkb = edit(wkb, "q_latent, head):", "q_latent, head, ready):")
    wkb += """
            cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_block_arrive(_sync_ops.PIPE.MTE2, ready + m_tile, mode=2)


"""
    if variant == "early-fixpipe":
        wkb = edit(wkb, """            cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_block_arrive(_sync_ops.PIPE.MTE2, ready + m_tile, mode=2)
""", """            cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, ready + m_tile, mode=2)
""")
    source = source[:start] + wkb + source[end:]
    start = source.index("                for head in range(block, N_HEADS, core_num):", source.index("                q_token = _MlaQHeadTokenEpilogMxfp8Unit("))
    end = source.index("        else:\n            q_bf16", start)
    source = source[:start] + """                for head in range(block, N_HEADS, core_num):
                    ready = self.head_ready + 2 * (head // core_num)
                    for m_tile in range_constexpr(2):
                        vec_sync_block_wait(_sync_ops.PIPE.S, ready + m_tile, mode=2)
                        if sub == m_tile:
                            for local_batch in range_constexpr(8):
                                batch = head * 16 + m_tile * 8 + local_batch
                                q_token.process_batch(nope, combined, output, descale_q, batch)
""" + source[end:]
    start = source.index("            wkb_unit.process_head(q_head, wkb, q_latent_h, head)")
    end = source.index("        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.tail_done", start)
    source = source[:start] + """            ready = self.head_ready + 2 * (head // core_num)
            wkb_unit.process_head(q_head, wkb, q_latent_h, head, ready)
""" + source[end:]
elif variant in ("crosscore", "crosscore-f32", "crosscore-producer-first", "crosscore-peer-loop"):
    source = edit(source, "from cannbotdsl import MemLoc", "from cannbotdsl import ChannelKind, MemLoc")
    start = source.index("class _MlaQHeadTokenEpilogMxfp8Unit:")
    end = source.index("class _MlaQHeadTokenEpilogBf16Unit:", start)
    epilog = source[start:end]
    direct = epilog[epilog.index("    @jit\n    def process_batch("):].rstrip()
    direct = edit(direct, "def process_batch(self, q_nope, q_rope, q_out, descale_q, batch):",
                  "def process_stream(self, nope0, nope1, nope2, nope3, q_rope, q_out, descale_q, batch, local_batch):")
    direct = edit(direct, "        nope_slot = self.nope_channel.produce()\n", "")
    direct = edit(direct, "        mem_copy(nope_slot, tile_slice(q_nope, (8, KV_LORA), (source_batch, 0)))\n", "")
    direct = edit(direct, "        q_nope_rows = self.nope_channel.consume()\n", "")
    old = """                    unpacked = vload_unpack(
                        q_nope_rows,
                        row * KV_LORA + segment * 64,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    )"""
    new = ""
    for n in range(4):
        branch = "if" if n == 0 else "elif" if n < 3 else "else:"
        if n < 3:
            branch += f" const_expr(segment // 2 == {n}):"
        new += f"""                    {branch}
                        unpacked = vload_unpack(
                            nope{n},
                            (local_batch * 8 + row) * 128 + (segment % 2) * 64,
                            unpack_mode=UnpackMode.B16_TO_B32,
                        )
"""
    assert direct.count(old) == 2
    direct = direct.replace(old, new.rstrip())
    if variant != "crosscore":
        for n in range(4):
            direct = direct.replace(f"""unpacked = vload_unpack(
                            nope{n},
                            (local_batch * 8 + row) * 128 + (segment % 2) * 64,
                            unpack_mode=UnpackMode.B16_TO_B32,
                        )""", f"""raw = vload(nope{n}, (local_batch * 8 + row) * 128 + (segment % 2) * 64)""")
        direct = direct.replace("                    value = vcast(unpacked, dtypes.float32, mask=full)",
                                """                    rounded = vcast(raw, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN)
                    value = vcast(rounded, dtypes.float32, mask=full)""")
    source = source[:end] + direct + "\n\n\n" + source[end:]

    start = source.index("class _O11MlaWkbTokenMajorUnit:")
    end = source.index("class _MlaInputTailUnit:", start)
    body = source[start:end]
    body = edit(body, "self.fixpipe = make_copy_engine()", "self.fixpipe = make_copy_engine(split_axis=0)")
    body = edit(body, "q_latent, head):", "stream0, stream1, stream2, stream3, head):")
    body = edit(body, "        out_flat = q_latent.view((N_HEADS * self.padded_m, KV_LORA))\n        out_head = tile_slice(out_flat, (self.padded_m, KV_LORA), (head, 0))\n", "")
    old = """                mem_copy(
                    tile_slice(out_head, (self.bm, 128), (m_tile, n_tile)),
                    self.c_l0.consume(),
                    engine=self.fixpipe,
                )"""
    new = ""
    for n in range(4):
        branch = "if" if n == 0 else "elif" if n < 3 else "else:"
        if n < 3:
            branch += f" const_expr(n_tile == {n}):"
        new += f"""                {branch}
                    mem_copy(stream{n}.produce(), self.c_l0.consume(), engine=self.fixpipe)
"""
    body = edit(body, old, new.rstrip())
    source = source[:start] + body + source[end:]

    start = source.index("        if core_num < 8:\n            vec_sync_block_wait(_sync_ops.PIPE.S, self.head_ready", source.index("class _MlaRuntimeTileUnit:"))
    end = source.index("        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.tail_done", start)
    if variant != "crosscore":
        end = source.index("        global_sync_all()", end)
    dtype = "float32" if variant != "crosscore" else "bfloat16"
    channels = "".join(
        f"        stream{n} = Channel(MemLoc.UB, (32, 128), dtypes.{dtype}, depth=2, kind=ChannelKind.CrossCore)\n"
        for n in range(4)
    )
    source = source[:start] + channels + """        q_token = _MlaQHeadTokenEpilogMxfp8Unit(
            64, self.padded_m, self.output_m, token_major_rope=True
        )
        combined = q_head.view((self.padded_m, N_HEADS * HEAD_DIM))
        output = q_out.view((self.output_m, N_HEADS * FEATURES))
        for head in range(block, N_HEADS, core_num):
            for m_tile in range_constexpr(2):
                nope0 = stream0.consume()
                nope1 = stream1.consume()
                nope2 = stream2.consume()
                nope3 = stream3.consume()
                for local_batch in range_constexpr(4):
                    batch = head * 16 + m_tile * 8 + sub * 4 + local_batch
                    q_token.process_stream(
                        nope0, nope1, nope2, nope3, combined, output, descale_q,
                        batch, local_batch,
                    )
        vec_sync_all()
        vec_sync_block_arrive(_sync_ops.PIPE.MTE3, self.tail_done, mode=2)
        wkb_unit = _O11MlaWkbTokenMajorUnit(self.padded_m, cdepth=2, bdepth=2)
        for head in range(block, N_HEADS, core_num):
            wkb_unit.process_head(q_head, wkb, stream0, stream1, stream2, stream3, head)
""" + source[end:]
    if variant != "crosscore":
        source = source.replace("        vec_sync_block_arrive(_sync_ops.PIPE.MTE3, self.tail_done, mode=2)\n        wkb_unit = _O11MlaWkbTokenMajorUnit",
                                "        wkb_unit = _O11MlaWkbTokenMajorUnit", 1)
    if variant in ("crosscore-producer-first", "crosscore-peer-loop"):
        start = source.index("        q_token = _MlaQHeadTokenEpilogMxfp8Unit(", source.index("        stream0 = Channel("))
        mid = source.index("        wkb_unit = _O11MlaWkbTokenMajorUnit(", start)
        end = source.index("        global_sync_all()", mid)
        if variant == "crosscore-producer-first":
            source = source[:start] + source[mid:end] + source[start:mid] + source[end:]
        else:
            producer = source[mid:end]
            loop = "        for head in range(block, N_HEADS, core_num):\n"
            declaration, send = producer.split(loop)
            consumer = edit(source[start:mid], loop, loop + send)
            source = source[:start] + declaration + consumer + source[end:]
elif variant == "front-reuse":
    start = source.index("    @jit\n    def project_kva_tile(", source.index("class _O11MlaRef17FrontUnit:"))
    end = source.index("class _O11QbHeadMixNoAlias:", start)
    pair = source[start:end].rstrip()
    pair = edit(pair, "def project_kva_tile(self, kva_gm, x_gm, wkva_gm, sx_gm, swkva_gm, block, m_idx):",
                "def project_qa_kva_pair(self, qa_gm, kva_gm, x_gm, wqa_gm, wkva_gm, sx_gm, swqa_gm, swkva_gm, m_idx):")
    pair = edit(pair, '"""Run one complete N64 tile using regular K1024 FIFO windows."""',
                '"""Share the activation transfer and L0A operand for QA0 and KVA8."""')
    pair = edit(pair, "        weight = _fp8_nz_transposed_view(wkva_gm, FEATURES, DIM)",
                """        weight = _fp8_nz_transposed_view(wkva_gm, FEATURES, DIM)
        qa_weight = _fp8_nz_transposed_view(wqa_gm, Q_LORA, DIM)
        qa_scales = Channel(MemLoc.L1, (self.scale_l1_len, bn), Float8E8M0, depth=1, data_format="nz")
        qa_acc = Channel(MemLoc.L0C, (self.m, bn), dtypes.float32, depth=1)
        block = 8""")
    pair = edit(pair, "        scale_b = self.l1_scale_b.consume()",
                """        mem_copy(
            qa_scales.produce(),
            tile_slice(swqa_gm, (bn, self.scale_groups, 2), (0, 0, 0)),
            engine=self.copy_scale_b,
        )
        qa_scale = qa_scales.consume()
        qa_write = qa_acc.produce()
        scale_b = self.l1_scale_b.consume()""")
    pair = edit(pair, "            l1_b_read = self.l1_b.consume()",
                """            mem_copy(self.l1_b.produce(), tile_slice(qa_weight, (kl1, bn), (k1, 0)))
            l1_b_read = self.l1_b.consume()
            qa_b_fp8 = self.l1_b.consume().reinterpret(dtype=Float8E4M3FN)""")
    pair = edit(pair, "                final = global_k0 + 1 == DIM // bk",
                """                final = global_k0 + 1 == DIM // bk
                a_read = self.l0a.consume()""")
    pair = edit(pair, "                    self.l0a.consume(),", "                    a_read,")
    at = pair.index("        mem_copy(\n            tile_slice(kva_gm")
    pair = pair[:at] + """                mem_copy(
                    self.l0b.produce(),
                    tile_slice(qa_b_fp8, (bk, bn), (k0, 0)),
                    transpose=True,
                    mx_scale=tile_slice(qa_scale, (self.scale_l0_len, bn), (global_k0, 0)),
                )
                matmul(
                    qa_write, a_read, self.l0b.consume(),
                    init=(global_k0 == 0), unit_flag=3 if final else 2,
                )
        mem_copy(tile_slice(qa_gm, (self.m, bn), (m_idx, 0)),
                 qa_acc.consume(), engine=self.fixpipe, unit_flag=3)
""" + pair[at:]
    source = source[:end] + pair + "\n\n\n" + source[end:]
    source = edit(source, "        if block < qa_tiles:\n            front.project_qa_owned_tile",
                  """        if block == 0:
            front.project_qa_kva_pair(qa_raw, kva_raw, x, wqa, wkva, sx, swqa, swkva, 0)
        elif block < qa_tiles:
            front.project_qa_owned_tile""")
    source = edit(source, """        # Keep the remaining KVA tile after the QA-ready handoff.
        if block == 0:
            tail_front = _O11MlaRef17FrontUnit(self.m)
            tail_front.project_kva_tile(kva_raw, x, wkva, sx, swkva, 8, 0)

""", "")
else:
    raise ValueError(variant)

compile(source, str(path), "exec")
path.write_text(source)
