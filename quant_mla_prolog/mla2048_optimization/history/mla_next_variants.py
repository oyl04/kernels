from pathlib import Path
import sys

path = Path(sys.argv[1])
variant = sys.argv[2]
source = path.read_text()

def edit(body, old, new):
    assert body.count(old) == 1, old
    return body.replace(old, new)

if variant in ("q-vector-max", "kva-pair-q-vector-max"):
    start = source.index("class _MlaQHeadTokenEpilogMxfp8Unit:")
    end = source.index("class _MlaQHeadTokenEpilogBf16Unit:", start)
    body = source[start:end]
    body = edit(body, "absmax = vdups(1.0e-8, dtypes.float32, mask=scalar)",
                "lane_max = vdups(1.0e-8, dtypes.float32, mask=full)")
    body = edit(body, """                    absmax = vmax(
                        absmax,
                        vreduce_max(vabs(value, mask=full), mask=full),
                        mask=scalar,
                    )""", """                    lane_max = vmax(lane_max, vabs(value, mask=full), mask=full)""")
    body = edit(body, """                absmax = vmax(
                    absmax,
                    vreduce_max(vabs(value, mask=full), mask=full),
                    mask=scalar,
                )""", """                lane_max = vmax(lane_max, vabs(value, mask=full), mask=full)
                absmax = vreduce_max(lane_max, mask=full)""")
    source = source[:start] + body + source[end:]

if variant == "q-register-reuse":
    start = source.index("class _MlaQHeadTokenEpilogMxfp8Unit:")
    end = source.index("class _MlaQHeadTokenEpilogBf16Unit:", start)
    body = source[start:end]
    first = body.index("                absmax = vdups(")
    last = body.index("        output_rows = self.output_channel.consume()", first)
    rows = ["                lane_max = vdups(1.0e-8, dtypes.float32, mask=full)"]
    for segment in range(9):
        if segment == 8:
            src, offset = "q_rope_rows", "row * ROPE"
        else:
            src, offset = "q_nope_rows", f"row * KV_LORA + {segment * 64}"
        rows += [
            f"                raw = vload_unpack({src}, {offset}, unpack_mode=UnpackMode.B16_TO_B32)",
            f"                value_{segment} = vcast(raw, dtypes.float32, mask=full)",
            f"                lane_max = vmax(lane_max, vabs(value_{segment}, mask=full), mask=full)",
        ]
    rows += [
        "                absmax = vreduce_max(lane_max, mask=full)",
        "                scale = vmuls(absmax, 1.0 / FP8_MAX, mask=scalar)",
        "                vstore_first(scales, row, scale)",
        "                scale_vector = vdup(scale, mask=full)",
    ]
    for segment in range(9):
        rows += [
            f"                quantized = vdiv(value_{segment}, scale_vector, mask=full)",
            "                fp8_value = vcast(quantized, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO)",
            f"                vstore_pack(encoded, row * FEATURES + {segment * 64}, fp8_value, full, pack_mode=PackMode.B32_TO_B8)",
        ]
    body = body[:first] + "\n".join(rows) + "\n" + body[last:]
    source = source[:start] + body + source[end:]

if variant in ("kva-pair", "kva-pair-q-vector-max"):
    start = source.index("    @jit\n    def project_qa_kva_pair(")
    end = source.index("class _O11QbHeadMixNoAlias:", start)
    body = source[start:end]
    body = edit(body, "def project_qa_kva_pair(self, qa_gm, kva_gm, x_gm, wqa_gm, wkva_gm, sx_gm, swqa_gm, swkva_gm, m_idx):",
                "def project_kva_pair(self, kva_gm, x_gm, wkva_gm, sx_gm, swkva_gm, m_idx):")
    body = edit(body, '"""Share the activation transfer and L0A operand for QA0 and KVA8."""',
                '"""Share activation loads for KVA tiles 7 and 8."""')
    body = edit(body, "        qa_weight = _fp8_nz_transposed_view(wqa_gm, Q_LORA, DIM)\n", "")
    body = edit(body, "tile_slice(swqa_gm, (bn, self.scale_groups, 2), (0, 0, 0))",
                "tile_slice(swkva_gm, (bn, self.scale_groups, 2), (7, 0, 0))")
    body = edit(body, "tile_slice(qa_weight, (kl1, bn), (k1, 0))",
                "tile_slice(weight, (kl1, bn), (k1, 7))")
    body = edit(body, "tile_slice(qa_gm, (self.m, bn), (m_idx, 0))",
                "tile_slice(kva_gm, (self.m, bn), (m_idx, 7))")
    body = body.replace("qa_", "paired_")
    source = source[:start] + body + source[end:]
    source = edit(source, """        if block == 0:
            front.project_qa_kva_pair(qa_raw, kva_raw, x, wqa, wkva, sx, swqa, swkva, 0)
        elif block < qa_tiles:
            front.project_qa_owned_tile""",
                  """        if block == 31:
            front.project_kva_pair(kva_raw, x, wkva, sx, swkva, 0)
        elif block < qa_tiles:
            front.project_qa_owned_tile""")
elif variant == "serial-kva-pair":
    source = edit(source, """        if block == 0:
            front.project_qa_kva_pair(qa_raw, kva_raw, x, wqa, wkva, sx, swqa, swkva, 0)
        elif block < qa_tiles:
            front.project_qa_owned_tile(qa_raw, x, wqa, sx, swqa, block, 0)
        else:
            front.project_kva_tile(kva_raw, x, wkva, sx, swkva, block - qa_tiles, 0)
""", """        if block < qa_tiles:
            front.project_qa_owned_tile(qa_raw, x, wqa, sx, swqa, block, 0)
        else:
            front.project_kva_tile(kva_raw, x, wkva, sx, swkva, block - qa_tiles, 0)
            if block == 31:
                front.project_kva_tile(kva_raw, x, wkva, sx, swkva, 8, 0)
""")
elif variant == "split-front":
    source = edit(source, """        if block == 0:
            front.project_qa_kva_pair(qa_raw, kva_raw, x, wqa, wkva, sx, swqa, swkva, 0)
        elif block < qa_tiles:
            front.project_qa_owned_tile(qa_raw, x, wqa, sx, swqa, block, 0)
        else:
            front.project_kva_tile(kva_raw, x, wkva, sx, swkva, block - qa_tiles, 0)
        _o11_front_sync_aic_to_aiv(self.m == 128)
""", """        if block < qa_tiles:
            front.project_qa_owned_tile(qa_raw, x, wqa, sx, swqa, block, 0)
        else:
            front.project_kva_tile(kva_raw, x, wkva, sx, swkva, block - qa_tiles, 0)
        channel_rewind(reset_sync_id=True)
        if block < 2:
            tail_front = _O11MlaRef17FrontUnit(64)
            tail_front.project_kva_tile(kva_raw, x, wkva, sx, swkva, 8, block)
        _o11_front_sync_aic_to_aiv(self.m == 128)
""")
elif variant not in ("q-vector-max", "q-register-reuse"):
    raise ValueError(variant)

compile(source, str(path), "exec")
path.write_text(source)
