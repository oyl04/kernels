"""Create isolated one-change candidates on top of the measured vdup source."""

from pathlib import Path
import sys

path = Path(sys.argv[1])
variant = sys.argv[2]
source = path.read_text()


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one match for {old!r}")
    source = source.replace(old, new)


if variant == "l2":
    for target in (
        "tile_slice(q_out, (8, FEATURES), (token_batch, head)), output_rows",
        "tile_slice(descale_q, (8, 1), (token_batch, head)), output_scales",
    ):
        replace_once(f"mem_copy({target})", f"mem_copy(\n            {target}, l2_cache_ctl=4\n        )")
    for target, slot in (
        ("tile_slice(cache_flat, (1, 32), (offset, 0))", "tile_slice(output_row, (1, 32), (0, feature_block))"),
        ("tile_slice(output, (4, FEATURES), (token_batch, head))", "output_rows"),
        ("tile_slice(scales_out, (4, 1), (token_batch, head))", "output_scales"),
    ):
        indent = " " * (20 if target.startswith("tile_slice(cache_flat") else 12)
        old = f"{indent}{target},\n{indent}{slot},\n"
        replace_once(old, old + f"{indent}l2_cache_ctl=4,\n")
elif variant == "wkb-depth2":
    replace_once("depth=1 if padded_m == 128 else 2,", "depth=2,")
elif variant == "front-k512":
    replace_once("self.base_n, self.base_k, self.k_l1 = 64, 256, 1024",
                 "self.base_n, self.base_k, self.k_l1 = 64, 256, 512")
elif variant in ("qb-head-owned-32", "qb-local-32"):
    # This experiment is only valid for the profiled 32-AIC T128 launch.
    replace_once("        qb = _O11QbHeadMixNoAlias(self.m)\n", "")
    replace_once(
        "        qb.projection_kernel(q_head, qa_mx, wqb, qa_scale, swqb)\n",
        "        qb = _O11QbHeadOwnedMixT128(self.m, 32)\n"
        "        qb.projection_kernel(q_head, qa_mx, wqb, qa_scale, swqb)\n",
    )
    if variant == "qb-local-32":
        start = source.index("        # QB assigns N128 tiles")
        end = source.index("        if const_expr(self.m == 128):", start)
        replace_once(source[start:end], "        cube_sync_all()\n")
else:
    raise ValueError(variant)

compile(source, str(path), "exec")
path.write_text(source)
