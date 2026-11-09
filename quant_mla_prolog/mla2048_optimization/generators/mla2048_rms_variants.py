"""Generate experimental RMS reductions; never overwrite the baseline.

For the validated, 32-AIC T=2048 specialization only. These variants change
FP32 summation order and MUST pass the existing golden and poison/recovery
checks before msprof timings can count as results. No speedup is assumed.
"""

import argparse
import ast
from pathlib import Path
import re


def transform(source: str, scope: str) -> str:
    pattern = re.compile(
        r"(?P<indent> +)total = vdups\(0\.0, dtypes\.float32, mask=(?P<mask>scalar|mask1)\)\n"
        r"(?P=indent)for (?P<seg>segment|seg) in range_constexpr\((?P<count>KV_LORA // 64|\(Q_LORA // 64\))\):\n"
        r"(?P=indent)    value = vload\((?P<x>x_row|x), (?P=seg) \* 64\)\n"
        r"(?P=indent)    total = vadd\(\n"
        r"(?P=indent)        total,\n"
        r"(?P=indent)        vreduce_sum\(vmul\(value, value, mask=full\), mask=full\),\n"
        r"(?P=indent)        mask=(?P=mask),\n"
        r"(?P=indent)    \)"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 3:
        raise ValueError(f"Expected exactly three baseline RMS loops, got {len(matches)}")

    def replace(match):
        qa = match['x'] == 'x'
        if scope == 'qa' and not qa or scope == 'kv' and qa:
            return match[0]
        width = 'Q_LORA' if qa else 'KV_LORA'
        lines = ['# Experimental four independent vector sum chains.']
        for i in range(4):
            lines.append(f'rms_acc{i} = vdups(0.0, dtypes.float32, mask=full)')
        lines.append(f'for rms_group in range_constexpr({width} // 256):')
        for i in range(4):
            lines.extend([
                f'    rms_v{i} = vload({match["x"]}, rms_group * 256 + {i * 64})',
                f'    rms_acc{i} = vadd(rms_acc{i}, vmul(rms_v{i}, rms_v{i}, mask=full), mask=full)',
            ])
        lines.extend([
            'rms_sum01 = vadd(rms_acc0, rms_acc1, mask=full)',
            'rms_sum23 = vadd(rms_acc2, rms_acc3, mask=full)',
            'total = vreduce_sum(vadd(rms_sum01, rms_sum23, mask=full), mask=full)',
        ])
        return '\n'.join(match['indent'] + line for line in lines)

    result = pattern.sub(replace, source)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scope', choices=['qa', 'kv', 'both'], required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.baseline.resolve():
        parser.error('output must not overwrite baseline')
    result = transform(args.baseline.read_text(), args.scope)
    # Exclusive creation prevents accidentally replacing a previous experiment.
    with args.output.open('x') as output:
        output.write(result)


if __name__ == '__main__':
    main()
