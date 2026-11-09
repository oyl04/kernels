"""Build an opt-in candidate source, preserving all generic kernel classes."""
import ast
import difflib
from pathlib import Path
import re
import sys


def isolate(baseline, experiment):
    classes = {
        '_MlaQHeadTokenEpilogMxfp8Unit': '_MlaQHeadTokenEpilogMxfp8UnitT2048',
        '_O11MlaWkbTokenMajorLargeUnit': '_O11MlaWkbTokenMajorLargeUnitT2048',
        '_MlaRuntimePrefillKernel': '_MlaRuntimePrefillKernelT2048',
        '_MlaRuntimePrefillLauncher': '_MlaRuntimePrefillLauncherT2048',
    }
    tree = ast.parse(experiment)
    lines = experiment.splitlines(keepends=True)
    parts = []
    for name in classes:
        node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        parts.append(''.join(lines[first - 1:node.end_lineno]).rstrip())
    combined = '\n\n\n'.join(parts)
    combined = re.sub(r'\b(' + '|'.join(classes) + r')\b',
                      lambda m: classes[m.group(0)], combined)
    a = baseline.index('class _O11Workspaces(NamedTuple):')
    baseline = baseline[:a] + combined + '\n\n\n' + baseline[a:]

    def replace(old, new):
        nonlocal baseline
        assert baseline.count(old) == 1, old
        baseline = baseline.replace(old, new)

    replace('def _run_o11_single_launch(', '''def _runtime_prefill_profile(profile, tokens, quant_mode_c, core_num):
    if profile == 0 and tokens == 2048 and quant_mode_c == 1 and core_num == 32:
        return 2048
    return profile


def _run_o11_single_launch(''')
    replace('    program = _get_native_program(profile, quant_mode_c, float(norm_eps))',
            '''    dispatch_profile = _runtime_prefill_profile(profile, m, quant_mode_c, core_num)
    program = _get_native_program(dispatch_profile, quant_mode_c, float(norm_eps))''')
    replace('    lower, upper = _native_token_bounds(tile_m)', '''    special = tile_m == 2048
    if special and quant_mode_c != 1:
        raise ValueError("T2048 specialization only supports FP8 output")
    if special:
        tile_m = 0  # Keep the tested dynamic tensor descriptors unchanged.
    lower, upper = _native_token_bounds(tile_m)''')
    replace('    return cannbotdsl.compile(\n        launcher.run,', '''    if special:
        launcher = _MlaRuntimePrefillLauncherT2048(float(norm_eps), quant_mode_c)
    return cannbotdsl.compile(
        launcher.run,''')
    return baseline


if __name__ == '__main__':
    original = Path(sys.argv[1]).read_text()
    result = isolate(original, Path(sys.argv[2]).read_text())
    after = ast.parse(result)
    old_classes = {n.name: ast.dump(n) for n in ast.parse(original).body if isinstance(n, ast.ClassDef)}
    new_classes = {n.name: ast.dump(n) for n in after.body if isinstance(n, ast.ClassDef)}
    assert all(new_classes[k] == v for k, v in old_classes.items())
    selector, = [n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == '_runtime_prefill_profile']
    ns = {}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), '<dispatch-check>', 'exec'), ns)
    check = ns[selector.name]
    for t in (8, 32, 128, 129, 2047, 2048, 2049):
        for quant in (0, 1):
            for cores in (8, 16, 24, 32, 48):
                profile = 0 if t > 128 else t
                assert check(profile, t, quant, cores) == (2048 if (t, quant, cores) == (2048, 1, 32) else profile)
    destination = Path(sys.argv[3])
    with destination.open('x') as f:
        if destination.suffix == '.patch':
            target = 'samples/quant_mla_prolog/quant_mla_prolog.py'
            f.writelines(difflib.unified_diff(original.splitlines(keepends=True),
                                              result.splitlines(keepends=True),
                                              'a/' + target, 'b/' + target))
        else:
            f.write(result)
