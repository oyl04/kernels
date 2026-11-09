"""Server-03-only, one-shot idle-card waiter and RMS experiment runner."""
import csv
import fcntl
import json
from pathlib import Path
import re
import subprocess as sp
import time

ROOT = Path('/root/attention_testing_950_workspace')


def run(args):
    return sp.check_output(args, text=True, timeout=30)


def free_cards():
    text = run(['npu-smi', 'info'])
    # Fail closed unless the expected eight-device table is present.
    memory = re.findall(r'\|\s*(\d+)\s*\|\s*Ascend950DT[^\n]*\n\|[^\n]*?\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)', text)
    busy = {int(x) for x in re.findall(r'\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\S+\s*\|\s*\d+\s*\|', text)}
    if len(memory) != 8:
        raise RuntimeError('Unrecognized npu-smi table; refusing to launch')
    return {int(i) for i, util, used, total in memory
            if int(util) == 0 and int(used) < 8192 and int(i) not in busy}


def main():
    lock = (ROOT / 'rms4-background.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / ('rms4-background-' + time.strftime('%Y%m%d-%H%M%S'))
    output.mkdir()
    print('RESULTS', output, flush=True)
    deadline = time.monotonic() + 86400
    results = []
    for index, variant in enumerate(['control', 'qa', 'kv', 'both', 'qa', 'kv', 'both', 'control']):
        while True:
            if (ROOT / 'rms4-background.cancel').exists():
                print('CANCELLED before next launch', flush=True)
                return
            if time.monotonic() > deadline:
                raise TimeoutError('24-hour waiting budget expired')
            try:
                first = free_cards()
                if first:
                    time.sleep(15)
                    available = first & free_cards()
                    if available:
                        card = min(available)
                        break
            except Exception as error:
                print('WAIT_CHECK', str(error), flush=True)
            time.sleep(60)
        name = f'attention_testing_950_rms4_{output.name[-15:]}_{index}'
        dest = output / f'{index}-{variant}'
        dest.mkdir()
        source = (ROOT / 'msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py'
                  if variant == 'control' else ROOT / f'mla2048_rms_{variant}.py')
        script = f'''set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
python3 -c 'import torch,torch_npu,runpy; t=torch.empty(1,device="npu:0"); s=runpy.run_path("test/quant_mla_prolog/test_quant_mla_prolog.py"); assert s["public_module"]._tensor_aic_core_num(t)==32'
repo=$(mktemp -d /tmp/rms4-test.XXXXXX)
git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
cp /workspace/{source.relative_to(ROOT)} "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output=/workspace/{dest.relative_to(ROOT)} python3 /workspace/bench_mla_2048.py "$repo"
'''
        args = ['docker', 'run', '--name', name, '--network', 'none', '--cgroupns', 'host', '--shm-size', '1g']
        for device in [f'/dev/davinci{card}', '/dev/davinci_manager', '/dev/hisi_hdc', '/dev/uburma', '/dev/ummu', '/dev/ubcore/ubcore']:
            args += ['--device', device]
        args += ['-v', '/usr/lib64/urma:/usr/lib64/urma:ro', '-v', f'{ROOT}:/workspace',
                 '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
                 '-v', '/etc/ascend_install.info:/etc/ascend_install.info:ro',
                 '-e', 'ASCEND_RT_VISIBLE_DEVICES=0', '-e', 'PYTHONUNBUFFERED=1',
                 '-e', 'OMP_NUM_THREADS=2', '--entrypoint', '/bin/bash',
                 'deepinfer-restored:20261028', '-c', script]
        print('START', variant, 'physical_card', card, name, flush=True)
        with (dest / 'run.log').open('w') as log:
            process = sp.Popen(args, stdout=log, stderr=sp.STDOUT)
            try:
                code = process.wait(timeout=600)
            except sp.TimeoutExpired:
                sp.run(['docker', 'stop', '-t', '10', name], timeout=30, check=False)
                process.wait(timeout=30)
                code = 124
        records = []
        for path in dest.rglob('*.csv'):
            with path.open() as data:
                reader = csv.DictReader(data)
                for row in reader:
                    if '_MlaRuntimePrefillKernel' in row.get('Op Name', ''):
                        records.append(row)
        passed = code == 0 and 'T2048 golden and recovery passed; 30 measured invocations' in (dest / 'run.log').read_text() and len(records) == 47
        mean = None
        if passed:
            records.sort(key=lambda r: float(r['Task Start Time(us)']))
            mean = sum(float(r['Task Duration(us)']) for r in records[-30:]) / 30
        results.append(dict(variant=variant, card=card, exit_code=code, validated=passed, records=len(records), mean_us=mean))
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print('RESULT', results[-1], flush=True)
    print('DONE', output, flush=True)


if __name__ == '__main__':
    main()
