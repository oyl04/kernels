set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
repo=$(mktemp -d /tmp/mla-shared-probe.XXXXXX)
git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
cp /workspace/msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
timeout -k 10 300 python3 /workspace/mla2048_shared_probe.py "$repo"
