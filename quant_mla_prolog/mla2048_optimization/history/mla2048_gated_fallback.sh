set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
repo=$(mktemp -d /tmp/mla-gated-fallback.XXXXXX)
git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
cp /workspace/mla2048_wkbquartersgatedv2.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
cd "$repo"
timeout -k 10 180 python3 -m pytest -q test/quant_mla_prolog/test_quant_mla_prolog.py
