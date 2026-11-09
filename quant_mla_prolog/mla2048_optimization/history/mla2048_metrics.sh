set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
out=$(mktemp -d /workspace/mla-metrics.XXXXXX)
echo RESULT_ROOT=$out
repo=$(mktemp -d /tmp/mla-metrics.XXXXXX)
git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
cp /workspace/msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
for metric in PipeUtilization Memory; do
  echo START=$metric
  mkdir "$out/$metric"
  status=0
  timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --aic-metrics="$metric" --output="$out/$metric" python3 /workspace/mla2048_shared_probe.py "$repo" > "$out/$metric/run.log" 2>&1 || status=$?
  echo STATUS=$metric:$status
  grep -E 'passed|ABORT|Error|error|PEAK' "$out/$metric/run.log" | tail -n 8 || true
done
echo DONE
