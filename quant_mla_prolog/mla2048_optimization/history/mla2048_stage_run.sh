set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
out=$(mktemp -d /workspace/mla-stage-diagnostics.XXXXXX)
echo RESULT_ROOT=$out
repo=$(mktemp -d /tmp/mla-stage-diagnostics.XXXXXX)
git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
cp /workspace/msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
python3 /workspace/mla2048_stage_diagnostics.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py" "$out/sources"
for stage in pad front qb wkb; do
  mkdir "$out/$stage"
  echo START=$stage
  status=0
  timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output="$out/$stage" python3 /workspace/mla2048_stage_bench.py "$repo" "$out/sources/$stage.py" > "$out/$stage/run.log" 2>&1 || status=$?
  echo STATUS=$stage:$status
  grep -E 'PASSED|COMPLETED|ABORT|Error|error' "$out/$stage/run.log" | tail -n 8 || true
  if grep -qE 'ABORT:|OutOfMemory|out of memory' "$out/$stage/run.log"; then exit 1; fi
done
echo DONE
