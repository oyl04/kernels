set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
out=$(mktemp -d /workspace/qbatch-shared.XXXXXX)
echo RESULT_ROOT=$out
for variant in ${MLA_QBATCH_VARIANTS:-control qbatch16 qbatch32 control-repeat}; do
  repo=$(mktemp -d /tmp/qbatch-shared.XXXXXX)
  git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
  source=/workspace/mla2048_${variant%-repeat}.py
  case "$variant" in control*) source=/workspace/msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py ;; esac
  cp "$source" "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
  dest="$out/$variant"
  mkdir "$dest"
  cp "$source" "$dest/quant_mla_prolog.py"
  echo START=$variant
  status=0
  timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output="$dest" python3 /workspace/mla2048_shared_probe.py "$repo" > "$dest/run.log" 2>&1 || status=$?
  echo STATUS=$variant:$status
  grep -E 'passed|ABORT|Error|error|PEAK' "$dest/run.log" | tail -n 8 || true
  if grep -qE 'ABORT:|OutOfMemory|out of memory' "$dest/run.log"; then exit 1; fi
done
echo DONE
