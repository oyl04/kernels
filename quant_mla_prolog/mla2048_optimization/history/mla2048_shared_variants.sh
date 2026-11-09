set -eu
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
out=$(mktemp -d /workspace/rms4-shared.XXXXXX)
echo "RESULT_ROOT=$out"
for variant in control qa kv both qa-repeat kv-repeat both-repeat control-repeat; do
  repo=$(mktemp -d /tmp/rms4-shared.XXXXXX)
  git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
  kind=${variant%-repeat}
  source=/workspace/mla2048_rms_${kind}.py
  if [ "$kind" = control ]; then source=/workspace/msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py; fi
  cp "$source" "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
  dest="$out/$variant"
  mkdir "$dest"
  cp "$source" "$dest/quant_mla_prolog.py"
  echo "START $variant"
  status=0
  timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output="$dest" python3 /workspace/mla2048_shared_probe.py "$repo" > "$dest/run.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$dest/exit.status"
  echo "EXIT $variant $status"
  grep -E 'DEVICE_MEMORY|PEAK_TORCH|passed|ABORT|AssertionError|OutOfMemory|Error:' "$dest/run.log" | tail -n 12 || true
  # Abort the batch on memory pressure; a numerical failure can still be compared.
  if grep -qE 'ABORT:|OutOfMemory|out of memory' "$dest/run.log"; then echo MEMORY_ABORT; exit 1; fi
done
echo DONE
