set -euo pipefail
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
python3 -c 'import torch, torch_npu, runpy; t=torch.empty(1,device="npu:0"); s=runpy.run_path("test/quant_mla_prolog/test_quant_mla_prolog.py"); assert s["public_module"]._tensor_aic_core_num(t)==32'
out=$(mktemp -d /workspace/msprof-prefill-port.XXXXXX)
echo "PROFILE_ROOT=$out"
for variant in control m64 qb-owned front-pair all; do
  repo=$(mktemp -d /tmp/mla2048.XXXXXX)
  git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
  cp /workspace/quant_mla_vdup.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
  python3 /workspace/mla_next_variants.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py" q-vector-max
  if [ "$variant" != control ]; then python3 /workspace/mla_prefill_variants.py "$repo/samples/quant_mla_prolog/quant_mla_prolog.py" "$variant" /workspace/mla_best.py; fi
  dest="$out/$variant"
  mkdir "$dest"
  cp "$repo/samples/quant_mla_prolog/quant_mla_prolog.py" "$dest/quant_mla_prolog.py"
  status=0
  timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output="$dest" python3 /workspace/bench_mla_2048.py "$repo" > "$dest/run.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$dest/exit.status"
  echo "STATUS $variant $status"
done
