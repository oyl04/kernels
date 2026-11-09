set -euo pipefail
cd /workspace/cannbot-dsl
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
python3 -c 'import torch, torch_npu, runpy; t=torch.empty(1,device="npu:0"); s=runpy.run_path("test/quant_mla_prolog/test_quant_mla_prolog.py"); assert s["public_module"]._tensor_aic_core_num(t)==32'
out=$(mktemp -d /workspace/msprof-2048-continue.XXXXXX)
echo "PROFILE_ROOT=$out"
for variant in control overlap m128 overlap-m128 control-repeat; do
  repo=$(mktemp -d /tmp/mla2048-continue.XXXXXX)
  git archive 1342a261519dead926fff56c49400cdea72e42bc | tar -x -C "$repo"
  target="$repo/samples/quant_mla_prolog/quant_mla_prolog.py"
  cp /workspace/msprof-prefill-port.ruRngW/qb-m64/quant_mla_prolog.py "$target"
  case "$variant" in
    overlap*) cp /workspace/prefill-offline.XwUl2Z/quant_mla_prolog.py "$target" ;;
  esac
  case "$variant" in
    *m128)
      sed -i '/class _O11MlaWkbTokenMajorLargeUnit:/,/class _MlaPrefillQbUnit:/s/self.bm = 64/self.bm = 128/' "$target"
      sed -i 's/_O11MlaWkbTokenMajorLargeUnit(cdepth=2, bdepth=2)/_O11MlaWkbTokenMajorLargeUnit(cdepth=1, bdepth=2)/' "$target"
      ;;
  esac
  dest="$out/$variant"
  mkdir "$dest"
  cp "$target" "$dest/quant_mla_prolog.py"
  echo "RUN $variant"
  status=0
  timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output="$dest" python3 /workspace/bench_mla_2048.py "$repo" > "$dest/run.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$dest/exit.status"
  echo "STATUS $variant $status"
done
