set -euo pipefail
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
out=$(mktemp -d /workspace/prefill-offline.XXXXXX)
echo "$out"
cp /workspace/quant_mla_vdup.py "$out/quant_mla_prolog.py"
python3 /workspace/mla_next_variants.py "$out/quant_mla_prolog.py" q-vector-max
python3 /workspace/mla_prefill_offline.py "$out/quant_mla_prolog.py" all-static-fit /workspace/mla_best.py
timeout 180 python3 -c 'import runpy,sys; m=runpy.run_path(sys.argv[1]); m["_compile_o11_native_variant"](0,1); print("COMPILE_PASSED")' "$out/quant_mla_prolog.py" > "$out/compile.log" 2>&1
