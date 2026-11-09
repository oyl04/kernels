set -eu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
python3 -m pip install --no-deps /workspace/cannbot_dsl-0.7.0-cp312-cp312-manylinux_2_28_aarch64.whl
out=$(mktemp -d /workspace/rms4-offline.XXXXXX)
echo "COMPILE_ROOT=$out"
for variant in qa kv both; do
  echo "COMPILE $variant"
  status=0
  timeout -k 10 180 python3 -c 'import runpy,sys; m=runpy.run_path(sys.argv[1]); m["_compile_o11_native_variant"](0,1); print("COMPILE_PASSED")' "/workspace/mla2048_rms_${variant}.py" > "$out/$variant.log" 2>&1 || status=$?
  echo "STATUS $variant $status"
  tail -n 8 "$out/$variant.log"
done
