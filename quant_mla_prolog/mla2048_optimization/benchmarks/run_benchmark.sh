#!/usr/bin/env bash
# Assumes the intended Ascend card is already mapped to logical device 0.
set -euo pipefail
if (( $# < 4 )); then
  echo 'Usage: bash run_benchmark.sh REPO NEW_RESULTS_DIR NPU_SMI_ID VARIANT...' >&2
  echo 'Variants: original baseline8 q16 half quarters gated' >&2
  exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(dirname -- "$here")
repo=$(cd -- "$1" && pwd)
out=$2
card=$3
shift 3
[[ $card =~ ^[0-9]+$ ]] || { echo 'Invalid NPU ID' >&2; exit 2; }
commit=1342a261519dead926fff56c49400cdea72e42bc
git -C "$repo" cat-file -e "$commit^{commit}"
mkdir -- "$out"
out=$(cd -- "$out" && pwd)

index=0
for variant in "$@"; do
  case "$variant" in
    original) source="$root/reference/samples/quant_mla_prolog/quant_mla_prolog.py" ;;
    baseline8) source="$root/sources/quant_mla_t2048_validated.py" ;;
    q16) source="$root/sources/mla2048_qbatch16fixed.py" ;;
    half) source="$root/sources/mla2048_wkbhalfshared.py" ;;
    quarters) source="$root/sources/mla2048_wkbquarter.py" ;;
    gated) source="$root/sources/mla2048_wkbquartersgatedv2.py" ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
  esac
  occupancy=$(npu-smi info)
  if ! grep -Eq "No running processes found in NPU ${card}[[:space:]]" <<< "$occupancy"; then
    echo "No explicit idle confirmation for NPU $card; refusing to start" >&2
    exit 1
  fi
  index=$((index + 1))
  dest="$out/$index-$variant"
  mkdir -- "$dest" "$dest/repo"
  git -C "$repo" archive "$commit" | tar -xf - -C "$dest/repo"
  cp -- "$source" "$dest/repo/samples/quant_mla_prolog/quant_mla_prolog.py"
  echo "Running $variant: $dest"
  (
    cd -- "$dest/repo"
    timeout -k 10 300 msprof --task-time=l0 --aicore-shape=on --output="$dest" \
      python3 "$here/mla2048_shared_probe.py" "$dest/repo"
  ) > "$dest/run.log" 2>&1
  python3 "$here/summarize.py" "$dest" "$variant"
done
