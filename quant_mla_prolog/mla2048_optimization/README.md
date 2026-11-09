# Ascend 950 quantized MLA prefill optimization

## Result and scope

The fastest validated candidate averages **735.5 µs** for **T=2048, FP8 output (C1), and 32 AICs**. Two runs on an otherwise idle card measured **735.101 and 735.807 µs**, versus **752.345 µs** for an interleaved two-chunk control: **2.24% lower latency**. That is also about **10.5% lower latency** than the earlier 822.195 µs 8-row control on the same idle card. This latter comparison is across batches, not a matched estimate. Do not add percentage gains from different sessions.

The candidate uses a 16-row Q epilogue and four 512-token WKB-to-Q handoffs. Runtime dispatch requires exactly the tested configuration. Existing kernel classes, fallback profiles, and the AOT export list are unchanged. Both T=2048 runs passed CPU-golden precision and poisoned-workspace recovery checks. The three existing small-token FP8 fallback tests passed. The patch is **not applied or deployed**; other large-token lengths, BF16, and AOT use of the new path were not hardware-qualified.

## Bottlenecks found

| Finding | Evidence | What we fixed or still need to verify |
|---|---|---|
| Q epilogue batching | 16 rows beat 8 rows after restoring the original quantization floor; 32 rows regressed even with split vector work | Keep 16 rows. The cause of the 32-row cliff is not established. |
| WKB-to-Q readiness | Q originally waited for a complete head; two chunks won, and four won again with interleaved controls | Signal after each 512-token chunk, retaining one WKB buffer set. |
| Front projections + QA normalization | Largest cumulative prefix increment, approximately 344 µs beyond padding | Still open. The tested simple overlap gave no gain; paired input reuse ran into compiler or buffer limits. |
| Transfers and output writes | Whole-kernel pipeline counters showed substantial MTE2 and Fixpipe activity | Profile by phase and core before attributing stalls or redesigning layout. |

### Stage diagnostics

The partial kernels were timed after validating the full baseline. They intentionally omit later outputs and were not full-operator correctness tests.

| Cumulative prefix | Mean (µs) | Increase over previous prefix (µs) |
|---|---:|---:|
| Padding | 11.5 | — |
| + QA/KVA projections and QA normalization | 355.6 | 344.1 |
| + KV epilogue and QB | 535.4 | 179.7 |
| + WKB, no Q output epilogue | 656.4 | 121.0 |
| Recent full-kernel controls | ~830 | ~174 vs. no-Q prefix |

Removing code changes allocation, synchronization, and overlap. Differences are useful for prioritization, not direct stage timings or fully recoverable overhead. Baseline `PipeUtilization` reported **35.9% MAC, 48.2% MTE2, 29.8% Fixpipe** on AICs; those activities overlap and cannot be added. Nonzero AIV ratios were reported alongside zero AIV times, so the latter were not used to infer no vector work.

## Optimization history

### T=2048: initial improvements

| Change | Purpose | Measured latency (µs) |
|---|---|---:|
| Pinned original | Starting reference | 874.0 |
| Replace scalar memory broadcasts with register `vdup` | Avoid scalar store/reload round trips | 858.5 |
| One horizontal Q max reduction per row | Reduce repeated reductions | 854.3 |
| WKB 32 → 64 token rows per tile | Reduce tile overhead; keep A live across four N tiles | 842.1, added to 854.3 control |
| Assign complete QB heads to their WKB cores | Remove global QB rendezvous; retain required local visibility | 835.7, added separately to 854.3 control |
| Combine the wins | 32-AIC experimental 8-row reference | 822.0–822.3 |

### T=2048: Q output and overlap

| Change | Comparison | Outcome |
|---|---|---|
| Corrected 16-row Q epilogue | 788.369 µs (786.686 / 790.051) vs. 830.199 µs 8-row | **5.04% lower**, shared NPU |
| Start Q after 1,024 tokens of each head | 762.781 µs (766.001 / 759.560) vs. 789.687 µs 16-row interleaved control | **3.41% lower**, shared NPU |
| Start Q after 512 tokens, four chunks per head | 735.595 µs (737.435 / 733.754) vs. 749.008 µs two-chunk mean | **1.79% lower**, shared NPU |
| Gate four-chunk specialization to tested shape and core count | 735.454 µs vs. 752.345 µs two-chunk control | **2.24% lower**, otherwise idle NPU |

The first batch-size generator accidentally changed the `1e-8` Q absmax floor when replacing standalone 8s. Its timings are historical only. The reported 16-row gain above is from the corrected rerun with the original floor restored.

### T=128: prior work, separate scope

The T=128 baseline was 69.10 µs. Register broadcasts brought it to about 65.7 µs; whole-head QB ownership and local handoff to 62.33 µs; QA/KVA overlap and WKB M64 to 60.85 µs; paired input reuse to 58.81 µs. The best tested combination paired the last two KVA tiles and used one Q row-max reduction, reaching **56.80 µs** (56.709 / 56.891). These are T=128 results, not gains to add to T=2048. The final T=128 snapshot is `sources/mla_best.py`.

## Experiments not adopted

| Experiment | Result |
|---|---|
| T=2048 simple QA/KVA overlap | 823.6 vs. 822.0 µs control; no improvement. |
| T=2048 front paired input reuse | Typed-region/compiler issues; static variants exceeded L1 or L0C limits. No valid timing. |
| WKB M128 | Timed out; not a latency comparison. |
| Reuse WKB B across M tiles | 839.5–842.6 vs. 823.1 µs; extra A traffic/overlap changes may matter, but cause not isolated. |
| QA/KV four-chain RMS variants | Correctness passed. Apparent 0.3–0.4% gains were within shared-card variation. |
| Full-head QB B residency | 865.6 µs M128 and 922.2 µs M64 vs. 828.7–830.1 µs; both slower. |
| 32-row Q epilogue | 1,472.2 µs in the original batch experiment. Corrected 32-row transfers split into two 16-row vector blocks also regressed, to 1,484.1 µs. |
| Separate WKB buffers for early readiness | L0B allocation failed (98,304 bytes requested; 65,536 limit). One shared buffer set fixed it. |
| Direct WKB-to-AIV transfer in earlier T=128 experiments | BF16 transport rejected by the compiler; FP32 variants hit compiler issues or poison/recovery NaNs. No valid speed. |
| Other T=128 trials | L2 bypass, smaller front K, larger WKB tiles, per-tile signals, separate UB arenas, and dedicated Q AIV showed regressions or no clear gain. |

## Next experiments, in priority order

1. **Measure the front-stage critical path, then pipeline it.** Separate QA, KVA, and QA normalization using matched prefixes or small diagnostic counters, including per-core imbalance. If the full QA barrier is visible, try ready-by-token-chunk normalization while the next projection chunk runs. Keep accumulation order, double buffering, and the KVA schedule fixed for the first test. Pass the poison/recovery check before timing.
2. **Test whether four chunks are near the overlap limit.** Compare 512-token chunks against an intermediate schedule with fewer early signals, or a smaller first chunk and larger later chunks. Verify event-ID bounds and flag lifetime first; do not assume more flags fit. Keep the 16-row epilogue and global layout fixed.
3. **Profile Q transfers before changing the ABI.** Compare head-wise NoPE loads, strided RoPE loads, and Q/scale stores with matched diagnostics. Try compatible local packing or store coalescing while keeping output order. A head-major external layout is only worth testing with a compatible consumer and an end-to-end benchmark.
4. **Reduce intermediate WKB traffic only if measured to be limiting.** `q_latent_h` is 192 MiB at this shape, so a full write/read is up to 384 MiB of logical traffic, not necessarily HBM traffic. Start with one bounded chunk and explicitly prove visibility and buffer reuse. Earlier direct transfer failed; preserve the validated path until the new one passes recovery.
5. **Qualify the gated patch for integration.** Repeat on idle cards and multiple seeds, include near-zero Q values and shape boundaries (2047/2049), check BF16 and non-32-AIC fallback where supported, and confirm graph/AOT behavior before enabling an exported specialization. Measure operator and end-to-end serving separately.

For any new change, interleave control/variant runs on the same idle card, require the expected kernel symbol and a correctness marker, report run variation and memory, and keep it only if the gain repeats. Current full-kernel measurements are device task times, not total application latency.

## Folder contents and reproduction

- [`patches/quant_mla_prolog_t2048_gated.patch`](patches/quant_mla_prolog_t2048_gated.patch): latest candidate against the bundled **vdup baseline**, preserving generic fallback.
- [`patches/quant_mla_prolog_t2048.patch`](patches/quant_mla_prolog_t2048.patch): earlier ~822 µs experimental prefill, against the **pinned upstream** source. It is not a prerequisite for the gated patch.
- `sources/`: baseline and winning standalone snapshots. `mla2048_wkbquartersgatedv2.py` is the exact gated source used for confirmation; `mla2048_wkbquarter.py` is the ungated experiment, not a drop-in replacement.
- `generators/`: transformations for accepted and rejected experiments. `history/` contains earlier source snapshots, original remote-oriented runners, and the [detailed experiment log](history/experiment-notes.md). Historical runners reference their original `/workspace` layout and should not be run blindly.
- `reference/`: source and correctness files from CANNBotDSL commit `1342a261519dead926fff56c49400cdea72e42bc`. The full checkout is required to run the tests.
- `benchmarks/`: portable entry point and T=2048 harness; the stage harness is diagnostic only. `profiles/` contains copied profiler summaries and run logs, not full trace databases or runtime dependencies.

### Where to find the key evidence

| Experiment | Folder under `profiles/` |
|---|---|
| Earlier ~822 µs prefill reference | `msprof-prefill-port.ruRngW/` |
| Pipe and memory counters | `mla-metrics.2Mnans/` |
| Cumulative stage diagnostics | `mla-stage-diagnostics.eigclX/` |
| Corrected 16 rows and split 32 rows | `qbatch-shared.hnackx/` |
| Two-chunk test and repeat | `qbatch-shared.8tdKpL/`, `qbatch-shared.dUyIh4/` |
| Four-chunk comparison | `qbatch-shared.IiqKNY/` |
| Gated confirmation | `qbatch-shared.6WWyJf/` |

Use an isolated Ascend 950 environment with the pinned repository available, CANN 9.2, Python 3.12, `torch_npu`, and CANNBotDSL 0.7.0. Expose only the reserved NPU as logical device 0 before starting. Pass the card ID shown by `npu-smi` in that environment for its idle check. The check is not a reservation. The script builds a fresh copy from the pinned commit for each run and does not edit your checkout.

```sh
bash benchmarks/run_benchmark.sh /path/to/cannbot-dsl /path/to/new-results-dir 6 \
  half gated half gated
```

Each valid run must pass golden precision and poison/recovery checks and produce **47** matching tasks (1 initial, 6 recovery, 10 warmup, 30 measured). The runner reports the mean of the final 30 device durations. A profiler exit code of 0 alone is not success. Peak PyTorch allocation for the tested variants was 519,084,544 bytes (~495 MiB); it does not include driver/runtime allocations.

The portable runner was syntax-checked locally, and its timing parser was checked against the copied confirmation profiles. The NPU measurements used the original container runner in `history/`.

To regenerate the gated source from the bundled inputs without touching the repository:

```sh
python3 generators/mla2048_isolate.py \
  sources/vdup_baseline.py sources/mla2048_wkbquarter.py /path/to/new_candidate.py
```
