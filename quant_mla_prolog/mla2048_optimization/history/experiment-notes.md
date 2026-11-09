# T=2048 experiment status

The validated 32-AIC specialization remains `quant_mla_prolog_t2048.patch`.
Previous repeated measurements were approximately 822 microseconds. This is
not a general production dispatch: its QB ownership assumes 32 AIC cores.

## Next candidate: four-chain RMS reduction

`mla2048_rms_variants.py` transforms a copy of the validated source into
QA-only, KV-only, or combined variants. It replaces a horizontal reduction
for every 64-element segment with four independent vector sum chains and
one final horizontal reduction. It keeps sqrt/division, epsilon, gamma
rounding, output conversion, and cache layout unchanged. It does not use
FMA, so fused multiply-add rounding is not an additional variable here.

The FP32 addition order changes. Syntax checks are not correctness checks.
Each variant must pass the existing T=2048 CPU golden and poisoned-workspace
recovery tests. Then measure with msprof, bracketed by the unchanged control.
Require the success marker and 47 matching kernel records; use the last 30
records for steady-state timing. A timeout or missing marker is not a timing.

At the latest occupancy check, all eight cards on server 03 had about 93 GB
HBM allocated. All eight cards on server 07 were at 100% utilization and
similarly allocated. No hardware benchmark was started for these variants.

All three variants completed `_compile_o11_native_variant(0, 1)` successfully
on server 03 in container `attention_testing_950_rms4_compile` (exit 0).
The container had no NPU devices, two CPUs and a 4 GB memory limit. Logs are
in `/root/attention_testing_950_workspace/rms4-offline.o336Js/` on server 03.
Device-query warnings are expected in this device-free container. Compilation
does not establish numerical correctness or runtime performance.

Example generation (output must not already exist):

```sh
python3 tools/mla2048_rms_variants.py BASELINE.py EXPERIMENT.py --scope qa
```

Do not replace the validated implementation until correctness and repeated
performance measurements show a benefit on an idle device.

## Shared-card measurements

User subsequently authorized testing in the remaining memory of server 03
NPU 6. The baseline probe and all six variant runs passed the T=2048 golden
and poisoned-workspace recovery tests. PyTorch peak allocated memory was
519,084,544 bytes, with a 2 GiB allocator cap and a 3.5 GiB free-memory guard.
Runtime/driver allocations are outside that cap. An existing serving worker
remained on the card; these are NOT exclusive-device performance results.

Results in microseconds, final 30 of 47 matching msprof records per run:

| Variant | First | Repeat |
|---|---:|---:|
| Unchanged control | 835.334 | 831.693 |
| QA four-chain | 831.427 | 830.349 |
| KV four-chain | 828.714 | 832.195 |
| Both | 833.016 | 828.255 |

Average variant gains against the average control are only about 0.3–0.4%,
within observed run-to-run variation. No convincing performance winner and
no change to the validated implementation. Remote artifacts:
`/root/attention_testing_950_workspace/rms4-shared.f0kEN0/`.

## QB full-head weight residency

Generator: `mla2048_qb_resident.py`. This replaces the repeated T128 QB calls
with head-outer/token-tile-inner loops, keeping the full 1536x192 FP8 B tile
and its scales resident in L1. It keeps the original K128 accumulation order
and double-buffered L0 channels; full-head B uses one L1 slot.

Both M128 and M64 versions compiled and passed T2048 golden/recovery tests
on shared server 03 NPU 6, with 47 msprof kernel records each. Results:

| Variant | Mean final 30 tasks (us) |
|---|---:|
| Opening control | 828.658 |
| Weight-resident M128 | 865.584 |
| Weight-resident M64 | 922.232 |
| Closing control | 830.088 |

Regressions versus mean control: about 4.4% and 11.2%. No adoption.
Fewer B transfers did not outweigh the other schedule changes; extra A
reloads and reduced overlap are hypotheses, not established profiler causes.
Artifacts: `/root/attention_testing_950_workspace/qbres-shared.PCTDZU/`.

## Baseline hardware-counter passes

Two app-scoped msprof passes (`PipeUtilization`, `Memory`) completed with
golden/recovery success, 47 records each, and container exit 0. Shared NPU 6;
not exclusive-device evidence. Artifacts:
`/root/attention_testing_950_workspace/mla-metrics.2Mnans/`.

Final-30 averages: pipe pass 833.4325 us, memory pass 827.1288 us.
AIC MAC ratio 35.85%, MTE1 26.64%, MTE2 48.19%, Fixpipe 29.83%.
AIV vector ratio 20.44%, MTE2 32.34%, MTE3 26.11%. Reported AIV times
were zero despite nonzero ratios, so do not interpret those zero times as
no vector work. Engine ratios overlap and are not additive runtime shares.
These are whole-kernel counters, not phase-local bottleneck attribution.
Next diagnostic: isolate front/QB/WKB/epilogue phases before choosing which
transfer or synchronization to restructure. No proven cause established.

## Cumulative diagnostic kernels

`mla2048_stage_diagnostics.py` generates pad, front, QB and WKB prefixes.
These intentionally do not implement the full operator. The harness first
runs full-baseline golden/recovery checks, then launches each partial kernel
10 warmups + 30 measured tasks. Require 40 diagnostic-name records and the
diagnostic completion marker. Unique Torch namespaces avoid collisions with
the baseline. Initial attempt before that fix had a registration error and
must not be used.

Artifacts after the fix:
`/root/attention_testing_950_workspace/mla-stage-diagnostics.eigclX/`.
Retrieved results: pad 11.5122 us; front 355.6286 us (40 records each).
Front includes QA/KVA projections and QA normalization. QB-prefix completion
was confirmed before SSH banner timeouts. After reconnection, all four
diagnostics were confirmed complete, 40 records each, container exit 0:

| Cumulative prefix | Mean final 30 tasks (us) |
|---|---:|
| Pad | 11.5122 |
| Front | 355.6286 |
| Front + KV epilogue + QB | 535.3747 |
| Above + WKB, no Q epilogue | 656.4030 |

Against recent full-kernel controls around 830 us, the no-Q-epilogue prefix
leaves an approximately 174 us difference. This is not a direct isolated
Q-epilogue measurement: prefix subtraction changes allocation, sync and
overlap. It motivates measuring Q batch size, strided stores and per-head
readiness before attempting another GEMM rewrite.

## Q output epilogue batch size

`mla2048_qbatch.py` generates experimental 16- and 32-row epilogues from the
validated 8-row source. Only the T=2048, 32-AIC FP8 path was tested; the
generator also changes a shared helper, so do not deploy its output as a
general operator replacement without isolating the specialization.

Shared server 03 NPU 6, mean of final 30 of 47 matching msprof records:

| Run | Mean (us) |
|---|---:|
| Opening 8-row control | 825.5725 |
| 16 rows | 786.2651 |
| 32 rows | 1472.1848 |
| Closing 8-row control | 838.9063 |
| 16 rows, confirmation 1 | 784.6399 |
| Interleaved 8-row control | 831.6479 |
| 16 rows, confirmation 2 | 784.5732 |

All individual runs passed golden and poisoned-workspace recovery checks.
The first runner encountered a shell error after its last measured run
because its script was overwritten while active. The independent confirmation
container completed normally, exit 0. Confirmed 16-row mean 784.6066 us,
5.66% lower latency against the interleaved control. The 32-row variant
regressed and is rejected. Peak PyTorch allocation remained 519,084,544 bytes.
No production kernel was modified. The next validation is to isolate the
16-row specialization and reproduce the gain on an exclusive NPU.

Remote artifacts under `/root/attention_testing_950_workspace/`:
`qbatch-shared.0W02xF/` and `qbatch-shared.NIXNZX/`.

### Corrected floor and split transfer experiment

The original generator inadvertently changed the `1e-8` minimum Q absmax
when replacing standalone 8s. Its measurements above are historical, not a
pure batch-size comparison. The corrected generator preserves the floor.
`--split` uses 32-row transfers with two 16-row vector blocks, holding all
other quantization operations fixed.

On shared server 03 NPU 6, the 8-row control measured 830.199 us, corrected
16-row controls measured 786.686 and 790.051 us, and the split variant
measured 1484.074 us. All passed golden/recovery, had 47 matching profiler
records, and reported the same peak PyTorch allocation. The container
exited 0. Use the corrected 16-row mean (788.369 us, 5.04% below the 8-row
control) as the experimental reference. Reject the split variant. The
experiment does not identify the precise cause of the 32-row slowdown.
Artifacts: `/root/attention_testing_950_workspace/qbatch-shared.hnackx/`.

### Two-chunk WKB-to-Q readiness

`mla2048_wkb_handoff.py` applies the experiment to the corrected 16-row
source. The 32-AIC FP8 path signals after 1024 of 2048 tokens per head and
again at head completion. Q quantizes the first half while WKB produces
the second. No layout or quantization math change. The initial version
with two WKB instances exceeded L0B capacity (98304 vs 65536 bytes); the
successful version uses a single buffer set and keeps double buffering.

Shared server 03 NPU 6, final 30 of 47 msprof records per valid run:
16-row controls 785.235 and 781.633 us; revised handoff 766.001 us;
interleaved 16-row control 789.687 us; handoff repeat 759.560 us.
Both handoff runs passed golden/recovery and exited 0; peak PyTorch
allocation remained 519,084,544 bytes. Mean 762.781 us, 3.41% lower than
the interleaved control (2.89% lower than the mean of all three controls).
Keep as the fastest tested experiment, not a production replacement.
Artifacts: `qbatch-shared.gqKq1R/`, `qbatch-shared.8tdKpL/`, and
`qbatch-shared.dUyIh4/` under `/root/attention_testing_950_workspace/`.

### Four-chunk WKB-to-Q readiness

`mla2048_wkb_quarters.py` extends the two-chunk experiment to four
512-token chunks per head, keeping the corrected 16-row epilogue and one
WKB buffer set. It uses separate readiness flags for the three early
boundaries; final completion retains the original per-head flags.
This is for T=2048 FP8 on 32 AICs only, not general dispatch.

Interleaved shared-NPU measurements on server 03 NPU 6: two-chunk controls
748.715 and 749.301 us; four-chunk 737.435 and 733.754 us. Both variants
passed golden/recovery on both runs with 47 matching msprof records each;
batch exit 0, unchanged PyTorch peak allocation (519,084,544 bytes).
Four-chunk mean 735.595 us is 1.79% lower than the same-batch two-chunk
mean 749.008 us. Do not compare across sessions as if load were unchanged.
Artifacts: `/root/attention_testing_950_workspace/qbatch-shared.IiqKNY/`.

### Isolated T=2048 runtime specialization

`mla2048_isolate.py` generates `quant_mla_prolog_t2048_gated.patch` against
the local vdup source. Existing kernel classes are unchanged (AST check).
Runtime dispatch selects the additional profile only for exact T=2048,
FP8 output, and 32 AICs; all other profiles and the AOT export list are
unchanged. AOT use of the new profile was not tested. `git apply --check`
passed; the patch was not applied to the working operator or deployed.

Server 03 NPU 6 had no listed competing processes at launch and during
sampled checks. Both gated runs passed golden/recovery and reported 47
matching records of the `KernelT2048` symbol. Final-30 means: 735.101 and
735.807 us versus interleaved two-chunk control 752.345 us (2.24% lower
mean latency). Earlier 8-row control on the same idle card: 822.195 us.
Peak PyTorch allocation remained 519,084,544 bytes. The three existing
small-token FP8 network correctness cases passed against the gated source.
The first generated candidate omitted a class decorator and did not run;
only the repaired candidate is reflected in the patch and valid results.
Artifacts: `/root/attention_testing_950_workspace/qbatch-shared.6WWyJf/`.
