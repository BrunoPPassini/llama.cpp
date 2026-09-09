# Engineering a 256K hybrid-KV runtime for Qwen3.8-27B on a 16 GB RTX 5070 Ti

This is an engineering record of a model-specific `llama.cpp` runtime developed for Qwen3.8-27B UD-IQ4_XS on one GeForce RTX 5070 Ti. The work is not a generic claim that host KV is fast or that every Blackwell GPU will reproduce these numbers. It documents the architecture, memory accounting, state machines, failed experiments, profiler evidence, and measured progression that led from a fast 64K-resident runtime to a 256K-capable hot/cold KV ring.

## Publication status

The `qwen38-blackwell-256k` branch contains the complete derived CUDA/runtime source used for continued development of this experiment. It is a compilable source snapshot based on upstream commit `d775b8967a46d8beb110d444aa3b8938179e0dd8`; the custom environment variables and server behavior described below are implemented in this tree. Model weights and generated binaries are intentionally excluded.

Two identities are kept separate for auditability. The frozen DLL hashes below identify the exact binaries used for the reported final measurements. This branch is a later development snapshot that contains those mechanisms plus subsequent integration and stability work, so rebuilding it is not claimed to reproduce the frozen DLL hashes byte-for-byte. Rebuild and benchmark the source commit you use, and record that commit together with the resulting binary hashes.

No model weights, private repositories, credentials, VPN details, application source, or user data are part of this record.

## Frozen system

| Component | Measured or frozen value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, Blackwell, 16,303 MiB reported by `nvidia-smi` |
| Driver used for the final audit | 591.86 |
| PCIe link | Gen5 x16 maximum and Gen5 x16 current under the audit |
| GPU memory | 16 GB GDDR7, 896 GB/s vendor specification |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| Host memory | 48 GiB, two 24 GiB DDR5-6400 DIMMs |
| Host memory peak theoretical rate | 102.4 GB/s for two 64-bit DDR5-6400 channels, before controller/protocol losses |
| PCIe 5.0 x16 raw payload ceiling | Approximately 63 GB/s in each direction, before transaction and software overhead |
| CUDA toolkit | 13.3, nvcc 13.3.73 |
| Generator | Visual Studio 17 2022, x64 |
| CUDA target | `120a-real` |

The bandwidth hierarchy matters. Local GDDR7 is roughly 14.2 times faster than the raw PCIe 5.0 x16 ceiling, and actual DMA throughput is lower than the raw link number. Fast DDR5 prevents host memory from becoming an even earlier bottleneck, but it cannot make a discrete GPU read host-resident KV at VRAM speed. The design therefore treats RAM as capacity and uses VRAM as a working set, rather than treating mapped host memory as slow substitute VRAM.

The local PCIe state above was queried directly:

```text
NVIDIA GeForce RTX 5070 Ti, driver 591.86, 16303 MiB,
PCIe max Gen5 x16, current Gen5 x16
```

## Hardware and model compatibility scope

This runtime was developed, profiled, and quality-gated only with the exact `Qwen3.8-27B-UD-IQ4_XS.gguf` artifact identified below on an RTX 5070 Ti 16 GB. It is not a generic Qwen runtime and it is not validated for other model families, Qwen sizes, quantizations, tensor layouts, MTP heads, or chat templates. A different model may reject the specialized path, fall back to ordinary kernels, fail its memory fit, or produce incorrect behavior if a model-specific invariant is accidentally violated. Treat every other model as unsupported until its tensor shapes, state layout, token behavior, and executable quality suite pass independently.

The source was compiled for CUDA target `120a-real`. Two other 16 GB Blackwell cards are plausible ports, but are not measured results from this study:

| GPU | Why it is a candidate | Expected limitation | Validation status |
|---|---|---|---|
| RTX 5070 Ti 16 GB | Development machine; 16 GB GDDR7 and PCIe 5.0 x16 | Display and driver reservations leave less than the nominal 16 GB available | Tested |
| RTX 5080 16 GB | Same Blackwell generation, 16 GB capacity, and a higher compute/bandwidth ceiling | Exact VRAM margin and scheduling still require a fresh build and benchmark | Not tested |
| RTX 5060 Ti 16 GB | Blackwell architecture and the same nominal memory capacity | Substantially lower memory bandwidth means the profile may fit but should be slower | Not tested |

The 8 GB RTX 5060 Ti is not a candidate for this exact all-weights-resident profile. Even on the 16 GB cards, do not copy measured RTX 5070 Ti throughput into a compatibility claim: rebuild for the installed GPU, verify that every intended layer remains on device, record usable VRAM after display allocation, and rerun token-identity plus repository tests.

## Frozen model and runtime identity

| Artifact | Size | SHA-256 |
|---|---:|---|
| `Qwen3.8-27B-UD-IQ4_XS.gguf` | 14,252,845,984 bytes | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| `llama-server.exe` wrapper | 10,752 bytes | `3515fe70c89f0525559f455ec0ce234c1b4221d1cd2a2e98d4171af932f78ffa` |
| `llama-server-impl.dll` | 14,229,504 bytes | `1dcff95776229c10feee690a6c1b0edb439a7d541ba1cc8e8b46ca2927340739` |
| `llama-common.dll` | 9,700,352 bytes | `8f75ff00be0b84ea2703dabc0e093b2d784d41029a554fce2ce94eef15a0778d` |
| `llama.dll` | 4,129,792 bytes | `8da2169163546b8069542e1e0e1e36d91118703adcdf0acb8880b1b6053de0f4` |
| `ggml.dll` | 67,072 bytes | `efb34baaba029f06624251b457ff2b203e298be9bfa30731dfd150c440161757` |
| `ggml-base.dll` | 680,960 bytes | `c9da90dddbc464ac9f4e5c625f61a0937ed5ed5ac19d3ef6dd2f3177255dafb7` |
| `ggml-cpu.dll` | 1,008,640 bytes | `7fd1f430860da183e90c74c48e44a84d1e8984843f747ebc354b1c9bcdd489ae` |
| `ggml-cuda.dll` | 54,412,800 bytes | `1c26d70a33e9996cbb7646328ea4c3a67b27a221432f7fae468112eecb860242` |
| `mtmd.dll` | 2,361,856 bytes | `b6a43402c897a5390c2854b6371d538fab32dc1df387de4c8a0c1d372518c150` |
| Unsloth chat template | 9,993 bytes | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |

The server EXE is only a loader. Reproducing a run requires the whole DLL set, not merely the EXE hash. The frozen build identifier is:

```text
b5e8b02f4182b42aba03def31e39437c729bd3f7-derived-
deterministic-mtp-blackwell-fa4-stability-recurrent-prefix-20260904
```

## Why this model admits a specialized design

The model is hybrid rather than a conventional 64-layer full-attention dense transformer:

- 64 backbone layers plus one NextN/MTP layer;
- embedding width 5,120;
- 24 query heads and 4 KV heads;
- relevant Flash Attention shape `DKQ=256`, `DV=256`, with a 128-value Q4 conversion sub-row in the ring path;
- 48 recurrent DeltaNet/GDN layers;
- 16 full-attention layers, in a 3 recurrent to 1 attention pattern.

Only the 16 full-attention layers need token-indexed K/V history. The 48 recurrent layers retain fixed-size state. This is the central opportunity: a 256K logical context does not require token-indexed KV for all 64 layers. The runtime is guarded by an exact architecture gate and refuses to enable this path for a different layer pattern, head shape, quantization, or CUDA target.

It is also the central correctness hazard. A recurrent GDN state summarizes history and does not have a general inverse. A generic KV shift can remove old attention cells, but cannot subtract arbitrary old tokens from the GDN state. Exact compaction therefore clears state and canonically replays the retained prefix. Treating this model as an ordinary transformer produced crashes or silent state errors in early prototypes.

## Baseline memory decomposition

At a 64K configuration before the later state compression, the main allocations were measured as follows:

| Allocation | Approximate size |
|---|---:|
| Quantized model weights | 13,061.10 MiB |
| Target KV | 1,152.00 MiB |
| Target plus MTP recurrent snapshots | 598.50 MiB |
| Shared target/MTP compute arena | 400.28 MiB |
| Draft KV | 72.00 MiB |

The recurrent 598.50 MiB consisted of about 22.50 MiB of R state and 576 MiB of S state across four planes: one active plane plus three speculative snapshots. This was the first large avoidable duplication. The KV itself grows by about 18 MiB per additional 1K target+draft tokens in the measured profile, with another roughly 4 MiB/1K of deep-context compute residency near the dense limit.

## Memory work: measured progression

The project did not jump directly to a ring. Each memory change was isolated and validated before its capacity was reinvested.

| Stage | Architectural change | Measured memory effect | Performance/correctness result |
|---|---|---:|---|
| Shared compute arena | Target and MTP use one mutually exclusive compute arena | Avoided a second roughly 400-460 MiB arena | No intended math change |
| Recurrent transaction log | Replace four S planes with base/work state plus exact per-token GDN operands | 598.50 -> 319.54 MiB, saving 278.96 MiB | 1,676.74/91.79 -> 1,691.19/91.21 pp/tg; effectively neutral; exact validation passed |
| Recurrent phase arena | Speculative forward no longer mutates persistent S; replay only accepted updates | 319.54 -> 175.54 MiB, saving another 144.00 MiB | Enabled 88,064 context; controlled output hash matched |
| Sparse compute VMM | Reserve a stable virtual range but commit physical GPU pages only for the largest graph actually seen | +336 MiB free when ready; +26 MiB near 80K full use | 80,656-token A/B was 1,118.33/61.66 vs dense 1,128.04/61.44 pp/tg |
| Sparse compute trim | Unmap whole compute mappings after the recent high-water mark drops | Recovered 274 MiB after long -> short | 99 -> 373 MiB free, vs 375 MiB in initial short state |
| Sparse KV VMM | Reserve the logical KV address range; map 8,192 initial rows and grow in 4,096-token chunks | Short/medium sessions pay for resident rows, not maximum window | 80,656 -> short -> 80,656 repeated without changing output hashes |
| Bidirectional KV trim | Unmap cold KV above occupied high-water plus incoming headroom | Recovered 1,626 MiB total: 1,428 MiB cold KV plus transient compute | First and second 80,656-token runs stayed around 60-62 tg |

The transaction log stores the exact per-token GDN operands needed for accepted replay: decay, expanded key, and delta. On full rejection the base state remains valid; on partial acceptance only the accepted prefix is replayed. The phase arena takes the idea further: speculative execution uses an ephemeral phase and commits accepted state exactly once. These are state-lifetime changes, not lower precision approximations.

Sparse VMM is also a lifetime optimization, not compression. The virtual address remains stable for CUDA graphs while physical pages are committed and unmapped in large units. An initial allocate-copy-free prototype failed near 41K because growing the buffer needed both old and new allocations at once. The VMM version removed that peak. A content epoch is required for all layer-major caches because remapping can preserve a virtual address while changing the data behind it.

## The hot/cold KV ring

### Data placement

The production experiment exposes one logical 262,144-token KV address space. It keeps a 65,536-token hot prefix device-resident and allows the older/cold tail to live in page-locked host memory. Q4_0 is used for target K/V and draft K/V. The ring uses fixed 8,192-token device staging tiles.

```text
logical KV: [0 ................................................ 262143]
             |----------- hot 65,536 -----------|--- cold tail ---|
physical:    |       persistent GPU pages       | pinned DDR5 RAM |

device ring slots:
  slot 0: H2D copy of cold tile N+2
  slot 1: Q4 dequantization of tile N+1
  slot 2: Flash Attention/MMA consumes tile N
```

Below 65,536 effective tokens the stock GPU path is used. The cold ring is not allowed to tax the common short-context case. At 42,109 effective tokens, the dual-path implementation measured 1,437.78 pp/s and 72.10 tg/s versus 1,440.37 and 72.28 for the stock control. The deltas, -0.18% prefill and -0.24% decode, are effectively the cost of the branch and capacity plumbing.

### Why the ring is stateful

Exact softmax cannot independently normalize each tile and average the results. For a query and a stream of score tiles, the ring carries the online-softmax state:

```text
m_new = max(m_old, max(scores_tile))
alpha = exp(m_old - m_new)
p     = exp(scores_tile - m_new)
l_new = alpha * l_old + sum(p)
O_new = alpha * O_old + sum(p * V_tile)
output = O_final / l_final
```

The state is `(m, l, O)`: running maximum, normalized denominator, and unnormalized value accumulator. It is small with respect to KV and survives while each 8,192-token K/V tile is copied, transformed, consumed, and discarded. Mask coordinates and absolute token positions remain global; the ring must not renumber a cold tile as if it began at zero.

The P8 path partitions the K/V axis into eight stable, disjoint ranges. Each partition keeps an independent `(m_p, l_p, O_p)` and runs in parallel. The final reduction is another log-sum-exp merge:

```text
m = max_p(m_p)
l = sum_p(exp(m_p - m) * l_p)
O = sum_p(exp(m_p - m) * O_p)
output = O / l
```

P16 was slower. It increased reduction and resource overhead enough to fall to 32.08 tg/s, whereas P8 balanced concurrency, occupancy, and staging cost.

### Three-stage pipeline

The first ring serialized copy, Q4 conversion, and attention. The final decode path has persistent CUDA streams and three rotating slots coordinated by events:

```text
FREE(epoch)
  -> COPYING(epoch)
  -> READY(epoch)
  -> CONSUMING(epoch)
  -> FREE(epoch + ring_size)
```

`copy_ready`, `ready`, and `consumed` events enforce ownership without a global `cudaDeviceSynchronize`. While the main stream consumes tile N, a transform stream expands Q4 K/V for N+1 and a copy stream DMA-transfers N+2 from pinned DDR5. The goal is not to create fictitious extra PCIe channels. There is still one physical x16 link. The gain comes from keeping the copy engine and compute stages occupied concurrently rather than paying their latency serially.

Direct-hot bypasses an old GPU-to-GPU copy: a segment already resident in the 65,536-token VMM prefix is dequantized from its original device address. Only a cold segment traverses PCIe and staging.

K and V Q4 conversion is issued in one CUDA grid. The model's useful Q4 row requires only 64 working threads in this conversion, while the generic launch used 256 threads and 192 returned without work. The accepted narrow-block kernel uses 64 threads but retains the same element order and conversion expression.

Prefill uses a two-slot variant. The next cold tile is copied and converted while the current tile is consumed. The third decode slot was not useful for large-Q prefill and would reserve more memory. At a real 256,257-token prompt, this prefill pipeline raised 658.24 to 698.84 pp/s and cut prompt time by about 22.6 seconds.

### Why bulk staging beat zero-copy

A tempting design was to let the attention/dequant kernel read mapped DDR5 directly. It was functionally valid in the tested sample but decode collapsed to 18.04 tg/s and GPU power fell to about 132 W. The GPU was starved by fine-grained PCIe reads. The accepted path performs large asynchronous H2D copies into VRAM, then lets CUDA cores/MMA consume local data. This is exactly the distinction between using PCIe as a DMA transport and using PCIe as load/store memory.

The link was confirmed at Gen5 x16 under load. That does not make it equivalent to 896 GB/s GDDR7. PCIe 5.0 x16 has approximately 63 GB/s of raw bandwidth per direction, and protocol, DMA, Windows/WDDM, and access-pattern overhead reduce the attainable payload rate. The study did not record a defensible end-to-end GB/s counter for the entire request, so it does not invent one. The measured direct-mapped collapse and the long-context throughput curve are the empirical bandwidth evidence.

## Ring performance progression

These rows use a real 87,160-token prompt, a 262,144-token allocated window, 64 output tokens, batch/ubatch 1024/512, and the same Q4 KV profile unless noted.

| Engine step | Decode tg/s | Incremental effect | Result identity |
|---|---:|---:|---|
| Serial stateful MMA ring, P1 | 23.3278 | Baseline | Reference for this table |
| Eight attention partitions, P8 | 34.9182 mean | +49.69% | Stable across two loads |
| P8 plus direct-hot | 35.7581 mean | +2.41% vs P8 | Same hash as P8 |
| P8 plus double pipeline and fused K/V conversion | 36.8441 | +3.04% vs direct-hot | Same hash |
| P8 plus triple pipeline and fused K/V | 40.2360 mean | +9.21% vs double pipeline | Same hash |
| Previous plus 64-thread Q4 block | 41.5319 mean | +3.22% | Same hash and MTP counters in 3 clean processes |
| Direct host-mapped Q4 reads | 18.0364 | -56.57% vs accepted final | Rejected: PCIe starvation |

The accepted final range was 41.3455 to 41.8282 tg/s across three fresh processes. Mean prefill was 1,104.85 pp/s and was effectively unchanged by the decode work. Relative to P1, the combined decode gain was 78.03%.

With a 512-token output at the same 87K prompt, the final profile measured about 47.62 tg/s. That larger number is not a hidden kernel optimization: MTP acceptance rose from 39/72 for the short 64-token completion to 337/519 for the longer generation. Sustained agentic output is a useful workload, but it must not be compared to the 64-token rows as if output length and speculative trajectory were controlled.

## Production and historical speed points

There is no single honest "speed of the model" number. Throughput changes with effective prompt length, output length, MTP acceptance, deterministic reduction geometry, and whether the cold ring is active. The following points are retained to make those regimes explicit:

| Runtime/protocol | Effective prompt + output | Prefill pp/s | Decode tg/s | Status |
|---|---:|---:|---:|---|
| Normalized all-GPU IQ4_XS MTP3, 3 repetitions | 7,326 + 512 | 1,760.90-1,765.15 | 92.25-92.73 | Fast historical ceiling; 67.99% aggregate MTP acceptance |
| Deterministic fixed-shape MTP3 | Short + controlled output | About 1,745 | 82.34 | Exact target/MTP path; promoted principle |
| Approved frozen stability runtime, 3 repetitions | 7,326 + 512 | 1,439.742 mean | 71.164 mean | Daily reference for the fully integrated binary |
| Experimental shape-gated XOR swizzle | 7,326 + 512 | 1,458.238 mean | 73.802 mean | Same sampled output, but not promoted after a fresh-load stall |
| Final cold ring | 87,160 + 512 | About 1,110.9 | About 47.62 | Cold tail active; sustained MTP workload |
| Final cold ring | 256,257 + 64 | 698.84 | 23.968 | Maximum filled-context demonstration |

The normalized 92.55 tg/s result is not substituted for the 71.16 tg/s frozen daily result. They came from different runtime generations and correctness constraints. The table intentionally preserves that distinction instead of publishing only the largest number.

The quantization selection study also tested higher-storage profiles. In the 64K-allocated comparison, UD-IQ4_XS was the only candidate within 10% of the fastest normalized decode. NVFP4-LOW averaged about 40.76 tg/s, NVFP4-MEDIUM peaked near 33.87 tg/s, and Q4_K_S reached 38.58 tg/s only in a more favorable 12K/offload condition. Those rows were placement-limited and are historical engineering evidence, not a causal quality ranking.

## Filled-context scaling

The table below is real processed context, not merely configured capacity.

| Real prompt | Output | Engine | Prefill pp/s | Decode tg/s | Final free VRAM |
|---:|---:|---|---:|---:|---:|
| 87,160 | 64 | Initial one-slot stateful ring | 1,237.67 | 31.49 | 140 MiB |
| 99,386 | 64 | Initial one-slot stateful ring | 1,180.41 | 26.97 | 167 MiB |
| 190,763 | 32 | Initial one-slot stateful ring | 860.03 | 15.18 | 165 MiB |
| 256,257 | 16 | Initial one-slot stateful ring | 724.61 | 10.55 | 163 MiB |
| 256,257 | 64 | P8 simple | 654.07 | 15.4047 | 273 MiB |
| 256,257 | 64 | Triple pipeline, fused K/V, narrow Q4 | 658.24 | 23.8438 | 106 MiB |
| 256,257 | 64 | Previous plus two-slot prefill pipeline | 698.84 | 23.9680 | 101 MiB |

GPU use at 99K, 191K, and 256K stayed around 15,829-15,833 MiB in the initial ring study. This is the capacity success: device memory becomes approximately O(hot prefix + staging tile) instead of O(total logical context). It is not a constant-time attention algorithm. Every exact full-attention layer still has to incorporate every retained cold K/V cell for a new query. Therefore host traffic and attention work grow with filled context even though VRAM stays flat.

The 256K-capable profile is best understood as a graceful capacity fallback:

- hot, common context remains on the stock all-GPU path;
- 80K-100K remains useful but slower;
- 191K and 256K are possible without OOM, but decode falls to approximately 15 and 24 tg/s depending on ring generation and MTP behavior;
- there is no claim of 64K-speed decode at a fully occupied 256K window.

## Blackwell-specific decode work

The inherited Flash Attention launch table used two warps for the relevant `DKQ=256`, `DV=256`, `ncols=8` shape. A Blackwell-only launch uses 128 threads, or four warps.

| Workload | Two-warps | Four-warps | Effect |
|---|---:|---:|---:|
| 512 prompt, 64 output | About 84.8 tg/s | About 91.5 tg/s | About +8% |
| 7,326 prompt, 512 output, 3 paired runs | 82.313 tg/s | 84.412 tg/s | +2.55% |
| Same 7,326 prefill | 1,746.35 pp/s | 1,743.76 pp/s | -0.15% |

The four-warp launch was accepted because it improved decode and did not reduce precision. It can change a token trajectory relative to the two-warp binary, because parallel floating-point reduction order changes. That is not equivalent to a quantization change, but near-tied logits can diverge, so deterministic token tests and executable quality tests were both retained.

An eight-warp IQ4_XS MMVQ experiment averaged 80.18 tg/s versus 80.304 for four warps and was reverted. More warps did not increase useful memory throughput.

## Deterministic MTP

MTP nondeterminism was not treated as unavoidable. The root cause was different partition geometry between target Q=1 and speculative Q>1 paths: padded or checkpoint-expanded K ranges changed the floating reduction grouping. The fix derives every partition from the same visible causal prefix, masks the partial 128-token tile identically, and handles empty partitions explicitly.

At short context, one partition is used when necessary for target/MTP identity. When the cold boundary is crossed, the corrected P8 geometry is used. The integration resets pending hidden state, verification buffers, deferred prefill, adaptive-chain state, and RNG state at cache-destructive boundaries.

| Prompt | Mode | Prefill pp/s | Decode tg/s | Interpretation |
|---:|---|---:|---:|---|
| Short | Base, no MTP | 1,888.63 | 42.11 | Control |
| Short | Corrected MTP3 | About 1,745 | 82.34 | About 1.95x decode |
| Long gate | Base, no MTP | 1,249.26 | 26.30 | Control |
| Long gate | Corrected MTP3 | 1,160.46 | 51.25 | About 1.95x decode |

The deterministic path cost roughly 9-10% versus an earlier approximately 92 tg/s short P8 mode that allowed target and draft to use different reduction shapes. The faster path was not promoted. Across the deterministic test matrix, 12/12 comparisons were exact, including a 78,651-token case.

Adaptive MTP1-3 did not beat fixed MTP3 overall. A 0.5 acceptance gate gained 0.48% in one short case and lost 0.85% in the long case. MTP4 diverged in 2/12 exactness probes and was rejected. Acceptance is a performance diagnostic, not a quality metric.

## Profiler evidence

Optimization decisions used internal CUDA-event graph timing and NVIDIA Nsight Compute 2026.2.1.0. The successful Nsight capture used driver 591.86 at an effective 100,116-token prompt. It intentionally replayed two selected kernels, so it is kernel evidence, not an end-to-end timeline.

For the two captured approximately 90 us kernels:

| Nsight metric | Kernel A | Kernel B |
|---|---:|---:|
| L2 throughput | About 72% of peak | About 72% of peak |
| DRAM throughput | About 46% of peak | About 46% of peak |
| Compute SM throughput | 8.75% | 8.82% |
| Achieved warp occupancy | 8.31% | 8.28% |
| Scheduler cycles without an eligible warp | 92.48% | 92.50% |
| Registers per thread | 236 | 236 |
| Dynamic shared memory per block | 67.73 KiB | 67.73 KiB |

The grid was 128 blocks x 128 threads. Shared-memory usage allowed only one resident block per SM, and the dominant wait was long scoreboard. Global-sector utilization was poor. This ruled out the simplistic theory that the kernel was compute-saturating Tensor Cores. It was latency/residency constrained at the captured point. It also did not prove that PCIe was the only global bottleneck, because node replay does not preserve the complete copy/compute timeline.

Internal graph timing provided the model-level complement: recurrent blocks accounted for about 82% of decode wall and full-attention blocks about 18%. The recurrent scan itself was only about 1.8%; quantized projections and FFN were dominant. This is why rewriting the scan or adding a persistent activation cache did not produce the expected gain. The useful all-GPU decode ceiling still has to stream roughly 13 GiB of quantized backbone weights for target evaluations.

## Experiments rejected by evidence

| Experiment | Observation | Decision |
|---|---|---|
| Direct kernel reads from mapped host Q4 | 18.0364 tg/s and roughly 132 W | Rejected; PCIe latency/starvation |
| P16 attention partitioning | 32.0829 tg/s vs P8 roughly 34.92 before later pipeline work | Rejected; excess partition/reduction overhead |
| 16K ring tile | Apparent 40.08 tg/s, but no-MTP control improved only 0.072% and MTP trajectory changed | Not credited |
| Delaying 64 MTP tail tokens during prefill | Prefill +4.68%, decode -24.81%, acceptance 39/72 -> 28/102 | Rejected |
| `cudaMemcpyBatchAsync` | CUDA 13.3 rejected it during CUDA Graph capture | Rejected without disabling graphs |
| Fused direct Q4 consumption in MMA | At most +1.01% in one 87K decode; neutral or negative elsewhere; saved 28-62 MiB depending snapshot | Experimental, not production |
| Persistent prefill FSM | -0.09% at 87K and +0.08% at 256K prefill | Correct but no material gain |
| Batched VMM barriers | 1,105.61 vs 1,106.57 pp/s control | Rejected; causal gain zero |
| Wider 8-warp IQ4 kernel | -0.149% generation | Reverted |
| Persistent Q8_1 activation cache | No robust speed gain, +144 MiB VRAM | Rejected |
| Fused GDN normalization | Changes model math and required separate quality proof | Disabled; reference GDN retained |

The rejected list is important. The final number is not the maximum from a pile of cherry-picked runs. Improvements entered the frozen profile only when the mechanism, output behavior, memory effect, and repeatability agreed.

## Target and draft KV precision: Q4_0 versus Q8_0

Q8 KV was tested in both the target and native-MTP draft caches. It was not promoted. This was a runtime precision experiment, not a model-weight quantization change: the UD-IQ4_XS weights, chat template, sampler, and logical 262,144-token window remained fixed.

| Workload | Q4_0 target/draft KV | Q8_0 target/draft KV | Q8 effect |
|---|---:|---:|---:|
| 4,213 input + 512 output, 65,536 hot, three restarts | 1,707.05 pp/s, 105.22 tg/s, 7.339 s wall | 1,707.53 pp/s, 100.24 tg/s, 7.579 s wall | +0.03% pp, -4.74% tg, +3.27% wall |
| 100,116 input + 1,024 output, 32,768 hot | 1,062.37 pp/s, 49.93 tg/s, 114.766 s wall | 905.00 pp/s, 38.44 tg/s, 137.296 s wall | -14.81% pp, -23.00% tg, +19.63% wall |

The short Q4 group peaked at 14,340 MiB and the short Q8 group at 14,544 MiB. For the long same-hot comparison, Q4 peaked at 15,138 MiB and Q8 at 15,730 MiB. Q8 carries 88.89% more KV payload bytes in this layout. The long Q8 value is the mean of two repetitions (38.43 and 38.45 tg/s); the long Q4 control was measured once in this phase.

Q8 was numerically more faithful in a synthetic attention test: quantization-only relative L2 error fell from 0.120951 for Q4 to 0.00753238 for Q8, about 16.06x lower. That is a tensor-fidelity result, not an intelligence or pass@1 score. All four newly collected long outputs had identical 1,024 token IDs across Q4 and Q8, and no agentic-quality gain was demonstrated. The correct conclusion is therefore that no practical quality improvement was observed in this study, not that Q8 can never change or improve an output.

The measured speed penalty is implementation-specific. This runtime has a fused Q4 dequantization path, while Q8 used the generic path; no equivalent fused Q8 kernel was implemented. Q8 with a 65,536-token hot set was measured only on the short prompt. The long Q8 run used 32,768 hot tokens by design, so this study does not claim that Q8 hot-64K produced an OOM. Given the clear long-context cost and absent demonstrated quality benefit, the frozen profile retains Q4_0 for target and draft KV.

## Thinking and multi-turn preservation

The frozen agent profile does not disable reasoning. It enables model reasoning with `--reasoning on`, parses it with `--reasoning-format deepseek`, and enables history preservation with `--reasoning-preserve`. The Unsloth template is also invoked with `enable_thinking=true` and `preserve_thinking=true`.

These controls solve two separate problems:

1. `enable_thinking` allows the model to generate its reasoning trace.
2. `preserve_reasoning`/`preserve_thinking` keeps earlier assistant reasoning in the serialized multi-turn history after a tool result is appended, rather than preserving only the last assistant turn.

The second point matters for an agent. Without preservation, the model can call a tool and then resume without the hypothesis, plan, or interpretation that motivated the call. With preservation, the next turn receives that state and can continue the same investigation. Preservation does not require displaying the trace in the terminal or final answer: a harness can hide reasoning from the user while retaining it in the request history sent back to the model.

This is a chat-template and harness correctness requirement, not a claim that longer visible thinking automatically improves every answer. It also consumes context, so the harness still needs compact tool results and controlled compaction. The repository benchmark below used preserved reasoning for both harnesses. A clean isolated pass@1 claim for preservation alone was not established; the quality gates support the complete frozen stack, including the template and preservation settings.

## Agent harness result: Pi versus Codex

Harness efficiency was measured separately from model inference. The clean paired comparison used the same local Qwen runtime, the same six repositories, three seeds per repository, preserved reasoning, and executable tests.

| Metric, 18 attempts per harness | Codex | Pi |
|---|---:|---:|
| Public tests passed | 18/18 | 18/18 |
| Fully resolved repositories | 15/18 | 15/18 |
| Robust cases passed | 148 | 147 |
| Output tokens | 521,004 | 370,613 |
| Tool calls | 446 | 528 |
| Total wall time | 121m 00s | 82m 01s |
| API time | 117m 28s | 79m 31s |
| Weighted decode | 78.73 tg/s | 81.42 tg/s |
| Weighted prefill | 1,122.05 pp/s | 1,009.53 pp/s |

Pi used 28.87% fewer output tokens and 32.2% less wall time while reaching the same 15/18 full success. It made more tool calls, but the calls and assistant turns were smaller. This is evidence for a lighter scaffold and tighter context management, not evidence that Pi made the GPU 32% faster or made the model more intelligent. The model/server throughput changed little; Pi simply requested less generated work and managed the interaction more economically.

The result also explains why harness wall time is often more important than a microbenchmark: an agent repeatedly prefills an evolving transcript, generates reasoning, calls tools, appends results, and resumes. Compact tool schemas, smaller observations, preserved reasoning state, correct context accounting, and controlled compaction can dominate total task time without changing one CUDA kernel.

## Quality and determinism gates

The runtime changes were evaluated with more than output text inspection:

- CPU copy tests: 476/476;
- CUDA copy tests: 246/246;
- fixed-shape deterministic MTP matrix: 12/12 exact;
- fixed 7,326-token prompt plus 512 output: three matching repetitions;
- 70,072-token state restore/replace/repeat probe: passed before a deliberately stopped cold-tail stress;
- six-repository Pi ring gate: 28/28 public tests, 49/51 robust cases, 5/6 fully solved, matching the frozen 80K executable score;
- clean server shutdown: passed in the frozen profile.

Exact hashes prove identity only for those samples. P8 changes floating-point reduction grouping relative to P1, so P8 can be deterministic with itself while differing from P1. Executable repository tests are therefore retained beside token identity. No LLM judge is used as the primary correctness measure.

Historical Q3 versus IQ4_XS EvalPlus compatibility belongs to the earlier Qwen3.6 quantization study and must not be presented as proof about Qwen3.8 ring arithmetic. Similarly, an Ornith 5/8 result came from a different historical harness and is not an apples-to-apples model ranking here.

## Frozen runtime profile

The final manifest uses one sequence, a 262,144-token logical window, 32 context checkpoints, all model layers on the GPU, Q4_0 target/draft KV, MTP3, and the reference GDN path.

```text
--ctx-size 262144
--parallel 1
--n-gpu-layers all
--load-mode none
--fit off --fit-target 170
--flash-attn on
--cache-type-k q4_0 --cache-type-v q4_0
--cache-type-k-draft q4_0 --cache-type-v-draft q4_0
--threads 16 --threads-batch 16
--batch-size 1024 --ubatch-size 512
--jinja --reasoning on --reasoning-format deepseek --reasoning-preserve
--ctx-checkpoints 32
--no-mmproj --no-webui
--spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-p-min 0.10 --spec-draft-p-split 0.1
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
--repeat-penalty 1 --presence-penalty 0 --seed 424242
```

The model-specific environment is:

```text
LLAMA_MTP_SHARED_COMPUTE=1
LLAMA_MTP_MMA_FIXED_SHAPE=1
LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking":true,"preserve_thinking":true}
LLAMA_QWEN35_BLACKWELL_ENGINE=1

LLAMA_KV_SPARSE_VMM=1
LLAMA_KV_SPARSE_INITIAL_TOKENS=8192
LLAMA_KV_SPARSE_CHUNK_TOKENS=4096
LLAMA_KV_SPARSE_PREFETCH_TOKENS=4096
LLAMA_KV_SPARSE_HOST_FALLBACK=1
LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS=65536
LLAMA_KV_RESERVE_TOKENS=65536

LLAMA_KV_HOST_STAGE_COLD_FA=1
LLAMA_KV_HOST_RING_FA=1
LLAMA_KV_HOST_RING_FORCE_VEC=0
LLAMA_KV_HOST_RING_TILE_TOKENS=8192
LLAMA_KV_HOST_RING_MMA=1
LLAMA_KV_HOST_RING_MMA_COLD_ONLY=1
LLAMA_KV_HOST_RING_MMA_TILE_TOKENS=8192
LLAMA_KV_HOST_RING_MMA_PARTITIONS=8
LLAMA_KV_HOST_RING_MMA_DIRECT_HOT=1
LLAMA_KV_HOST_RING_MMA_PIPELINE=1
LLAMA_KV_HOST_RING_MMA_PREFILL_PIPELINE=1
LLAMA_KV_HOST_RING_MMA_TRIPLE_PIPELINE=1
LLAMA_KV_HOST_RING_MMA_FUSED_Q4_DEQUANT=1
LLAMA_KV_HOST_RING_MMA_Q4_NARROW_BLOCK=1
LLAMA_KV_HOST_RING_MMA_PREFILL_FSM=1
LLAMA_KV_HOST_RING_MMA_INLINE_Q4=0

LLAMA_RS_TRANSACTION_LOG=1
LLAMA_RS_PHASE_ARENA=1
LLAMA_COMPUTE_SPARSE_VMM=1
LLAMA_COMPUTE_SPARSE_TRIM=1
LLAMA_COMPUTE_SPARSE_TRIM_WINDOW=8
LLAMA_COMPUTE_SPARSE_TRIM_HEADROOM_MIB=32
LLAMA_COMPUTE_SPARSE_TRIM_FLOOR_MIB=160
LLAMA_COMPUTE_SPARSE_TRIM_THRESHOLD_MIB=64
LLAMA_COMPUTE_SPARSE_REQUEST_RESET=0
LLAMA_QWEN35_LAYER_MAJOR_WINDOW=0
LLAMA_KV_HOST_RING_MMA_LAYER_CACHE_MIB=0
LLAMA_DISABLE_FUSED_GDN_CH=1
```

Some enabled flags are correctness plumbing or no-op experimental scaffolding rather than measured speedups. In particular, the prefill FSM was functionally validated but measured neutral, and direct inline Q4 is off.

## Build configuration actually used

The build was generated with Visual Studio 2022 x64 and CUDA 13.3 for the native Blackwell `sm_120a` target. The relevant cache values were:

```text
CMAKE_CUDA_ARCHITECTURES=120a-real
GGML_CUDA=ON
GGML_CUDA_FA_ALL_QUANTS=ON
GGML_CUDA_GRAPHS=ON
GGML_CUDA_FORCE_CUBLAS=OFF
GGML_CUDA_FORCE_MMQ=OFF
```

Equivalent configure/build commands are:

```powershell
cmake -S . -B build-blackwell `
  -G "Visual Studio 17 2022" -A x64 `
  -DGGML_CUDA=ON `
  -DGGML_CUDA_FA_ALL_QUANTS=ON `
  -DGGML_CUDA_GRAPHS=ON `
  -DGGML_CUDA_FORCE_CUBLAS=OFF `
  -DGGML_CUDA_FORCE_MMQ=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120a-real

cmake --build build-blackwell --config Release `
  --target llama-server --parallel 4
```

These commands describe the frozen compiler configuration and build this branch's derived ring runtime as `build-blackwell\bin\Release\llama-server-mtpctx.exe`. The distinct output name prevents accidental substitution of a stock server. The convenience launcher in `examples/qwen38-blackwell` records the corresponding environment and server arguments without embedding machine-specific paths.

### Source snapshot validation

The published source snapshot was configured and compiled from a clean build directory with the commands above. The resulting server reported build 10573 from base commit `d775b8967`, accepted the custom command-line controls, loaded the specified UD-IQ4_XS model, reached the health endpoint, and completed an OpenAI-compatible chat-completion smoke request with native MTP activity.

The exact Qwen dense projection checks passed on CUDA: Q4_K gate/up at `[17408,512,5120]` and Q6_K down at `[5120,512,17408]`. A broad CUDA backend run with the legacy allocation pool completed 14,349 of 14,369 cases. The 20 non-passing generic cases were 16 F16 KV-view flash-attention cases with 40-wide heads and four grouped/repeated-head GDN cases. The published production profile does not use either failing shape: Qwen uses 128-wide Q4_0 KV attention, and `LLAMA_DISABLE_FUSED_GDN_CH=1` selects the validated reference graph for chunked GDN. The broad test with the default VMM pool also exposed a LIFO pool assertion after a very long mixed-operation sequence; this did not reproduce in the model smoke path. These results are recorded rather than represented as an all-green generic backend suite.

## Reproduction protocol

1. Verify the model, server, DLL, and template hashes.
2. Confirm PCIe current/max width and generation; do not infer them from the motherboard specification.
3. Close other model servers. Running two copies invalidates the VRAM result.
4. Start one server with the full argument and environment manifest.
5. Warm up once and include sequential long-request repetitions when validating a rebuilt runtime.
6. Record configured window and effective prompt tokens separately.
7. Record prompt tokens/s, decode tokens/s, output tokens/wall second, TTFT, total wall, MTP proposed/accepted, GPU ready/peak memory, and output hash.
8. Compare identical prompt, output limit, sampling, seed, MTP mode, batch, and process lifecycle. Do not mix 64-token and 512-token MTP trajectories.
9. Treat any change to reduction order as a new deterministic variant even if its mathematical formula is equivalent.
10. Require backend tests, exact token probes, executable repository tests, and fresh-load stability before promotion.

## Conclusions

The main result is not that RAM became as fast as VRAM. It did not. The result is that model structure, exact state-lifetime compression, sparse VMM, and a staged DMA/compute ring made a 262K logical context possible on a 16 GB card while preserving a near-zero-cost stock path for the first 65,536 tokens.

The largest individual ring gains came from eight-way online-softmax partitioning and overlapping bulk H2D, Q4 conversion, and MMA consumption. The combined 87K decode improvement was 23.33 -> 41.53 tg/s (+78.03%). At 256K filled, successive generations moved approximately 10.47 -> 23.97 tg/s, while the cold prefill pipeline moved 658.24 -> 698.84 pp/s. Capacity stayed flat in VRAM, but exact attention remained O(context), so speed still declined with filled history.

The NVIDIA profiler prevented optimization by intuition alone. It showed a captured kernel with low compute utilization, one-block-per-SM residency, high long-scoreboard stalls, and poor sector utilization. It also helped distinguish local kernel constraints from the separate PCIe cold-tail problem. Several plausible rewrites were implemented and rejected because they were neutral, slower, unstable, or changed MTP behavior.

For daily agent use, the 65K hot path plus controlled compaction remains the fastest regime. The 256K ring is valuable when capacity matters more than decode latency, and its current process-lifecycle limitation must remain visible rather than hidden behind a benchmark headline.

## External technical references

- [NVIDIA CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [NVIDIA CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [NVIDIA CUDA Programming Guide: Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [NVIDIA RTX 5070 Ti launch specifications](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)
- [PCI-SIG: PCI Express 5.0 transfer rate](https://pcisig.com/what-bit-rates-does-pcie-50-specification-support-and-how-does-it-compare-prior-pcie-generations)
- [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Upstream PR 27663](https://github.com/ggml-org/llama.cpp/pull/27663)
- [Upstream PR 27870](https://github.com/ggml-org/llama.cpp/pull/27870)
- [Upstream PR 28475](https://github.com/ggml-org/llama.cpp/pull/28475)
- [Upstream PR 28549](https://github.com/ggml-org/llama.cpp/pull/28549)
- [Upstream PR 27755](https://github.com/ggml-org/llama.cpp/pull/27755)
- [Upstream PR 27991](https://github.com/ggml-org/llama.cpp/pull/27991)
