# Local Qwen3.8-27B on RTX 5070 Ti

This document records the reproducible, local engineering study performed with a Qwen3.8-27B UD-IQ4_XS GGUF and a CUDA/llama.cpp runtime tuned for a 16 GB RTX 5070 Ti. It is a research profile, not a claim that every GPU or model benefits from the same settings.

## Frozen model and runtime

- Model: `Qwen3.8-27B-UD-IQ4_XS.gguf` (UD-IQ4_XS; 14,252,845,984 bytes).
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
- Derived runtime build: `b5e8b02f4182b42aba03def31e39437c729bd3f7-derived-deterministic-mtp-blackwell-fa4-stability-recurrent-prefix-20260904`.
- Frozen server SHA-256: `3515fe70c89f0525559f455ec0ce234c1b4221d1cd2a2e98d4171af932f78ffa`.
- CUDA library SHA-256: `1c26d70a33e9996cbb7646328ea4c3a67b27a221432f7fae468112eecb860242`.
- Chat template SHA-256: `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`.

The production profile uses a 262,144-token allocated window, 32 context checkpoints, one sequence, 16 CPU threads, batch/ubatch 1024/512, Flash Attention, and Q4_0 KV for both target and draft. Reasoning is enabled and preserved. Native MTP3 is enabled with deterministic causal partitioning. The allocated window is not the same as an actually filled 256K prompt; all reported prompt lengths below are effective tokens processed.

## What was changed

The stable runtime combines correctness and memory-lifetime fixes that were validated together:

1. CUDA copy destination-contiguity guard (upstream PR #27663).
2. Uniform Flash Attention barriers and warp synchronization (PRs #27870 and #28475).
3. Separate CUDA graph identities for MTP prefill/no-output and decode/output paths, with active-identity gating and compute-arena invalidation (adapted from PR #28549).
4. Byte-cursor state copying for fragmented writer/reader buffers and batched contiguous KV restore runs (PRs #27755 and #27991).
5. Sparse KV virtual-memory/ring path with hot device prefix and host-backed cold tail, plus recurrent-prefix state restoration.
6. Reference GDN math is retained (`LLAMA_DISABLE_FUSED_GDN_CH=1`) because the fused normalization variant requires a separate quality study.

The runtime does not change model weights, quantization, sampling semantics, chat-template tokens, or the mathematical GDN path. The launch profile is intentionally explicit so that an accidental automatic fit does not silently change placement.

## Build

The source tree is ordinary llama.cpp and can be built with the CUDA backend. A typical Windows build is:

```powershell
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server llama-cli llama-bench
```

Use a CUDA toolkit/driver supported by the installed NVIDIA driver. The frozen binary was produced from the derived source/runtime tree recorded in the local study manifest; this repository intentionally does not commit GGUF weights or generated binaries.

## Run the frozen profile

Use the supplied launcher from the study workspace, or translate its manifest into an equivalent command. The essential options are:

```text
--ctx-size 262144 --parallel 1 --n-gpu-layers all --fit off
--flash-attn on --cache-type-k q4_0 --cache-type-v q4_0
--cache-type-k-draft q4_0 --cache-type-v-draft q4_0
--threads 16 --threads-batch 16 --batch-size 1024 --ubatch-size 512
--jinja --reasoning on --reasoning-format deepseek --reasoning-preserve
--spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-p-min 0.10 --spec-draft-p-split 0.1
--ctx-checkpoints 32 --no-mmproj --no-webui
```

The production launcher also sets the sparse-KV and recurrent-prefix environment variables recorded in the profile manifest. Keep the server lifecycle conservative: after a near-full long-context session, restart it before starting a new one. This avoids retaining large device/host mappings longer than intended.

## Measurements

The main stable end-to-end result at an effective 7,326-token prompt plus 512 generated tokens was:

| Runtime | Prompt tok/s | Decode tok/s | MTP acceptance | Output identity |
|---|---:|---:|---:|---|
| Approved frozen runtime (3 repetitions) | 1,439.742 | 71.164 | unchanged vs control | identical |
| Shape-gated XOR-swizzle experiment (3 repetitions) | 1,458.238 | 73.802 | 73.375% | identical to control |

At an effective 31,890-token prompt, the shape-gated swizzle measured 1,242.835 prompt tok/s and 49.792 decode tok/s with 67.857% acceptance, versus 1,278.38/46.56 for the approved control. Decode improved 6.949%, while total wall time was effectively unchanged (+0.035% benefit) and all three completion hashes matched. The swizzle was **not** promoted because a later fresh-load run stalled in `load_model`; the stable production binary remains the approved control.

The long-context path is hybrid. A 65,536-token (and larger) value in a launcher means allocated capacity; it does not prove real-time processing of a 65K prompt. At approximately 70K effective history, the sparse VMM exhausted device backing and moved a 28 MiB allocation to host; prefill fell to 167.66 tok/s. This is documented as a cold/ring stress result, not as a 64K-real-time claim.

Earlier quantization/MTP studies found that UD-IQ4_XS was the fastest reproducible 64K-allocated candidate on this card (about 92.55 decode tok/s in a normalized all-GPU MTP3 run). Higher-BPW S and NVFP4-MEDIUM profiles required more offload and were substantially slower in the tested 64K conditions. Those comparisons are retained as historical evidence; they are not silently mixed with the stable runtime measurements above.

The cross-profile snapshot was: UD-IQ4_XS MTP3 92.55 tok/s (three normalized repetitions); NVFP4-LOW about 40.76 tok/s; NVFP4-MEDIUM peaked at about 33.87 tok/s while missing its prefill target; and Q4_K_S reached 38.58 tok/s only in a more favorable 12K/offload condition, not at 64K. The 90%-of-fastest rule therefore made XS the only eligible 64K candidate. In quality work, Qwen Q3 versus IQ4_XS stayed statistically compatible on the frozen EvalPlus selection, while the official Ornith 35B Ollama run solved 5/8 agentic repository tasks under the tested harness. These are workload-specific observations, not universal rankings.

## Correctness and benchmark coverage

- CPU copy tests: 476/476.
- CUDA copy tests: 246/246.
- Deterministic short-output comparison: passed.
- Fixed 7,326-token prompt plus 512 output: three repetitions, identical output and MTP acceptance to control.
- 70,072-token history/replace/repeat state-restoration probe: passed before the intentionally stopped cold-tail stress.
- Clean server shutdown: passed.

Quality work also included deterministic token exactness, EvalPlus subsets/expansions, repository repair tasks, agentic harness runs, long-context tasks, and comparisons against Qwen Q3/Q4 and the official Ornith 35B Ollama model. The detailed raw artifacts remain outside this source repository because they contain local paths, generated outputs, and large logs. Results must be interpreted as task-suite evidence rather than a universal intelligence ranking.

## Limitations and non-results

- 16 GB VRAM is the binding constraint; all-GPU MEDIUM at 8K context produced an OOM while reserving a 140.28 MiB compute buffer.
- Closest-fit MEDIUM and partial-KV escalations were not fully completed when the local subprocess route was unavailable; no claim is made that 45 tok/s is mathematically impossible.
- Context-checkpoint retention (PR #28302) was not promoted after an unexplained connection reset in one long request.
- XOR swizzle is experimental and not part of the frozen launcher until fresh-load stability is revalidated.
- GDN normalization changes were not integrated because they alter model math and require independent quality testing.
- MTP acceptance is a performance diagnostic, not a quality metric. Deterministic token identity must be checked separately.
- Allocated context, effective prompt length, prompt throughput, decode throughput, and wall time are distinct measurements.
- No model weights, private repositories, credentials, VPN details, user data, or application source code are included in this documentation.

## Reproduction checklist

1. Verify the model SHA-256 and the runtime/library hashes above.
2. Start one server with the explicit profile; do not run multiple copies concurrently.
3. Warm up once, then collect at least three repetitions per effective prompt length.
4. Record prompt token count, decode token count, TTFT, prompt/decode/wall tok/s, acceptance, VRAM/RAM, and termination reason.
5. Separate allocated-window tests from real-filled-context tests.
6. Treat any new binary or math change as experimental until backend tests, exactness, fresh-load stability, and quality checks all pass.

Upstream references: [llama.cpp](https://github.com/ggml-org/llama.cpp), [PR #27663](https://github.com/ggml-org/llama.cpp/pull/27663), [PR #27870](https://github.com/ggml-org/llama.cpp/pull/27870), [PR #28475](https://github.com/ggml-org/llama.cpp/pull/28475), [PR #28549](https://github.com/ggml-org/llama.cpp/pull/28549), [PR #27755](https://github.com/ggml-org/llama.cpp/pull/27755), and [PR #27991](https://github.com/ggml-org/llama.cpp/pull/27991).
