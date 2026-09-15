# Qwen3.8-27B Blackwell launcher

This directory provides a path-parameterized launcher for the custom runtime in
this branch. It intentionally contains no model, chat template, generated binary,
private path, or user data.

## Build

From a Visual Studio 2022 developer PowerShell with CUDA 13.3 installed:

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

The CMake target is `llama-server`; this branch deliberately names the generated
executable `build-blackwell\bin\Release\llama-server-mtpctx.exe` so it cannot be
silently confused with a stock binary.

## Run

The launcher has two fixed profiles:

| Profile | Logical context | GPU-hot prefix | Checkpoints | Purpose |
|---|---:|---:|---:|---|
| `hot-88k` | 88,064 | 88,064 | 8 | Keep the complete 88K window resident when VRAM permits |
| `ring-256k` | 262,144 | 65,536 | 32 | Keep 64K resident and stream the exact cold tail from pinned host RAM |

For the full-hot profile:

```powershell
.\examples\qwen38-blackwell\start-hot-88k.ps1 `
  -Model D:\models\Qwen3.8-27B-UD-IQ4_XS.gguf `
  -ChatTemplate D:\templates\qwen3-coder.jinja
```

For the 256K ring profile:

```powershell
.\examples\qwen38-blackwell\start-ring-256k.ps1 `
  -Model D:\models\Qwen3.8-27B-UD-IQ4_XS.gguf `
  -ChatTemplate D:\templates\qwen3-coder.jinja
```

Both use one slot, Q4_0 target/draft KV, MTP3, preserved thinking, and no
multimodal projector. Override `-Port` or `-Server` when needed. The 88K full-hot
fit depends on driver overhead and other GPU users; verify VRAM margin on the
target machine instead of assuming that every 16 GB card has the same headroom.

## GitHub Actions package

Run `Build Qwen3.8 Blackwell runtime` from the Actions tab. Select `both`,
`hot-88k`, or `ring-256k`. The job builds the custom server for CUDA 13.3 and
native Blackwell `sm_120a`, then publishes a Windows artifact containing the
server, its runtime DLLs, the matching launcher, this guide, and SHA-256 hashes.

GitHub-hosted Windows runners do not provide the target RTX GPU. The workflow
therefore verifies the binary and required command-line surface but does not
claim a GPU load test or benchmark. Model weights and the chat-template file are
not embedded in the artifact.

Reasoning is enabled and preserved across tool turns by the launcher. The trace
may be hidden by the client UI, but it remains in the serialized history so the
model can continue the hypothesis and plan that led to a tool call. Removing
`--reasoning-preserve` or the template preservation kwargs changes agent behavior
and is not the published profile.

Important operational constraints:

- Start only one model server. A second process invalidates the VRAM fit.
- The first 65,536 tokens are the fast resident path. A filled cold tail remains
  exact attention and therefore has O(context) decode cost.
- The published measurements used Qwen3.8-27B UD-IQ4_XS. Other models and
  quantizations need independent validation.
- RTX 5060 Ti 16 GB and RTX 5080 16 GB are plausible Blackwell targets, not
  measured configurations. The 8 GB RTX 5060 Ti cannot fit this exact profile.
- Q8_0 target/draft KV was tested but not promoted: it used more VRAM, slowed the
  long same-hot workload, and showed no practical quality gain in the measured
  long outputs. The frozen launcher intentionally uses Q4_0 KV.
- The visual projector and Web UI are disabled in the frozen profile.
- Environment switches are experimental. The ordinary path remains available by
  clearing them and using normal `llama-server` arguments.

See `docs/LOCAL-QWEN-BLACKWELL-STUDY.md` for the state machine, memory accounting,
profiler findings, benchmark progression, limitations, and interpretation.
