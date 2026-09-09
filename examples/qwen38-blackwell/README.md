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

```powershell
.\examples\qwen38-blackwell\start-server.ps1 `
  -Model D:\models\Qwen3.8-27B-UD-IQ4_XS.gguf `
  -ChatTemplate D:\templates\qwen3-coder.jinja
```

The default is one slot, a 262,144-token logical context, a 65,536-token device
prefix, Q4_0 KV for main and draft caches, MTP3, and 32 context checkpoints.
Override `-Context`, `-Port`, or `-Server` when needed.

Important operational constraints:

- Start only one model server. A second process invalidates the VRAM fit.
- The first 65,536 tokens are the fast resident path. A filled cold tail remains
  exact attention and therefore has O(context) decode cost.
- The published measurements used Qwen3.8-27B UD-IQ4_XS. Other models and
  quantizations need independent validation.
- The visual projector and Web UI are disabled in the frozen profile.
- Environment switches are experimental. The ordinary path remains available by
  clearing them and using normal `llama-server` arguments.

See `docs/LOCAL-QWEN-BLACKWELL-STUDY.md` for the state machine, memory accounting,
profiler findings, benchmark progression, limitations, and interpretation.
