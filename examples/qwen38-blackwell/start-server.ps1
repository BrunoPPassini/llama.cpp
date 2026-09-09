[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $ChatTemplate,

    [string] $Server,
    [int] $Port = 18752,
    [int] $Context = 262144
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Server)) {
    $Server = Join-Path $PSScriptRoot "..\..\build-blackwell\bin\Release\llama-server-mtpctx.exe"
}

$serverPath = [System.IO.Path]::GetFullPath($Server)
$modelPath = [System.IO.Path]::GetFullPath($Model)
$templatePath = [System.IO.Path]::GetFullPath($ChatTemplate)

foreach ($requiredPath in @($serverPath, $modelPath, $templatePath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

$runtimeEnvironment = [ordered]@{
    LLAMA_MTP_SHARED_COMPUTE                     = "1"
    LLAMA_MTP_MMA_FIXED_SHAPE                    = "1"
    LLAMA_ARG_CHAT_TEMPLATE_KWARGS               = '{"enable_thinking":true,"preserve_thinking":true}'
    LLAMA_QWEN35_BLACKWELL_ENGINE                = "1"
    LLAMA_KV_SPARSE_VMM                          = "1"
    LLAMA_KV_SPARSE_INITIAL_TOKENS               = "8192"
    LLAMA_KV_SPARSE_CHUNK_TOKENS                 = "4096"
    LLAMA_KV_SPARSE_PREFETCH_TOKENS              = "4096"
    LLAMA_KV_SPARSE_HOST_FALLBACK                = "1"
    LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS         = "65536"
    LLAMA_KV_RESERVE_TOKENS                      = "65536"
    LLAMA_KV_HOST_STAGE_COLD_FA                  = "1"
    LLAMA_KV_HOST_RING_FA                        = "1"
    LLAMA_KV_HOST_RING_FORCE_VEC                 = "0"
    LLAMA_KV_HOST_RING_TILE_TOKENS               = "8192"
    LLAMA_KV_HOST_RING_MMA                       = "1"
    LLAMA_KV_HOST_RING_MMA_COLD_ONLY             = "1"
    LLAMA_KV_HOST_RING_MMA_TILE_TOKENS           = "8192"
    LLAMA_KV_HOST_RING_MMA_PARTITIONS            = "8"
    LLAMA_KV_HOST_RING_MMA_DIRECT_HOT            = "1"
    LLAMA_KV_HOST_RING_MMA_PIPELINE              = "1"
    LLAMA_KV_HOST_RING_MMA_PREFILL_PIPELINE      = "1"
    LLAMA_KV_HOST_RING_MMA_TRIPLE_PIPELINE       = "1"
    LLAMA_KV_HOST_RING_MMA_FUSED_Q4_DEQUANT      = "1"
    LLAMA_KV_HOST_RING_MMA_Q4_NARROW_BLOCK       = "1"
    # The persistent prefill kernel has no material throughput gain on WDDM
    # and can exceed the Windows watchdog interval on long prompts.
    LLAMA_KV_HOST_RING_MMA_PREFILL_FSM           = "0"
    LLAMA_KV_HOST_RING_MMA_INLINE_Q4             = "0"
    LLAMA_RS_TRANSACTION_LOG                     = "1"
    LLAMA_RS_PHASE_ARENA                         = "1"
    LLAMA_COMPUTE_SPARSE_VMM                     = "1"
    LLAMA_COMPUTE_SPARSE_TRIM                    = "1"
    LLAMA_COMPUTE_SPARSE_TRIM_WINDOW             = "8"
    LLAMA_COMPUTE_SPARSE_TRIM_HEADROOM_MIB       = "32"
    LLAMA_COMPUTE_SPARSE_TRIM_FLOOR_MIB          = "160"
    LLAMA_COMPUTE_SPARSE_TRIM_THRESHOLD_MIB      = "64"
    LLAMA_COMPUTE_SPARSE_REQUEST_RESET           = "0"
    LLAMA_QWEN35_LAYER_MAJOR_WINDOW              = "0"
    LLAMA_KV_HOST_RING_MMA_LAYER_CACHE_MIB       = "0"
    LLAMA_DISABLE_FUSED_GDN_CH                   = "1"
}

foreach ($entry in $runtimeEnvironment.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
}

$serverArguments = @(
    "--model", $modelPath,
    "--host", "127.0.0.1",
    "--port", $Port,
    "--ctx-size", $Context,
    "--parallel", "1",
    "--n-gpu-layers", "all",
    "--load-mode", "none",
    "--fit", "off",
    "--fit-target", "170",
    "--flash-attn", "on",
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0",
    "--cache-type-k-draft", "q4_0",
    "--cache-type-v-draft", "q4_0",
    "--threads", "16",
    "--threads-batch", "16",
    "--prio", "0",
    "--prio-batch", "0",
    "--poll", "50",
    "--poll-batch", "1",
    "--batch-size", "1024",
    "--ubatch-size", "512",
    "--jinja",
    "--chat-template-file", $templatePath,
    "--reasoning", "on",
    "--reasoning-format", "deepseek",
    "--reasoning-preserve",
    "--ctx-checkpoints", "32",
    "--no-mmproj",
    "--no-webui",
    "--log-verbosity", "1",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3",
    "--spec-draft-p-min", "0.10",
    "--spec-draft-p-split", "0.1",
    "--temp", "0.6",
    "--top-p", "0.95",
    "--top-k", "20",
    "--min-p", "0.0",
    "--repeat-penalty", "1.0",
    "--presence-penalty", "0.0",
    "--seed", "424242"
)

Write-Host "Starting custom Qwen3.8 Blackwell runtime"
Write-Host "  server:  $serverPath"
Write-Host "  model:   $modelPath"
Write-Host "  context: $Context"
Write-Host "  port:    $Port"

& $serverPath @serverArguments
exit $LASTEXITCODE
