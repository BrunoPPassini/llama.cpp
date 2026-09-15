[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $ChatTemplate,

    [string] $Server,
    [int] $Port = 18752
)

$arguments = @{
    Model        = $Model
    ChatTemplate = $ChatTemplate
    Port         = $Port
    Profile      = "hot-88k"
}

if (-not [string]::IsNullOrWhiteSpace($Server)) {
    $arguments.Server = $Server
}

& (Join-Path $PSScriptRoot "start-server.ps1") @arguments
exit $LASTEXITCODE
