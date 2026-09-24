param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ModelSource,

    [ValidateNotNullOrEmpty()]
    [string]$Service = "birdly-api",

    [ValidateNotNullOrEmpty()]
    [string]$Region = "us-east4",

    [string]$Project,

    [ValidateRange(1, 300)]
    [int]$HealthTimeoutSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or is not on PATH"
}
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud is not installed or is not on PATH"
}

$repositoryRoot = $PSScriptRoot
$buildDirectory = Join-Path $repositoryRoot "deploy/cloud-run/build"
$resolvedModelSource = (Resolve-Path -LiteralPath $ModelSource).Path

Push-Location $repositoryRoot
try {
    Invoke-CheckedCommand -Command "uv" -Arguments @(
        "run",
        "python",
        "scripts/prepare_cloud_run.py",
        "--model-source",
        $resolvedModelSource
    )

    if (-not (Test-Path -LiteralPath $buildDirectory -PathType Container)) {
        throw "Cloud Run build directory was not created: $buildDirectory"
    }

    $deployArguments = @(
        "run",
        "deploy",
        $Service,
        "--source",
        $buildDirectory,
        "--region",
        $Region,
        "--quiet"
    )
    if ($Project) {
        $deployArguments += @("--project", $Project)
    }
    Invoke-CheckedCommand -Command "gcloud" -Arguments $deployArguments

    $describeArguments = @(
        "run",
        "services",
        "describe",
        $Service,
        "--region",
        $Region,
        "--format=value(status.url)"
    )
    if ($Project) {
        $describeArguments += @("--project", $Project)
    }
    $serviceUrlOutput = & gcloud @describeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud service lookup failed with exit code $LASTEXITCODE"
    }
    $serviceUrl = ($serviceUrlOutput | Select-Object -Last 1).TrimEnd("/")
    if (-not $serviceUrl) {
        throw "Cloud Run did not return a service URL"
    }

    $health = Invoke-RestMethod `
        -Method Get `
        -Uri "$serviceUrl/health" `
        -TimeoutSec $HealthTimeoutSeconds
    if ($health.status -ne "ok" -or $health.model_loaded -ne $true) {
        throw "The deployed Birdly API is not healthy"
    }

    Write-Host "Birdly deployment succeeded"
    Write-Host "Service URL: $serviceUrl"
    Write-Host "Health status: $($health.status)"
}
finally {
    Pop-Location
}
