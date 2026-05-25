param(
    [switch]$Launch
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$CommandArgs = @("-m", "hamoco.cli.solid_edge_probe")
if ($Launch) {
    $CommandArgs += "--launch"
}

Set-Location $RepoRoot
& $Python @CommandArgs
