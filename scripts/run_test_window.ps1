param(
    [int]$Camera = 0,
    [int]$Width = 960,
    [int]$Height = 540,
    [int]$CubeWidth = 560,
    [switch]$Control,
    [double]$MinimumPredictionConfidence = 0.8
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$Culture = [System.Globalization.CultureInfo]::InvariantCulture
$CommandArgs = @(
    "-m", "hamoco.cli.hamoco_test_window",
    "--camera", $Camera.ToString($Culture),
    "--width", $Width.ToString($Culture),
    "--height", $Height.ToString($Culture),
    "--cube_width", $CubeWidth.ToString($Culture),
    "--minimum_prediction_confidence", $MinimumPredictionConfidence.ToString($Culture)
)

if ($Control) {
    $CommandArgs += "--control"
}

Set-Location $RepoRoot
& $Python @CommandArgs
