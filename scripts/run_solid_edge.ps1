param(
    [switch]$Show,
    [double]$Sensitivity = 0.5,
    [double]$ScrollingSpeed = 40.0,
    [double]$ScrollingThreshold = 0.1,
    [string]$DragModifier = "none",
    [string]$DragButton = "middle"
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
    "-m", "hamoco.cli.hamoco_run",
    "--sensitivity", $Sensitivity.ToString($Culture),
    "--scrolling_speed", $ScrollingSpeed.ToString($Culture),
    "--scrolling_threshold", $ScrollingThreshold.ToString($Culture),
    "--drag_modifier", $DragModifier,
    "--drag_button", $DragButton
)

if ($Show) {
    $CommandArgs += "--show"
}

Set-Location $RepoRoot
& $Python @CommandArgs
