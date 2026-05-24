param(
    [ValidateSet("3.9", "3.10", "3.11")]
    [string]$PythonVersion = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $RepoRoot ".venv"

Set-Location $RepoRoot

function Test-CompatiblePython {
    param(
        [string]$Exe,
        [string[]]$PythonArgs = @()
    )

    $Check = @'
import platform
import struct
import sys

version_ok = sys.version_info[:2] in ((3, 9), (3, 10), (3, 11))
arch_ok = struct.calcsize("P") * 8 == 64
print(f"{platform.python_implementation()} {sys.version.split()[0]} {struct.calcsize('P') * 8}-bit")
raise SystemExit(0 if version_ok and arch_ok else 1)
'@

    & $Exe @PythonArgs -c $Check 2>$null
    return $LASTEXITCODE -eq 0
}

$SelectedPython = $null
$SelectedPythonArgs = @()
$CandidateVersions = if ($PythonVersion) { @($PythonVersion) } else { @("3.11", "3.10", "3.9") }

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($Version in $CandidateVersions) {
        $CandidateArgs = @("-$Version")
        if (Test-CompatiblePython -Exe "py" -PythonArgs $CandidateArgs) {
            $SelectedPython = "py"
            $SelectedPythonArgs = $CandidateArgs
            break
        }
    }
}

if (-not $SelectedPython -and (Get-Command python -ErrorAction SilentlyContinue)) {
    if (Test-CompatiblePython -Exe "python") {
        $SelectedPython = "python"
        $SelectedPythonArgs = @()
    }
}

if (-not $SelectedPython) {
    throw "Install 64-bit Python 3.9, 3.10, or 3.11 first. TensorFlow/MediaPipe for this project require one of those versions on Windows."
}

if ($Recreate -and (Test-Path $VenvPath)) {
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment with $SelectedPython $($SelectedPythonArgs -join ' ')"
    & $SelectedPython @SelectedPythonArgs -m venv $VenvPath
}

$Python = Join-Path $VenvPath "Scripts\python.exe"

& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -r requirements.txt
& $Python -m pip install -e .

Write-Host "Windows setup complete."
Write-Host "Run scripts\run_test_window.ps1 for a safe camera preview."
Write-Host "Run scripts\run_solid_edge.ps1 -Show to start Solid Edge controls with preview."
