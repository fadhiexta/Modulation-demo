# run_all_experiments.ps1
# Usage: buka PowerShell, aktifkan venv, lalu:
#    .\run_all_experiments.ps1
# Optional args: -SNRList "0,2,4,6,8,10" -DemoBits 256 -BerBits 5000 -Seed 0

param(
    [string]$SNRList = "0,2,4,6,8,10",
    [int]$DemoBits = 256,
    [int]$BerBits = 5000,
    [int]$Seed = 0
)

# ensure running from project root
$projRoot = (Get-Location).Path
Write-Host "Project root: $projRoot"

# ensure results dir exists
$plotsDir = Join-Path $projRoot "results\plots"
New-Item -ItemType Directory -Path $plotsDir -Force | Out-Null

# 1) AM demo
Write-Host "`n== Running AM demo =="
python -m experiments.run_am_demo
Write-Host "AM demo finished."

# 2) BPSK demo for multiple SNRs
$splits = $SNRList -split ","
Write-Host "`n== Running BPSK demos for SNRs: $SNRList =="
foreach ($s in $splits) {
    $sTrim = $s.Trim()
    Write-Host "`n-> Running BPSK demo SNR=$sTrim dB ..."
    python -m experiments.run_bpsk_demo --snr $sTrim --bit-rate 100 --n-bits $DemoBits
    Start-Sleep -Milliseconds 300
}

# 3) BER sweep
Write-Host "`n== Running BER sweep =="
$saveCsvFlag = "--save-csv"
$seedFlag = ""
if ($Seed -ne 0) { $seedFlag = "--seed $Seed" }
python -m experiments.run_bpsk_ber --snr-min 0 --snr-max 10 --snr-step 1 --n-bits $BerBits $seedFlag $saveCsvFlag

Write-Host "`nAll experiments finished. Check results\plots folder."
