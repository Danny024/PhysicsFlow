# PhysicsFlow Installer Build Script
# =====================================
# Requires:
#   - WiX Toolset v4 (dotnet tool install --global wix)
#   - .NET 8 SDK
#   - Python 3.11 embedded distributable in engine/python/
#
# Usage:
#   .\build.ps1 [-Configuration Release] [-Version 1.1.0]

param(
    [string]$Configuration = "Release",
    [string]$Version = "1.1.0",
    [string]$Platform = "x64"
)

$ErrorActionPreference = "Stop"
$Root       = Split-Path $PSScriptRoot -Parent
$AppDir     = "$Root\desktop\src\PhysicsFlow.App"
$EngineDir  = "$Root\engine"
$InstallerDir = "$Root\installer"
$PublishDir = "$InstallerDir\publish"
$OutputMsi  = "$InstallerDir\PhysicsFlow-Setup-$Version-x64.msi"
$OutputExe  = "$InstallerDir\PhysicsFlow-Installer-$Version-x64.exe"

Write-Host "PhysicsFlow Installer Build v$Version" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration" -ForegroundColor Gray

# ── Step 1: Publish .NET application ────────────────────────────────────
Write-Host "`n[1/4] Publishing .NET application..." -ForegroundColor Yellow
dotnet publish "$AppDir\PhysicsFlow.App.csproj" `
    --configuration $Configuration `
    --runtime "win-$Platform" `
    --self-contained true `
    --output "$PublishDir\app" `
    -p:PublishSingleFile=false `
    -p:PublishTrimmed=false

if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }
Write-Host "  Published to $PublishDir\app" -ForegroundColor Green

# ── Step 2: Copy engine files ────────────────────────────────────────────
Write-Host "`n[2/4] Copying Python engine..." -ForegroundColor Yellow
$engineOutput = "$PublishDir\engine"
New-Item -ItemType Directory -Force -Path $engineOutput | Out-Null
Copy-Item -Path "$EngineDir\*" -Destination $engineOutput -Recurse -Force
Write-Host "  Engine copied to $engineOutput" -ForegroundColor Green

# ── Step 3: Build MSI ────────────────────────────────────────────────────
Write-Host "`n[3/4] Building MSI..." -ForegroundColor Yellow
Push-Location $InstallerDir

wix build PhysicsFlow.wxs `
    -d "AppPublishDir=$PublishDir\app" `
    -d "EngineDir=$engineOutput" `
    -o $OutputMsi

if ($LASTEXITCODE -ne 0) { Pop-Location; throw "wix build MSI failed" }
Pop-Location
Write-Host "  MSI: $OutputMsi" -ForegroundColor Green

# ── Step 4: Build bootstrapper EXE ──────────────────────────────────────
Write-Host "`n[4/4] Building bootstrapper bundle..." -ForegroundColor Yellow
Push-Location $InstallerDir

wix build PhysicsFlow.Bundle.wxs `
    -d "MsiPath=$OutputMsi" `
    -ext WixToolset.Bal.wixext `
    -o $OutputExe

if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Warning "Bundle build failed (requires bal extension)" }
Pop-Location

# ── Summary ──────────────────────────────────────────────────────────────
Write-Host "`n=== Build Complete ===" -ForegroundColor Cyan
if (Test-Path $OutputMsi) {
    $msiSize = [math]::Round((Get-Item $OutputMsi).Length / 1MB, 1)
    Write-Host "  MSI  : $OutputMsi  ($msiSize MB)" -ForegroundColor Green
}
if (Test-Path $OutputExe) {
    $exeSize = [math]::Round((Get-Item $OutputExe).Length / 1MB, 1)
    Write-Host "  EXE  : $OutputExe  ($exeSize MB)" -ForegroundColor Green
}
