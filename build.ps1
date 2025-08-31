#!/usr/bin/env powershell
# Fast development workflow for TTS-STS

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "run", "dev", "shell", "clean", "watch", "help")]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Args = "",
    
    [switch]$Force,
    [switch]$Production
)

# Configuration
$IMAGE_DEV = "tts-sts:dev"
$IMAGE_PROD = "tts-sts:prod"
$DOCKERFILE = "Dockerfile.dev"

function Show-Help {
    Write-Host @"
TTS-STS Fast Development Tools

Usage: .\build.ps1 <command> [options]

Commands:
  build     Build Docker image (dev by default, use -Production for prod)
  run       Run the application
  dev       Run in development mode with live code mounting
  shell     Start interactive shell in container
  clean     Clean Docker resources
  watch     Watch for changes and auto-restart
  help      Show this help

Options:
  -Production    Use production build/image
  -Force         Force rebuild without cache
  -Args <args>   Pass arguments to the application

Examples:
  .\build.ps1 build                    # Build development image
  .\build.ps1 build -Production        # Build production image
  .\build.ps1 run --keep-chunks        # Run with arguments
  .\build.ps1 dev                      # Run in development mode
  .\build.ps1 shell                    # Interactive shell
"@ -ForegroundColor Green
}

function Build-Image {
    param([bool]$IsProd, [bool]$ForceRebuild)
    
    $target = if ($IsProd) { "production" } else { "development" }
    $tag = if ($IsProd) { $IMAGE_PROD } else { $IMAGE_DEV }
    
    Write-Host "Building $target image: $tag" -ForegroundColor Green
    
    $buildArgs = @(
        "build"
        "-f", $DOCKERFILE
        "--target", $target
        "-t", $tag
    )
    
    if ($ForceRebuild) {
        $buildArgs += "--no-cache"
    }
    
    $buildArgs += "."
    
    & docker @buildArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Build completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Build failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Run-Container {
    param([string]$Image, [string]$Arguments, [bool]$WithCodeMount = $false)
    
    $volumeArgs = @(
        "-v", "${PWD}\input:/app/input"
        "-v", "${PWD}\output:/app/output" 
        "-v", "${PWD}\model_cache:/root/.local/share/tts"
    )
    
    if ($WithCodeMount) {
        $volumeArgs += "-v", "${PWD}\app.py:/app/app.py:ro"
        Write-Host "Running with live code mounting..." -ForegroundColor Cyan
    }
    
    $runArgs = @("run", "--rm") + $volumeArgs + $Image
    
    if ($Arguments) {
        $runArgs += $Arguments.Split(' ')
    }
    
    Write-Host "Command: docker $($runArgs -join ' ')" -ForegroundColor DarkGray
    & docker @runArgs
}

function Start-Shell {
    param([string]$Image)
    
    Write-Host "Starting interactive shell..." -ForegroundColor Cyan
    
    & docker run --rm -it `
        -v "${PWD}\input:/app/input" `
        -v "${PWD}\output:/app/output" `
        -v "${PWD}\model_cache:/root/.local/share/tts" `
        -v "${PWD}\app.py:/app/app.py:ro" `
        --entrypoint bash `
        $Image
}

function Clean-Resources {
    Write-Host "Cleaning Docker resources..." -ForegroundColor Yellow
    & docker system prune -f
    & docker image prune -f
    Write-Host "✅ Cleanup completed!" -ForegroundColor Green
}

# Main logic
switch ($Command) {
    "build" {
        Build-Image -IsProd $Production -ForceRebuild $Force
    }
    "run" {
        $image = if ($Production) { $IMAGE_PROD } else { $IMAGE_DEV }
        Run-Container -Image $image -Arguments $Args
    }
    "dev" {
        Run-Container -Image $IMAGE_DEV -Arguments $Args -WithCodeMount $true
    }
    "shell" {
        $image = if ($Production) { $IMAGE_PROD } else { $IMAGE_DEV }
        Start-Shell -Image $image
    }
    "clean" {
        Clean-Resources
    }
    "watch" {
        Write-Host "File watching not implemented in this script. Use dev-tools.ps1 Watch-Dev function." -ForegroundColor Yellow
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
