# TTS-STS Development Scripts
# PowerShell commands for faster development workflow

# Build development image (only rebuilds if dependencies change)
function Build-Dev {
    Write-Host "Building development image..." -ForegroundColor Green
    docker build -f Dockerfile.dev --target development -t tts-sts:dev .
}

# Build production image
function Build-Prod {
    Write-Host "Building production image..." -ForegroundColor Green
    docker build -f Dockerfile.dev --target production -t tts-sts:prod .
}

# Quick rebuild (only app.py changes)
function Build-Quick {
    Write-Host "Quick rebuild for app.py changes..." -ForegroundColor Yellow
    docker build -f Dockerfile.dev --target development -t tts-sts:dev . --no-cache --build-arg CACHEBUST=$(Get-Date -UFormat %s)
}

# Run development container with live code mounting
function Run-Dev {
    param(
        [string]$Args = ""
    )
    Write-Host "Running development container..." -ForegroundColor Green
    docker run --rm `
        -v "${PWD}\input:/app/input" `
        -v "${PWD}\output:/app/output" `
        -v "${PWD}\model_cache:/root/.local/share/tts" `
        -v "${PWD}\app.py:/app/app.py:ro" `
        tts-sts:dev $Args
}

# Run production container
function Run-Prod {
    param(
        [string]$Args = ""
    )
    Write-Host "Running production container..." -ForegroundColor Green
    docker run --rm `
        -v "${PWD}\input:/app/input" `
        -v "${PWD}\output:/app/output" `
        -v "${PWD}\model_cache:/root/.local/share/tts" `
        tts-sts:prod $Args
}

# Interactive development shell
function Dev-Shell {
    Write-Host "Starting interactive development shell..." -ForegroundColor Cyan
    docker run --rm -it `
        -v "${PWD}\input:/app/input" `
        -v "${PWD}\output:/app/output" `
        -v "${PWD}\model_cache:/root/.local/share/tts" `
        -v "${PWD}\app.py:/app/app.py:ro" `
        --entrypoint bash `
        tts-sts:dev
}

# Clean up Docker resources
function Clean-Docker {
    Write-Host "Cleaning up Docker resources..." -ForegroundColor Yellow
    docker system prune -f
    docker image prune -f
}

# Watch for file changes and auto-restart (requires PowerShell 7+)
function Watch-Dev {
    Write-Host "Watching for file changes..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop watching" -ForegroundColor Yellow
    
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = $PWD
    $watcher.Filter = "app.py"
    $watcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite
    $watcher.EnableRaisingEvents = $true
    
    Register-ObjectEvent -InputObject $watcher -EventName Changed -Action {
        Write-Host "File changed, restarting container..." -ForegroundColor Yellow
        Run-Dev
    }
    
    try {
        while ($true) {
            Start-Sleep 1
        }
    }
    finally {
        $watcher.Dispose()
    }
}

# Export functions
Export-ModuleMember -Function Build-Dev, Build-Prod, Build-Quick, Run-Dev, Run-Prod, Dev-Shell, Clean-Docker, Watch-Dev

Write-Host @"
TTS-STS Development Commands Available:
  Build-Dev       - Build development image
  Build-Prod      - Build production image  
  Build-Quick     - Quick rebuild for minor changes
  Run-Dev         - Run with live code mounting
  Run-Prod        - Run production version
  Dev-Shell       - Interactive development shell
  Clean-Docker    - Clean up Docker resources
  Watch-Dev       - Auto-restart on file changes
"@ -ForegroundColor Green
