# TTS-STS Quick Reference
# Development workflow optimized for minimal rebuild time

## Initial Setup (One-time)
```powershell
# Build the base development image (takes time, but cached)
.\build.ps1 build

# Or use docker-compose
docker-compose build tts-dev
```

## Fast Development Cycle

### Option 1: Using the build script (Recommended)
```powershell
# Run in development mode (with live code mounting)
.\build.ps1 dev

# Run with arguments
.\build.ps1 dev "--keep-chunks --skip-pdf"

# Interactive shell for debugging
.\build.ps1 shell

# Force rebuild if dependencies changed
.\build.ps1 build -Force
```

### Option 2: Using docker-compose
```powershell
# Run development version
docker-compose run --rm tts-dev

# Run with arguments
docker-compose run --rm tts-dev --keep-chunks --skip-pdf
```

### Option 3: Direct docker commands (if needed)
```powershell
# Development mode with live mounting (no rebuild needed for code changes)
docker run --rm `
    -v "${PWD}\input:/app/input" `
    -v "${PWD}\output:/app/output" `
    -v "${PWD}\model_cache:/root/.local/share/tts" `
    -v "${PWD}\app.py:/app/app.py:ro" `
    tts-sts:dev --skip-pdf
```

## Time-Saving Tips

1. **Use `--skip-pdf`** for faster iterations when testing TTS changes
2. **Use `--no-resume`** only when you want to regenerate all chunks
3. **Keep chunks** during development with `--keep-chunks`
4. **Live code mounting** means no rebuilds for app.py changes
5. **Model cache** is persisted between runs

## Troubleshooting

### Clean everything and start fresh:
```powershell
.\build.ps1 clean
.\build.ps1 build -Force
```

### Check what's running:
```powershell
docker ps
docker images | grep tts-sts
```

### View logs:
```powershell
docker logs <container_id>
```

## Performance Comparison

| Method | Rebuild Time | Use Case |
|--------|-------------|----------|
| Original Dockerfile | ~5-10 minutes | Production |
| Multi-stage Dev | ~1-2 minutes | Dependencies changed |
| Live mounting | ~5 seconds | Code changes only |
| Skip PDF + Resume | ~30 seconds | TTS testing |
