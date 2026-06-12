# Stack Watch

Live local dashboard for Monterey Bay AI Lab GPU/model/data flow.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\stack_viz.ps1
```

Open `http://127.0.0.1:8765`.

Open Chrome on startup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\stack_viz.ps1 -OpenChrome
```

Use `-OpenBrowser` to open the system default browser instead.

Stack Watch also starts the durable telemetry recorder by default. Logs are
written as daily JSONL files under `ops/telemetry` and are kept forever unless
you set a retention window.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\system_telemetry.ps1 -Status
powershell -NoProfile -ExecutionPolicy Bypass -File .\ops\stack_viz.ps1 -NoTelemetry
```

The dashboard is read-only. It observes:

- RTX GPU and VRAM state from `nvidia-smi`
- relevant Python, Ollama/Qwen, and agent processes
- production, smoke, NeuralForecast, and ops log tails
- recent lakehouse and checkpoint artifacts
- host CPU, RAM/swap, disk, network, and process rollups in telemetry logs

Useful knobs:

- `MBAL_GPU_GUARD_MAX_USED_PCT`
- `MBAL_GPU_GUARD_RESERVE_MIB`
- `-TelemetryIntervalSeconds`
- `-TelemetryRetentionDays` (`0` keeps logs forever)
