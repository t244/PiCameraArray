# PiCameraArray (TBD)

A Python library for managing multiple camera modules on Raspberry Pi systems.

## Features

- Support for multiple camera modules
- Easy configuration and setup
- Capture images and video streams
- Real-time processing capabilities

## Requirements

- Raspberry Pi with camera module support
- Python 3.7+
- picamera library

## Setup & Operation (Camera Agent)

Each Pi runs `capture/camera_agent.py` as a resident service (port 8000).
It boots into **capture** mode (external trigger, headless) and can be
switched to **preview** mode (free-running MJPEG stream) via HTTP for
on-site focusing and exposure adjustment.

### Field setup workflow

1. Power on the array. Bring a portable router (local AP, no internet
   needed); connect the laptop to the same SSID as the Pis.
2. Open `dashboard/dashboard.html` in a browser (set the host pattern,
   e.g. `e{NN}` or `e{NN}.local`).
3. Click **全台 Preview** — all 16 streams appear with a live focus
   metric (higher = sharper). Adjust each lens.
4. Adjust the exposure slider and **Previewに反映** until the image looks
   right (fully manual; the same value is applied to all cameras).
5. **Arduinoへ書込み** — sends `E<us>` to the Arduino via the master Pi
   (the Pi with the USB serial connection); saved to EEPROM.
6. **全台 Capture**, then **トリガ開始**.
7. Remove the router. From now on the system is fully headless: on
   power-up the Pis boot into capture mode and the Arduino auto-starts
   after 2 minutes using the EEPROM exposure value.

### Agent HTTP API (port 8000)

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | mode, focus, temp, storage, capture count |
| `/mode` | POST | `{"mode": "preview"\|"capture"}` |
| `/stream.mjpg` | GET | MJPEG stream (preview only) |
| `/controls` | POST | `{"exposure_us": int, "gain": float}` (preview only) |
| `/last.jpg` | GET | thumbnail of last captured frame (capture mode) |
| `/trigger/exposure` | POST | `{"exposure_us": int}` → Arduino `E` (master only) |
| `/trigger/period` | POST | `{"period_ms": int}` → Arduino `T` (master only) |
| `/trigger/fps` | POST | `{"fps": int}` → Arduino `F` (master only) |
| `/trigger/burst` | POST | `{"burst_ms": int}` → Arduino `B` (master only) |
| `/trigger/start` `/trigger/stop` | POST | Arduino `S` / `P` (master only) |
| `/trigger/status` | GET | Arduino `R` (master only) |

Note: in IMX296 trigger mode the exposure time equals the trigger pulse
width, so the exposure is a single global value for the whole array.

### Burst capture (video analysis)

The Arduino fires a pulse train every period: `fps` (default 30) pulses/s
for `burst` (default 3 s) → 90 hardware-synchronized frames per burst.
Each Pi buffers the frames in RAM and writes one npz file per burst
(`<host>_burstNNNN_<timestamp>.npz`, keys `frames` (N,1088,1456) uint8 and
`timestamps`) during the idle time between bursts.

Constraints: exposure < 1/fps (e.g. <33 ms at 30 fps, <15 ms at 60 fps);
max 450 frames/burst (RAM); storage ≈ 1.58 MB × fps × burst_s per burst
per camera (256 GB SSD ≈ 28 h at defaults: 30 fps × 3 s / 60 s period).

## Documentation

For detailed documentation, see the [docs](./docs) directory.

## Useful things

For Windows' PowerShell, add the below into `notepad $PROFILE`:

```powershell
function Invoke-PiCommand {
    param([string]$Command)
    0..15 | ForEach-Object -Parallel {
    	$h = "e{0:D2}" -f $_
    	$result = plink -pw pi -batch pi@$h $using:Command 2>&1
    	Write-Output "=== $h ==="
    	Write-Output $result
    } -ThrottleLimit 16
}
```

## License

MIT License
