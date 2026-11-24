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
6. **全台 Capture**. Check the trigger box lamp turns steady green (all
   16 Pis ready), then flip the toggle ON: the first burst fires at once.
   (**トリガ開始/停止** still work as a setup override until the toggle is
   next moved.)
7. Remove the router. From now on the system is fully headless and is run
   with the toggle on the trigger box only (see below).

### Unattended operation (trigger box)

Wiring on the Arduino Uno (details in `capture/exposure_trigger/exposure_trigger.ino`):

| Uno pin | Connection |
|---|---|
| D13 | sync output, fanned out to the 16 camera trigger inputs |
| D2 | capture toggle switch to GND (closed = ON) |
| D5 / D6 | bicolor LED red / green anode, each via 330 Ω (common cathode to GND) |
| A0 | shared READY line: 10 kΩ pull-up to the Uno **3.3V** pin, plus one wire to GPIO21 (header pin 40) of every Pi, with ground from pin 34 |

Each Pi holds the READY line LOW unless it is recording (booting, waiting
for the SSD, SSD full, agent stopped); the line is HIGH only when all 16
have released it. Run `scripts/setup_ready_line.sh` once on every Pi (adds
`gpio=21=op,dl` to config.txt so the line is LOW from early boot) and
reinstall `config/picamera-capture.service` (pulls the line LOW when the
agent stops). Never pull the line up to 5 V: Pi GPIOs are 3.3 V only.

The toggle is level-triggered: capture runs whenever all Pis are ready and
the toggle is ON. After a power cut, capture therefore resumes by itself.
Power on with the toggle OFF.

| Lamp | Meaning |
|---|---|
| green blinking | waiting for all Pis to become ready |
| green steady | ready, toggle OFF |
| red steady | recording, between bursts |
| red fast blinking | burst in progress |
| red/green alternating | not all Pis ready 5 min after power-on |

A Pi dropping out during capture is deliberately not shown on the lamp;
capture continues with the others. The trigger box latches it and reports
`FAULT 1` in `/trigger/status` until the toggle is next moved, and each
Pi's log says when and why.

Each Pi also shows its own state on its on-board LEDs (visible through the
enclosure); find the Pi that dropped out by looking for one without a
steady green:

| Green (ACT) | Red (PWR) | Meaning |
|---|---|---|
| steady | off | recording |
| blinking | off | agent starting / preview mode |
| off | blinking | waiting for the SSD |
| off | steady | stopped: SSD full, error, or agent not running |

During the first seconds of boot the LEDs show the kernel defaults (red
on, green flickering with SD activity). The red LED no longer flags
under-voltage; the agent logs `vcgencmd get_throttled` instead.

Each Pi records to one SSD mounted at `/media/pi/HIKSEMI`
(`scripts/deploy_prepare_ssd.ps1`). A Pi without a large enough disk can
pool several instead (e13: four 250 GB SSDs) with
`scripts/setup_ssd_pool.sh`: each disk is ext4 `HIKSEMI1..N` at
`/mnt/hiksemiN`, pooled by mergerfs, so a failed disk loses only its own
bursts. On a PC, ingest each disk's `pool` directory
(`construct/ingest_ssd.sh ... <disk>/pool`) into the same collection.
Bus-powered disks share the Pi's USB supply: `scripts/preflight.ps1` lists
every disk and flags under-voltage (THR column).

Each Pi starts a new session directory when no trigger has arrived for
5 min (`SESSION_GAP_S`), i.e. one directory per drying run between
toggle ON and OFF. The agent runs `stty -hupcl` on the serial port so an
agent restart does not reset the Uno; a reset loses only the burst phase.

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
for `burst` → `fps × burst` hardware-synchronized frames per burst
(firmware defaults 30 fps × 2 s / 120 s; values in EEPROM take precedence).
Each Pi buffers the frames in RAM and writes one npz file per burst
(`<host>_burstNNNN_<timestamp>.npz`) during the idle time between bursts.
The first tiny burst after trigger start is warm-up (pipeline drop
frames) — ignore it in analysis.

With `CAPTURE_RAW = True` (default, for spectral measurements) the npz
contains the linear 10-bit sensor data, bypassing gamma/ALSC/denoise:
`frames_raw` (N,1088,stride) CSI2-packed uint8, `timestamps`,
`raw_format`, `raw_shape`, `black_levels` (16-bit scale), and
`analogue_gain`. Load with `analyze/raw10.py`:

```python
from raw10 import load_burst
frames, ts, meta = load_burst("e00_burst0002_....npz")
# frames: (N,1088,1456) uint16, linear, black level subtracted
```

Constraints: exposure < 1/fps (e.g. <33 ms at 30 fps, <15 ms at 60 fps);
max 360 frames/burst (RAM); storage ≈ 1.98 MB × fps × burst_s per burst
per camera (256 GB SSD ≈ 22 h at defaults: 30 fps × 3 s / 60 s period).
`analyze/play_bursts.py` plays a burst from all 16 cameras as a 4x4
mosaic (handles both raw and 8-bit npz).

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
