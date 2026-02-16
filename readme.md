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
