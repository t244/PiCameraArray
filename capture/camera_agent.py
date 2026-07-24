#!/usr/bin/env python3
"""
Camera Agent for IMX296 Camera Array

Resident service on each Pi. Boots into 'capture' mode (external trigger,
headless) and exposes an HTTP API for setup-time operation from a laptop:

  GET  /status            JSON status (mode, focus, temp, storage, ...)
  POST /mode              {"mode": "preview" | "capture"}
  GET  /stream.mjpg       MJPEG stream (preview mode only)
  POST /controls          {"exposure_us": int, "gain": float}  (preview only)

If an Arduino is connected via USB serial (auto-detected), additionally:

  POST /trigger/exposure  {"exposure_us": int}   -> "E<us>" (saved to EEPROM)
  POST /trigger/period    {"period_ms": int}     -> "T<ms>" (saved to EEPROM)
  POST /trigger/start                            -> "S"
  POST /trigger/stop                             -> "P"
  GET  /trigger/status                           -> "R"

Modes:
  capture: /sys/module/imx296/parameters/trigger_mode = 1
           still configuration, blocking capture_request loop,
           images saved to SSD (HIKSEMI) or SD fallback
  preview: trigger_mode = 0 (free running)
           video configuration, MJPEG streaming, manual exposure/gain,
           focus metric (variance of Laplacian) computed continuously

Run: sudo python3 camera_agent.py   (root required for trigger_mode sysfs)
"""

import io
import os
import sys
import glob
import json
import time
import signal
import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

try:
    import serial  # pyserial (only needed on the master Pi)
except ImportError:
    serial = None


# ==================== CONFIGURATION ====================

HTTP_PORT = 8000
IMAGE_WIDTH = 1456
IMAGE_HEIGHT = 1088
IMAGE_FORMAT = "png"

PREVIEW_WIDTH = 728          # half resolution keeps 16 streams light
PREVIEW_HEIGHT = 544

TRIGGER_MODE_PATH = "/sys/module/imx296/parameters/trigger_mode"
SSD_MOUNT = "/media/pi/HIKSEMI"
SD_FALLBACK = "/home/pi/PiCameraArray/data"

CAPTURE_WAIT_CHUNK = 5.0     # SIGALRM chunk; mode-switch latency upper bound
TRIGGER_TIMEOUT = 600.0      # warn if no trigger for this long
MAX_STORAGE_PERCENT = 95.0
MAX_TEMPERATURE = 90.0
TEMP_WARNING = 80.0
HEALTH_CHECK_INTERVAL = 50   # captures between health checks

DEFAULT_EXPOSURE_US = 10000
DEFAULT_GAIN = 1.0

SERIAL_BAUD = 115200
SERIAL_GLOBS = ("/dev/ttyACM*", "/dev/ttyUSB*")


# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("camera_agent")


# ==================== UTILITIES ====================

def get_hostname() -> str:
    return os.uname().nodename


def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def get_storage_usage(path: str) -> float:
    import shutil
    try:
        stat = shutil.disk_usage(path)
        return (stat.used / stat.total) * 100
    except Exception:
        return 0.0


def make_session_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.isdir(SSD_MOUNT) and os.path.ismount(SSD_MOUNT):
        path = f"{SSD_MOUNT}/data/{ts}"
    else:
        path = f"{SD_FALLBACK}/{ts}"
    os.makedirs(path, exist_ok=True)
    return path


def set_trigger_mode(enabled: bool):
    """Write imx296 sysfs trigger_mode (requires root)."""
    try:
        with open(TRIGGER_MODE_PATH, "w") as f:
            f.write("1" if enabled else "0")
        log.info(f"trigger_mode = {1 if enabled else 0}")
    except Exception as e:
        log.error(f"Failed to set trigger_mode: {e}")


def focus_metric(gray: np.ndarray) -> float:
    """Variance of Laplacian (numpy only, no OpenCV dependency)."""
    g = gray.astype(np.float32)
    lap = (
        4.0 * g[1:-1, 1:-1]
        - g[:-2, 1:-1] - g[2:, 1:-1]
        - g[1:-1, :-2] - g[1:-1, 2:]
    )
    return float(lap.var())


# ==================== MJPEG STREAMING OUTPUT ====================

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


# ==================== ARDUINO SERIAL (master Pi only) ====================

class ArduinoLink:
    """Persistent serial connection to the trigger Arduino.

    Opened once at startup and kept open: opening the port resets a typical
    Arduino, so reconnecting on every command would interrupt triggering.
    """

    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()
        self.port = None
        if serial is None:
            return
        for pattern in SERIAL_GLOBS:
            matches = sorted(glob.glob(pattern))
            if matches:
                self.port = matches[0]
                break
        if self.port is None:
            return
        try:
            self.ser = serial.Serial(self.port, SERIAL_BAUD, timeout=2)
            time.sleep(2.5)  # Arduino resets on open; wait for boot
            self.ser.reset_input_buffer()
            log.info(f"Arduino connected on {self.port}")
        except Exception as e:
            log.warning(f"Arduino open failed on {self.port}: {e}")
            self.ser = None

    @property
    def available(self) -> bool:
        return self.ser is not None

    def command(self, cmd: str) -> str:
        if not self.available:
            raise RuntimeError("Arduino not connected")
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + "\n").encode())
            self.ser.flush()
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                raise RuntimeError("Arduino did not respond")
            return line


# ==================== CAMERA MANAGER ====================

class CameraManager:
    """Owns the camera. Mode switching runs on the main thread only."""

    MODE_CAPTURE = "capture"
    MODE_PREVIEW = "preview"

    def __init__(self):
        self.picam2 = None
        self.mode = None                      # current mode
        self.desired_mode = self.MODE_CAPTURE
        self.mode_lock = threading.Lock()

        self.stream_output = StreamingOutput()
        self.hostname = get_hostname()

        # capture state
        self.session_dir = None
        self.capture_count = 0
        self.no_trigger_elapsed = 0.0
        self.healthy = True

        # preview state
        self.exposure_us = DEFAULT_EXPOSURE_US
        self.gain = DEFAULT_GAIN
        self.focus_value = None
        self.actual_exposure_us = None
        self._focus_thread = None
        self._focus_stop = threading.Event()

    # ----- mode switching (main thread) -----

    def request_mode(self, mode: str):
        with self.mode_lock:
            self.desired_mode = mode

    def _close_camera(self):
        self._focus_stop.set()
        if self._focus_thread is not None:
            self._focus_thread.join(timeout=3)
            self._focus_thread = None
        if self.picam2 is not None:
            try:
                if self.mode == self.MODE_PREVIEW:
                    self.picam2.stop_recording()
                else:
                    self.picam2.stop()
            except Exception as e:
                log.debug(f"stop error: {e}")
            try:
                self.picam2.close()
            except Exception as e:
                log.debug(f"close error: {e}")
            self.picam2 = None
        time.sleep(0.5)

    def enter_capture(self):
        self._close_camera()
        set_trigger_mode(True)
        self.session_dir = make_session_dir()
        self.capture_count = 0
        self.no_trigger_elapsed = 0.0
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)},
            buffer_count=10,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        self.mode = self.MODE_CAPTURE
        log.info(f"CAPTURE mode - waiting for triggers, saving to {self.session_dir}")

    def enter_preview(self):
        self._close_camera()
        set_trigger_mode(False)
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT)},
        )
        self.picam2.configure(config)
        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": int(self.exposure_us),
            "AnalogueGain": float(self.gain),
        })
        self.picam2.start_recording(JpegEncoder(), FileOutput(self.stream_output))
        self.mode = self.MODE_PREVIEW
        self._focus_stop.clear()
        self._focus_thread = threading.Thread(target=self._focus_loop, daemon=True)
        self._focus_thread.start()
        log.info("PREVIEW mode - MJPEG streaming, manual exposure")

    # ----- preview helpers -----

    def set_controls(self, exposure_us=None, gain=None):
        if self.mode != self.MODE_PREVIEW:
            raise RuntimeError("controls only available in preview mode")
        if exposure_us is not None:
            self.exposure_us = int(exposure_us)
        if gain is not None:
            self.gain = float(gain)
        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": int(self.exposure_us),
            "AnalogueGain": float(self.gain),
        })

    def _focus_loop(self):
        while not self._focus_stop.is_set():
            try:
                arr = self.picam2.capture_array("main")
                # subsample + green-ish channel for speed
                gray = arr[::2, ::2, 1] if arr.ndim == 3 else arr[::2, ::2]
                self.focus_value = focus_metric(gray)
                md = self.picam2.capture_metadata()
                self.actual_exposure_us = md.get("ExposureTime")
            except Exception as e:
                log.debug(f"focus loop error: {e}")
            self._focus_stop.wait(0.5)

    # ----- capture loop (main thread, SIGALRM based) -----

    def _alarm_handler(self, signum, frame):
        raise TimeoutError()

    def capture_one(self):
        """Wait up to CAPTURE_WAIT_CHUNK for a trigger; save frame if received."""
        signal.signal(signal.SIGALRM, self._alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, CAPTURE_WAIT_CHUNK)
        try:
            request = self.picam2.capture_request()
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.no_trigger_elapsed = 0.0

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.hostname}_{self.capture_count:06d}_{ts}.{IMAGE_FORMAT}"
            filepath = os.path.join(self.session_dir, filename)
            request.save("main", filepath)
            request.release()
            self.capture_count += 1
            log.info(f"Captured: {filename}")

            if self.capture_count % HEALTH_CHECK_INTERVAL == 0:
                self._health_check()
        except TimeoutError:
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.no_trigger_elapsed += CAPTURE_WAIT_CHUNK
            if self.no_trigger_elapsed >= TRIGGER_TIMEOUT:
                log.warning(f"No trigger received in {TRIGGER_TIMEOUT:.0f}s")
                self.no_trigger_elapsed = 0.0
        except Exception as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            log.error(f"Capture failed: {e}")
            time.sleep(1)

    def _health_check(self):
        usage = get_storage_usage(self.session_dir)
        if usage > MAX_STORAGE_PERCENT:
            log.error(f"Storage limit exceeded: {usage:.1f}%")
            self.healthy = False
        temp = get_cpu_temp()
        if temp is not None:
            if temp > MAX_TEMPERATURE:
                log.critical(f"Critical temperature: {temp}°C")
                self.healthy = False
            elif temp > TEMP_WARNING:
                log.warning(f"High temperature: {temp}°C")

    # ----- status -----

    def status(self, arduino_available: bool) -> dict:
        return {
            "hostname": self.hostname,
            "mode": self.mode,
            "healthy": self.healthy,
            "capture_count": self.capture_count,
            "session_dir": self.session_dir,
            "storage_usage": round(
                get_storage_usage(self.session_dir or SD_FALLBACK), 1),
            "cpu_temp": get_cpu_temp(),
            "focus": self.focus_value,
            "exposure_us": self.exposure_us,
            "actual_exposure_us": self.actual_exposure_us,
            "gain": self.gain,
            "has_arduino": arduino_available,
        }

    # ----- main loop -----

    def run(self):
        while True:
            with self.mode_lock:
                desired = self.desired_mode
            if desired != self.mode:
                try:
                    if desired == self.MODE_PREVIEW:
                        self.enter_preview()
                    else:
                        self.enter_capture()
                except Exception as e:
                    log.error(f"Mode switch to {desired} failed: {e}")
                    time.sleep(3)
                    continue

            if self.mode == self.MODE_CAPTURE and self.healthy:
                self.capture_one()
            else:
                time.sleep(0.2)


# ==================== HTTP SERVER ====================

manager = CameraManager()
arduino = None  # initialized in main()


class AgentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # silence per-request logging

    # ----- helpers -----

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    # ----- CORS preflight -----

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ----- GET -----

    def do_GET(self):
        if self.path == "/status":
            self._send_json(manager.status(
                arduino.available if arduino else False))

        elif self.path == "/stream.mjpg":
            if manager.mode != CameraManager.MODE_PREVIEW:
                self._send_json({"error": "not in preview mode"}, 409)
                return
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while manager.mode == CameraManager.MODE_PREVIEW:
                    with manager.stream_output.condition:
                        if not manager.stream_output.condition.wait(timeout=5):
                            continue
                        frame = manager.stream_output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/trigger/status":
            if not (arduino and arduino.available):
                self._send_json({"error": "no arduino"}, 404)
                return
            try:
                self._send_json({"response": arduino.command("R")})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "not found"}, 404)

    # ----- POST -----

    def do_POST(self):
        try:
            body = self._read_json()
        except Exception:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        if self.path == "/mode":
            mode = body.get("mode")
            if mode not in (CameraManager.MODE_PREVIEW,
                            CameraManager.MODE_CAPTURE):
                self._send_json({"error": "mode must be preview|capture"}, 400)
                return
            manager.request_mode(mode)
            self._send_json({"ok": True, "requested": mode})

        elif self.path == "/controls":
            try:
                manager.set_controls(
                    exposure_us=body.get("exposure_us"),
                    gain=body.get("gain"),
                )
                self._send_json({"ok": True,
                                 "exposure_us": manager.exposure_us,
                                 "gain": manager.gain})
            except Exception as e:
                self._send_json({"error": str(e)}, 409)

        elif self.path.startswith("/trigger/"):
            if not (arduino and arduino.available):
                self._send_json({"error": "no arduino"}, 404)
                return
            try:
                if self.path == "/trigger/exposure":
                    us = int(body["exposure_us"])
                    resp = arduino.command(f"E{us}")
                elif self.path == "/trigger/period":
                    ms = int(body["period_ms"])
                    resp = arduino.command(f"T{ms}")
                elif self.path == "/trigger/start":
                    resp = arduino.command("S")
                elif self.path == "/trigger/stop":
                    resp = arduino.command("P")
                else:
                    self._send_json({"error": "not found"}, 404)
                    return
                ok = resp.startswith("OK")
                self._send_json({"ok": ok, "response": resp},
                                200 if ok else 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "not found"}, 404)


# ==================== MAIN ====================

def main():
    global arduino
    arduino = ArduinoLink()

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), AgentHandler)
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    log.info(f"HTTP API listening on :{HTTP_PORT}"
             f" (arduino: {'yes' if arduino.available else 'no'})")

    def shutdown(signum, frame):
        log.info("Shutdown requested")
        try:
            manager._close_camera()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    manager.run()  # never returns


if __name__ == "__main__":
    sys.exit(main() or 0)
