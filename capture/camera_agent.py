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
import queue
import concurrent.futures
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

try:
    import simplejpeg  # picamera2 dependency; used for capture thumbnails
except ImportError:
    simplejpeg = None


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

CAPTURE_WAIT_CHUNK = 1.0     # polling slice; mode-switch latency upper bound
TRIGGER_TIMEOUT = 600.0      # warn if no trigger for this long

# Burst buffering: frames are accumulated in RAM during a trigger burst
# and flushed to disk (npz) in the idle time between bursts.
BURST_GAP_S = 0.8            # no frame for this long = burst has ended
MAX_BURST_FRAMES = 360       # RAM safety cap (~710 MB raw-packed full res)
CAPTURE_BUFFER_COUNT = 16    # camera driver buffers for burst capture

# Save the 10-bit packed raw stream (linear, no gamma/ALSC/denoise) for
# spectral measurements. False = save the 8-bit Y plane instead.
CAPTURE_RAW = True
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
        self._pending_job = None
        self.last_jpeg = None            # thumbnail of last captured frame
        self.last_capture_time = None

        # burst buffering
        self.burst_count = 0
        self._burst_frames = []
        self._burst_times = []
        self._burst_meta = None          # per-burst sensor metadata
        self._last_y_small = None        # half-res Y for thumbnails
        self._last_frame_mono = 0.0
        self._raw_format = None
        self._writer_queue = queue.Queue()
        threading.Thread(target=self._writer_loop, daemon=True).start()

        # preview state
        self.exposure_us = DEFAULT_EXPOSURE_US
        self.gain = DEFAULT_GAIN
        self.focus_value = None
        self.actual_exposure_us = None
        self._focus_thread = None
        self._focus_stop = threading.Event()

    # ----- mode switching (main thread) -----

    def request_mode(self, mode: str):
        log.info(f"Mode change requested: {mode}")
        with self.mode_lock:
            self.desired_mode = mode

    def _close_camera(self):
        if self._burst_frames:
            self._flush_burst()  # don't lose a partial burst on mode switch
        self._pending_job = None
        self._focus_stop.set()
        if self._focus_thread is not None:
            self._focus_thread.join(timeout=3)
            self._focus_thread = None
        if self.picam2 is not None:
            # Cancel pending capture_request jobs first: without a trigger
            # they never complete, and stop() would queue behind them
            # (this caused minutes-long mode-switch delays).
            try:
                self.picam2.cancel_all_and_flush()
            except Exception as e:
                log.debug(f"cancel_all_and_flush: {e}")
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
        self.burst_count = 0
        self._burst_frames = []
        self._burst_times = []
        self.no_trigger_elapsed = 0.0
        self.picam2 = Picamera2()
        # Video configuration: main YUV420 for thumbnails; raw stream
        # (10-bit packed, linear) is what gets saved when CAPTURE_RAW.
        streams = {
            "main": {"size": (IMAGE_WIDTH, IMAGE_HEIGHT), "format": "YUV420"},
            "buffer_count": CAPTURE_BUFFER_COUNT,
        }
        if CAPTURE_RAW:
            streams["raw"] = {"size": (IMAGE_WIDTH, IMAGE_HEIGHT)}
        config = self.picam2.create_video_configuration(**streams)
        self.picam2.configure(config)
        self._raw_format = (
            self.picam2.camera_configuration().get("raw") or {}
        ).get("format") if CAPTURE_RAW else None
        self._burst_meta = None
        # Disable all per-frame automatics for quantitative capture:
        # exposure is fixed by the trigger pulse width; gain must be fixed
        # too, otherwise AGC ramps brightness over the first frames and
        # drifts with scene changes. Uses the gain set in preview mode.
        self.picam2.set_controls({
            "AeEnable": False,
            "AnalogueGain": float(self.gain),
            "NoiseReductionMode": 0,  # off - keep pixel values unfiltered
        })
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
        frame_us = max(int(self.exposure_us) + 1000, 33333)
        self.picam2.set_controls({
            "AeEnable": False,
            "FrameDurationLimits": (frame_us, frame_us),
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
        # Extend frame duration for long exposures (default video config
        # caps exposure at the ~33ms frame time otherwise)
        frame_us = max(int(self.exposure_us) + 1000, 33333)
        self.picam2.set_controls({
            "AeEnable": False,
            "FrameDurationLimits": (frame_us, frame_us),
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

    # ----- capture loop (async job + polling; signal-free) -----

    def capture_one(self):
        """Poll for triggered frames in CAPTURE_WAIT_CHUNK slices.

        Frames arriving in a burst are appended to an in-RAM buffer; when
        no frame arrives for BURST_GAP_S the burst is considered finished
        and the buffer is queued for writing (npz) by the writer thread.
        """
        try:
            if self._pending_job is None:
                self._pending_job = self.picam2.capture_request(wait=False)
            request = self._pending_job.get_result(timeout=CAPTURE_WAIT_CHUNK)
            self._pending_job = None
        except (TimeoutError, concurrent.futures.TimeoutError):
            # No frame in this slice: flush if a burst just ended
            if self._burst_frames and (
                    time.monotonic() - self._last_frame_mono > BURST_GAP_S):
                self._flush_burst()
            self.no_trigger_elapsed += CAPTURE_WAIT_CHUNK
            if self.no_trigger_elapsed >= TRIGGER_TIMEOUT:
                log.warning(f"No trigger received in {TRIGGER_TIMEOUT:.0f}s")
                self.no_trigger_elapsed = 0.0
            return
        except Exception as e:
            self._pending_job = None
            log.error(f"Capture failed: {e}")
            time.sleep(1)
            return

        try:
            self.no_trigger_elapsed = 0.0
            arr = request.make_array("main")
            if CAPTURE_RAW:
                frame = request.make_array("raw")  # (H, stride) packed uint8
            if self._burst_meta is None:
                md = request.get_metadata()
                self._burst_meta = {
                    "black_levels": md.get("SensorBlackLevels"),
                    "analogue_gain": md.get("AnalogueGain"),
                }
            request.release()
            # YUV420 array is (H*3/2, W); the Y plane is the first H rows.
            # Keep a half-res copy of the latest Y for the /last.jpg thumbnail
            self._last_y_small = np.ascontiguousarray(
                arr[:IMAGE_HEIGHT:2, :IMAGE_WIDTH:2])
            if not CAPTURE_RAW:
                frame = np.ascontiguousarray(
                    arr[:IMAGE_HEIGHT, :IMAGE_WIDTH])
            self._burst_frames.append(frame)
            self._burst_times.append(
                datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
            self._last_frame_mono = time.monotonic()
            self.capture_count += 1
            self.last_capture_time = datetime.now().isoformat(
                timespec="seconds")

            if len(self._burst_frames) >= MAX_BURST_FRAMES:
                log.warning("Burst buffer full - flushing early")
                self._flush_burst()
        except Exception as e:
            log.error(f"Frame handling failed: {e}")

    def _flush_burst(self):
        """Move the RAM buffer to the writer queue as one npz file."""
        if not self._burst_frames:
            return
        frames = np.stack(self._burst_frames)
        times = np.array(self._burst_times)
        meta = self._burst_meta or {}
        self._burst_frames = []
        self._burst_times = []
        self._burst_meta = None
        self.burst_count += 1

        fname = (f"{self.hostname}_burst{self.burst_count:04d}"
                 f"_{times[0]}.npz")
        path = os.path.join(self.session_dir, fname)

        if CAPTURE_RAW:
            arrays = {
                "frames_raw": frames,  # (N, H, stride) CSI2-packed 10bit
                "timestamps": times,
                "raw_format": np.array(self._raw_format or ""),
                "raw_shape": np.array([IMAGE_HEIGHT, IMAGE_WIDTH]),
                # SensorBlackLevels are on a 16-bit scale (>>6 for 10-bit)
                "black_levels": np.array(meta.get("black_levels") or []),
                "analogue_gain": np.array(meta.get("analogue_gain") or 0.0),
            }
        else:
            arrays = {"frames": frames, "timestamps": times}

        self._writer_queue.put((path, arrays))
        log.info(f"Burst {self.burst_count}: {len(times)} frames "
                 f"({frames.nbytes / 1e6:.0f} MB) queued -> {fname}")

        if self._last_y_small is not None:
            self._update_thumbnail(self._last_y_small)
        self._health_check()

    def _writer_loop(self):
        """Background thread: write queued bursts to disk."""
        while True:
            path, arrays = self._writer_queue.get()
            try:
                t0 = time.monotonic()
                np.savez(path, **arrays)
                nbytes = sum(a.nbytes for a in arrays.values()
                             if isinstance(a, np.ndarray))
                log.info(f"Saved {os.path.basename(path)} "
                         f"({nbytes / 1e6:.0f} MB "
                         f"in {time.monotonic() - t0:.1f}s)")
            except Exception as e:
                log.error(f"Write failed for {path}: {e}")
            finally:
                self._writer_queue.task_done()

    def _update_thumbnail(self, frame):
        """Store a half-resolution JPEG of a (2D) frame for /last.jpg."""
        if simplejpeg is None:
            return
        try:
            thumb = np.ascontiguousarray(frame[::2, ::2])
            thumb = np.ascontiguousarray(
                np.repeat(thumb[:, :, None], 3, axis=2))
            self.last_jpeg = simplejpeg.encode_jpeg(
                thumb, quality=80, colorspace="BGR")
        except Exception as e:
            log.debug(f"thumbnail failed: {e}")

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
            "burst_count": self.burst_count,
            "buffering": len(self._burst_frames),
            "last_capture_time": self.last_capture_time,
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
        self.path = self.path.split("?", 1)[0]  # ignore query string
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

        elif self.path == "/last.jpg":
            if manager.last_jpeg is None:
                self._send_json({"error": "no capture yet"}, 404)
                return
            body = manager.last_jpeg
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

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
                elif self.path == "/trigger/fps":
                    resp = arduino.command(f"F{int(body['fps'])}")
                elif self.path == "/trigger/burst":
                    resp = arduino.command(f"B{int(body['burst_ms'])}")
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
    if os.geteuid() != 0:
        log.error("Root required for trigger_mode control - "
                  "run with: sudo python3 camera_agent.py")
        return 1

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
