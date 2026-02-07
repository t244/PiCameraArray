#!/usr/bin/env python3
"""
External Trigger Capture Module for IMX296 Camera
Supports both standalone script execution and module import

Usage as script:
    python3 trigger_capture.py

Usage as module:
    from trigger_capture import TriggerCapture
    
    capture = TriggerCapture(storage_path="/custom/path")
    capture.start_capture_loop()
"""

import time
import signal
from datetime import datetime
from picamera2 import Picamera2
import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import sys


# ==================== CONFIGURATION ====================

@dataclass
class CaptureConfig:
    """Configuration for triggered capture"""
    storage_path: str = f"/home/pi/PiCameraArray/data/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    image_format: str = "png"
    image_width: int = 1456
    image_height: int = 1088
    trigger_timeout: float = 60.0  # seconds
    buffer_count: int = 1
    max_storage_percent: float = 95.0
    max_temperature: float = 80.0
    temp_warning: float = 75.0
    check_interval: int = 50  # Check health every N captures
    log_file: str = f"{storage_path}/capture.log"
    log_to_console: bool = True


# ==================== TIMEOUT EXCEPTION ====================

class TimeoutException(Exception):
    """Exception raised when capture times out"""
    pass


# ==================== LOGGER SETUP ====================

def setup_logger(config: CaptureConfig) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    if config.log_file:
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
    
    return logger


# ==================== UTILITY FUNCTIONS ====================

def get_storage_usage(path: str) -> float:
    """
    Get storage usage percentage for given path
    
    Args:
        path: Storage path to check
        
    Returns:
        Storage usage as percentage (0-100)
    """
    import shutil
    try:
        stat = shutil.disk_usage(path)
        return (stat.used / stat.total) * 100
    except Exception:
        return 0.0


def ensure_storage_ready(path: str) -> bool:
    """
    Ensure storage directory exists and is writable
    
    Args:
        path: Storage directory path
        
    Returns:
        True if storage is ready, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception:
        return False


def get_cpu_temp() -> Optional[float]:
    """
    Get Raspberry Pi CPU temperature
    
    Returns:
        Temperature in Celsius, or None if unavailable
    """
    import subprocess
    try:
        temp = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
        temp = float(temp.replace('temp=', '').replace("'C\n", ''))
        return temp
    except Exception:
        return None


def get_hostname() -> str:
    """Get system hostname"""
    return os.uname().nodename


# ==================== MAIN CLASS ====================

class TriggerCapture:
    """
    External trigger capture manager for IMX296 camera
    
    This class handles initialization, configuration, and continuous
    capture triggered by external hardware signals.
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize trigger capture system
        
        Args:
            config: CaptureConfig object, uses defaults if None
        """
        self.config = config or CaptureConfig()
        self.logger = setup_logger(self.config)
        self.camera: Optional[Picamera2] = None
        self.hostname = get_hostname()
        self.capture_count = 0
        self.running = False
        self._timeout_occurred = False
    
    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout"""
        self._timeout_occurred = True
        raise TimeoutException()
    
    def initialize_camera(self) -> bool:
        """
        Initialize and configure camera for triggered capture
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing camera...")
            self.logger.info("Trigger mode must be enabled via imx296_trigger tool")
            
            self.camera = Picamera2()
            
            config = self.camera.create_still_configuration(
                main={
                    "size": (self.config.image_width, self.config.image_height)
                },
                buffer_count=self.config.buffer_count
            )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Allow camera to stabilize
            time.sleep(2)
            
            self.logger.info("Camera initialized successfully")
            self.logger.info(f"Timeout: {self.config.trigger_timeout} seconds")
            self.logger.info(f"Resolution: {self.config.image_width}x{self.config.image_height}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_single(self, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Capture a single triggered image
        
        Args:
            filename: Optional custom filename (without path)
            
        Returns:
            Tuple of (success: bool, filepath: Optional[str])
        """
        if not self.camera:
            self.logger.error("Camera not initialized")
            return False, None
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = (f"{self.hostname}_"
                           f"{self.capture_count:06d}_"
                           f"{timestamp}.{self.config.image_format}")
            
            filepath = os.path.join(self.config.storage_path, filename)
            
            # Setup timeout
            self._timeout_occurred = False
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(int(self.config.trigger_timeout))
            
            try:
                # Wait for and capture request (blocking until trigger arrives)
                request = self.camera.capture_request()
                
                # Cancel alarm - we got the frame
                signal.alarm(0)
                
                # Save image
                request.save("main", filepath)
                request.release()
                
                self.logger.info(f"✓ Captured: {filename}")
                self.capture_count += 1
                
                return True, filepath
                
            except TimeoutException:
                # Timeout occurred
                signal.alarm(0)
                self.logger.warning(
                    f"✗ Timeout - no trigger received in {self.config.trigger_timeout}s"
                )
                return False, None
                
        except Exception as e:
            signal.alarm(0)  # Make sure to cancel alarm
            self.logger.error(f"Capture failed: {e}")
            return False, None
    
    def check_system_health(self) -> bool:
        """
        Check system health (storage, temperature)
        
        Returns:
            True if system healthy, False if should stop
        """
        # Check storage
        storage_usage = get_storage_usage(self.config.storage_path)
        if storage_usage > self.config.max_storage_percent:
            self.logger.error(
                f"Storage limit exceeded: {storage_usage:.1f}% > "
                f"{self.config.max_storage_percent}%"
            )
            return False
        
        # Check temperature
        cpu_temp = get_cpu_temp()
        if cpu_temp:
            if cpu_temp > self.config.max_temperature:
                self.logger.critical(
                    f"Critical temperature: {cpu_temp}°C - STOPPING"
                )
                return False
            elif cpu_temp > self.config.temp_warning:
                self.logger.warning(f"High temperature: {cpu_temp}°C")
        
        return True
    
    def start_capture_loop(self) -> int:
        """
        Start continuous triggered capture loop
        
        Returns:
            Total number of captures completed
        """
        self.logger.info("="*50)
        self.logger.info(f"External Trigger Capture - {self.hostname}")
        self.logger.info("="*50)
        
        # Check storage
        if not ensure_storage_ready(self.config.storage_path):
            self.logger.error("Storage not accessible. Exiting.")
            return 0
        
        self.logger.info(f"Storage path: {self.config.storage_path}")
        self.logger.info(f"Storage usage: {get_storage_usage(self.config.storage_path):.1f}%")
        
        # Initialize camera
        if not self.initialize_camera():
            self.logger.error("Camera initialization failed. Exiting.")
            return 0
        
        self.logger.info("="*50)
        self.logger.info("System ready - waiting for trigger pulses")
        self.logger.info("Press Ctrl+C to stop")
        self.logger.info("="*50)
        
        self.running = True
        self.capture_count = 0
        
        try:
            while self.running:
                # Periodic health check
                if self.capture_count % self.config.check_interval == 0 and self.capture_count > 0:
                    if not self.check_system_health():
                        break
                
                # Capture image
                self.capture_single()
        
        except KeyboardInterrupt:
            self.logger.info("Stopped by user")
        
        finally:
            self.stop()
        
        return self.capture_count
    
    def stop(self):
        """Stop capture and cleanup resources"""
        self.running = False
        
        # Cancel any pending alarm
        signal.alarm(0)
        
        self.logger.info(f"Session ended. Total captures: {self.capture_count}")
        
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                self.logger.info("Camera closed")
            except Exception as e:
                self.logger.error(f"Error closing camera: {e}")
    
    def get_stats(self) -> dict:
        """
        Get capture statistics
        
        Returns:
            Dictionary with current statistics
        """
        return {
            'hostname': self.hostname,
            'capture_count': self.capture_count,
            'storage_usage': get_storage_usage(self.config.storage_path),
            'cpu_temp': get_cpu_temp(),
            'running': self.running
        }


# ==================== SIGNAL HANDLER ====================

def signal_handler(sig, frame):
    """Signal handler for graceful shutdown"""
    print("\nShutting down...")
    sys.exit(0)

# ==================== MAIN FUNCTION ====================

def main():
    """
    Main entry point when running as script
    """
    # Create default configuration
    config = CaptureConfig()
    
    # You can customize configuration here if needed
    # config.storage_path = "/custom/path"
    # config.trigger_timeout = 120.0
    
    # Create and run capture system
    capture = TriggerCapture(config)
    
    try:
        capture.start_capture_loop()
    except Exception as e:
        logging.critical(f"Unexpected error: {e}", exc_info=True)
    
    return 0


# ==================== SCRIPT ENTRY POINT ====================

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    exit(main())