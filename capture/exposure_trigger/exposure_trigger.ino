/*
 * exposure_trigger.ino - External trigger generator for IMX296 camera array
 *
 * The trigger line is normally HIGH; each trigger is an active-LOW pulse
 * whose width equals the sensor exposure time (IMX296 trigger mode).
 *
 * Burst mode: every PERIOD, a train of pulses is generated at FPS rate
 * for BURST milliseconds (e.g. 30 fps x 3000 ms = 90 frames), so all 16
 * cameras stay frame-synchronized for video analysis.
 *
 * Serial protocol (115200 baud, newline-terminated commands):
 *   E<us>   Exposure time / pulse width in microseconds     e.g. "E10000"
 *   T<ms>   Period between burst starts in milliseconds     e.g. "T60000"
 *   F<fps>  Frame rate within a burst (1..60)               e.g. "F30"
 *   B<ms>   Burst duration in milliseconds                  e.g. "B3000"
 *   S       Start triggering
 *   P       Pause (stop) triggering
 *   R       Report: "EXP <us> PERIOD <ms> FPS <fps> BURST <ms> RUNNING <0|1>"
 * All setting commands are saved to EEPROM.
 *
 * Note: during a burst (a few seconds) serial commands are not processed.
 *
 * Headless fail-safe: if no serial command is received within
 * AUTO_START_DELAY_MS after power-on, triggering starts automatically
 * using the EEPROM values.
 */

#include <EEPROM.h>

const int SYNC_PIN = 13;  // Output pin (built-in LED for visual feedback)

const unsigned long AUTO_START_DELAY_MS = 120000;  // 2 minutes

// Defaults used when EEPROM is uninitialized
const unsigned long DEFAULT_EXPOSURE_US = 10000;   // 10 ms
const unsigned long DEFAULT_PERIOD_MS   = 60000;   // 60 s between bursts
const unsigned long DEFAULT_FPS         = 30;      // frames/s within burst
const unsigned long DEFAULT_BURST_MS    = 3000;    // 3 s burst

// EEPROM layout (magic bumped: layout changed from single-shot version)
const uint32_t EEPROM_MAGIC = 0xC0DE5A02;
struct Settings {
  uint32_t magic;
  uint32_t exposure_us;
  uint32_t period_ms;
  uint32_t fps;
  uint32_t burst_ms;
};

unsigned long exposure_us = DEFAULT_EXPOSURE_US;
unsigned long period_ms   = DEFAULT_PERIOD_MS;
unsigned long fps         = DEFAULT_FPS;
unsigned long burst_ms    = DEFAULT_BURST_MS;

bool running = false;
bool commandReceived = false;   // any serial command disables auto-start
unsigned long lastBurstTime = 0;

String rxBuffer = "";

void loadSettings() {
  Settings s;
  EEPROM.get(0, s);
  if (s.magic == EEPROM_MAGIC &&
      s.exposure_us > 0 && s.period_ms > 0 &&
      s.fps >= 1 && s.fps <= 60 && s.burst_ms > 0) {
    exposure_us = s.exposure_us;
    period_ms   = s.period_ms;
    fps         = s.fps;
    burst_ms    = s.burst_ms;
  }
}

void saveSettings() {
  Settings s;
  s.magic = EEPROM_MAGIC;
  s.exposure_us = exposure_us;
  s.period_ms   = period_ms;
  s.fps         = fps;
  s.burst_ms    = burst_ms;
  EEPROM.put(0, s);  // EEPROM.put only writes changed bytes
}

// One active-LOW pulse of exposure_us microseconds.
// delayMicroseconds() is only accurate up to ~16383 us.
void firePulse() {
  unsigned long us = exposure_us;
  digitalWrite(SYNC_PIN, LOW);
  if (us > 16000) {
    delay(us / 1000);
    if (us % 1000 > 0) delayMicroseconds(us % 1000);
  } else {
    delayMicroseconds(us);
  }
  digitalWrite(SYNC_PIN, HIGH);
}

// Pulse train: n frames at fps. Blocking (a few seconds).
void fireBurst() {
  unsigned long interval_us = 1000000UL / fps;
  // If exposure doesn't fit in the frame interval, stretch the interval
  if (exposure_us + 500 > interval_us) interval_us = exposure_us + 500;

  unsigned long n = burst_ms * fps / 1000UL;
  if (n < 1) n = 1;

  for (unsigned long i = 0; i < n; i++) {
    unsigned long t0 = micros();
    firePulse();
    // Busy-wait the remainder of the frame interval (handles micros wrap)
    while ((unsigned long)(micros() - t0) < interval_us) { }
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  commandReceived = true;

  char c = cmd.charAt(0);
  long v = cmd.substring(1).toInt();

  switch (c) {
    case 'E': case 'e':
      if (v >= 30 && v <= 10000000L) {          // 30 us .. 10 s
        exposure_us = (unsigned long)v;
        saveSettings();
        Serial.print(F("OK E ")); Serial.println(exposure_us);
      } else Serial.println(F("ERR E range 30..10000000"));
      break;
    case 'T': case 't':
      if (v >= 1000 && v <= 3600000L) {         // 1 s .. 1 h
        period_ms = (unsigned long)v;
        saveSettings();
        Serial.print(F("OK T ")); Serial.println(period_ms);
      } else Serial.println(F("ERR T range 1000..3600000"));
      break;
    case 'F': case 'f':
      if (v >= 1 && v <= 60) {                  // sensor limit ~60 fps
        fps = (unsigned long)v;
        saveSettings();
        Serial.print(F("OK F ")); Serial.println(fps);
      } else Serial.println(F("ERR F range 1..60"));
      break;
    case 'B': case 'b':
      if (v >= 100 && v <= 30000L) {            // 0.1 s .. 30 s
        burst_ms = (unsigned long)v;
        saveSettings();
        Serial.print(F("OK B ")); Serial.println(burst_ms);
      } else Serial.println(F("ERR B range 100..30000"));
      break;
    case 'S': case 's':
      running = true;
      lastBurstTime = millis() - period_ms;     // first burst immediately
      Serial.println(F("OK S"));
      break;
    case 'P': case 'p':
      running = false;
      Serial.println(F("OK P"));
      break;
    case 'R': case 'r':
      Serial.print(F("EXP "));    Serial.print(exposure_us);
      Serial.print(F(" PERIOD ")); Serial.print(period_ms);
      Serial.print(F(" FPS "));    Serial.print(fps);
      Serial.print(F(" BURST "));  Serial.print(burst_ms);
      Serial.print(F(" RUNNING "));
      Serial.println(running ? 1 : 0);
      break;
    default:
      Serial.println(F("ERR unknown command"));
      break;
  }
}

void setup() {
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, HIGH);  // Signal normally HIGH (idle)
  Serial.begin(115200);
  loadSettings();
}

void loop() {
  // --- Serial command handling (non-blocking) ---
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (rxBuffer.length() > 0) {
        handleCommand(rxBuffer);
        rxBuffer = "";
      }
    } else if (rxBuffer.length() < 32) {
      rxBuffer += ch;
    }
  }

  unsigned long now = millis();

  // --- Headless fail-safe auto start ---
  if (!running && !commandReceived && now >= AUTO_START_DELAY_MS) {
    running = true;
    lastBurstTime = now - period_ms;  // start first burst immediately
  }

  // --- Periodic burst ---
  if (running && (now - lastBurstTime >= period_ms)) {
    lastBurstTime = now;  // period measured start-to-start
    fireBurst();
  }
}
