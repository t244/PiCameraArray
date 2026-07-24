/*
 * exposure_trigger.ino - External trigger generator for IMX296 camera array
 *
 * The trigger line is normally HIGH; each trigger is an active-LOW pulse
 * whose width equals the sensor exposure time (IMX296 trigger mode).
 *
 * Serial protocol (115200 baud, newline-terminated commands):
 *   E<us>   Set exposure time in microseconds (saved to EEPROM)  e.g. "E10000"
 *   T<ms>   Set trigger period in milliseconds (saved to EEPROM) e.g. "T30000"
 *   S       Start triggering
 *   P       Pause (stop) triggering
 *   R       Report status: "EXP <us> PERIOD <ms> RUNNING <0|1>"
 *
 * Headless fail-safe: if no serial command is received within
 * AUTO_START_DELAY_MS after power-on, triggering starts automatically
 * using the values stored in EEPROM. This preserves fully headless
 * operation (power-on -> capture) while allowing configuration from
 * the master Pi when connected.
 */

#include <EEPROM.h>

const int SYNC_PIN = 13;  // Output pin (built-in LED for visual feedback)

// Fail-safe auto start delay (gives Pis time to boot; was STARTUP_DELAY_MS)
const unsigned long AUTO_START_DELAY_MS = 120000;  // 2 minutes

// Defaults used when EEPROM is uninitialized
const unsigned long DEFAULT_EXPOSURE_US = 10000;   // 10 ms
const unsigned long DEFAULT_PERIOD_MS   = 30000;   // 30 s

// EEPROM layout
const uint32_t EEPROM_MAGIC = 0xC0DE5A01;
struct Settings {
  uint32_t magic;
  uint32_t exposure_us;
  uint32_t period_ms;
};

unsigned long exposure_us = DEFAULT_EXPOSURE_US;
unsigned long period_ms   = DEFAULT_PERIOD_MS;

bool running = false;
bool commandReceived = false;   // any serial command disables auto-start
unsigned long lastTriggerTime = 0;

String rxBuffer = "";

void loadSettings() {
  Settings s;
  EEPROM.get(0, s);
  if (s.magic == EEPROM_MAGIC && s.exposure_us > 0 && s.period_ms > 0) {
    exposure_us = s.exposure_us;
    period_ms   = s.period_ms;
  }
}

void saveSettings() {
  Settings s;
  s.magic = EEPROM_MAGIC;
  s.exposure_us = exposure_us;
  s.period_ms   = period_ms;
  EEPROM.put(0, s);  // EEPROM.put only writes changed bytes
}

// Generate one active-LOW pulse of exposure_us microseconds.
// delayMicroseconds() is only accurate up to ~16383 us, so split
// long exposures into a millisecond part and a microsecond part.
void firePulse() {
  unsigned long us = exposure_us;
  digitalWrite(SYNC_PIN, LOW);
  if (us > 16000) {
    unsigned long ms_part = us / 1000;
    unsigned int  us_part = us % 1000;
    delay(ms_part);
    if (us_part > 0) delayMicroseconds(us_part);
  } else {
    delayMicroseconds(us);
  }
  digitalWrite(SYNC_PIN, HIGH);
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  commandReceived = true;

  char c = cmd.charAt(0);
  String arg = cmd.substring(1);

  switch (c) {
    case 'E': case 'e': {
      long v = arg.toInt();
      if (v >= 30 && v <= 10000000L) {   // 30 us .. 10 s
        exposure_us = (unsigned long)v;
        saveSettings();
        Serial.print(F("OK E "));
        Serial.println(exposure_us);
      } else {
        Serial.println(F("ERR E range 30..10000000"));
      }
      break;
    }
    case 'T': case 't': {
      long v = arg.toInt();
      if (v >= 100 && v <= 3600000L) {   // 100 ms .. 1 h
        period_ms = (unsigned long)v;
        saveSettings();
        Serial.print(F("OK T "));
        Serial.println(period_ms);
      } else {
        Serial.println(F("ERR T range 100..3600000"));
      }
      break;
    }
    case 'S': case 's':
      running = true;
      lastTriggerTime = millis();
      Serial.println(F("OK S"));
      break;
    case 'P': case 'p':
      running = false;
      Serial.println(F("OK P"));
      break;
    case 'R': case 'r':
      Serial.print(F("EXP "));
      Serial.print(exposure_us);
      Serial.print(F(" PERIOD "));
      Serial.print(period_ms);
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
    lastTriggerTime = now;
  }

  // --- Periodic trigger ---
  if (running && (now - lastTriggerTime >= period_ms)) {
    firePulse();
    lastTriggerTime = now;
  }
}
