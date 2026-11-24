/*
 * exposure_trigger.ino - External trigger generator for IMX296 camera array
 *
 * The trigger line is normally HIGH; each trigger is an active-LOW pulse
 * whose width equals the sensor exposure time (IMX296 trigger mode).
 *
 * Burst mode: every PERIOD, a train of pulses is generated at FPS rate
 * for BURST milliseconds (e.g. 30 fps x 2000 ms = 60 frames), so all 16
 * cameras stay frame-synchronized for video analysis.
 *
 * ---------------------------------------------------------------------
 * Unattended operation (factory runs)
 * ---------------------------------------------------------------------
 * Wiring (Arduino Uno):
 *   D13  sync output, fanned out to the 16 camera trigger inputs
 *   D2   capture toggle switch to GND        (closed = ON, INPUT_PULLUP)
 *   D5   bicolor LED, red anode   via 330 ohm (common cathode to GND)
 *   D6   bicolor LED, green anode via 330 ohm
 *   A0   shared READY line: 10 k pull-up to the Uno 3.3V pin, and one
 *        wire to GPIO26 (pin 37) of every Pi. A Pi holds the line LOW
 *        while it is not ready to record (booting, waiting for the SSD,
 *        SSD full, agent stopped) and releases it when it is. The line is
 *        HIGH only when all 16 Pis have released it. Pulled up to 3.3 V,
 *        never 5 V: Pi GPIOs are not 5 V tolerant. Read as analog because
 *        3.3 V is at the Uno's digital HIGH threshold.
 *
 * The toggle is level-triggered: capture runs whenever the system is
 * ready AND the toggle is ON. Turning it ON fires the first burst at once
 * (operator sets the start time); after a power cut capture resumes by
 * itself once every Pi is ready again. Turning it OFF stops after the
 * current burst. A reset of this board loses nothing but the burst phase:
 * every setting is in EEPROM and the state is rebuilt from the inputs.
 *
 * Pilot lamp:
 *   green blinking (1 Hz)      waiting for all Pis to become ready
 *   green steady               ready, toggle OFF
 *   red steady                 recording (between bursts)
 *   red fast blinking          burst in progress
 *   red/green alternating      fatal: not all Pis ready READY_TIMEOUT_MS
 *                              after power-on (recovers if they come up)
 *
 * A Pi dropping out during capture is not shown on the lamp (kept simple
 * for the operators); it is latched and reported as FAULT 1 by "R" until
 * the toggle is next moved, and capture goes on with the other Pis.
 *
 * Serial protocol (115200 baud, newline-terminated commands):
 *   E<us>   Exposure time / pulse width in microseconds     e.g. "E10000"
 *   T<ms>   Period between burst starts in milliseconds     e.g. "T120000"
 *   F<fps>  Frame rate within a burst (1..60)               e.g. "F30"
 *   B<ms>   Burst duration in milliseconds                  e.g. "B2000"
 *   S       Start triggering now (setup override, ignores toggle/ready)
 *   P       Stop triggering now  (setup override)
 *           An override lasts until the toggle is next moved.
 *   R       Report: "EXP <us> PERIOD <ms> FPS <fps> BURST <ms> RUNNING <0|1>
 *            TOGGLE <0|1> LINE <adc> READY <0|1> FAULT <0|1> OVERRIDE <-|S|P>"
 * All setting commands are saved to EEPROM.
 *
 * Note: during a burst (a few seconds) serial commands are not processed.
 */

#include <EEPROM.h>

const int SYNC_PIN      = 13;  // Output pin (built-in LED for visual feedback)
const int TOGGLE_PIN    = 2;
const int LED_RED_PIN   = 5;
const int LED_GREEN_PIN = 6;
const int READY_PIN     = A0;

const int READY_THRESHOLD = 300;          // ADC counts, ~1.5 V at 5 V ref

const unsigned long READY_GRACE_MS    = 60000;   // ignore the line at boot
const unsigned long READY_STABLE_MS   = 10000;   // HIGH this long = ready
const unsigned long READY_TIMEOUT_MS  = 300000;  // not ready by then = fatal
const unsigned long FAULT_LOW_MS      = 3000;    // LOW this long = Pi fault
const unsigned long TOGGLE_DEBOUNCE_MS = 50;

// Defaults used when EEPROM is uninitialized
const unsigned long DEFAULT_EXPOSURE_US = 30000;   // 30 ms
const unsigned long DEFAULT_PERIOD_MS   = 120000;  // 2 min between bursts
const unsigned long DEFAULT_FPS         = 30;      // frames/s within burst
const unsigned long DEFAULT_BURST_MS    = 2000;    // 2 s burst

// EEPROM layout (unchanged from the previous firmware: settings survive)
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

// ----- state -----
bool running = false;            // bursts are being fired
unsigned long lastBurstTime = 0;

bool toggleOn = false;           // debounced switch position
int  toggleRaw = HIGH;
unsigned long toggleChangedAt = 0;

int  lineAdc = 0;
bool lineHigh = false;
unsigned long lineHighSince = 0; // 0 = currently LOW
unsigned long lineLowSince = 0;  // 0 = currently HIGH
bool systemReady = false;        // latched once all Pis were ready
bool faultLatched = false;       // a Pi dropped out since the last toggle move

char overrideMode = '-';             // '-' none, 'S' forced run, 'P' forced stop

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

// ==================== pilot lamp ====================

void setLamp(bool red, bool green) {
  digitalWrite(LED_RED_PIN, red ? HIGH : LOW);
  digitalWrite(LED_GREEN_PIN, green ? HIGH : LOW);
}

bool fatal() {
  return !systemReady && overrideMode != 'S' && millis() >= READY_TIMEOUT_MS;
}

// Cheap enough to call inside the burst busy-wait loop.
void updateLamp(bool inBurst) {
  unsigned long t = millis();

  if (fatal()) {                                   // red/green alternating
    bool r = (t / 250) % 2;
    setLamp(r, !r);
    return;
  }

  if (running) {
    if (inBurst) setLamp((t / 100) % 2, false);    // red fast blink
    else         setLamp(true, false);             // red steady
  } else if (systemReady) {
    setLamp(false, true);                          // green steady
  } else {
    setLamp(false, (t / 500) % 2);                 // green blinking
  }
}

// Delay that keeps the lamp alive.
void lampDelay(unsigned long ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) updateLamp(false);
}

// ==================== trigger output ====================

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

void firePulseTrain(unsigned long n) {
  unsigned long interval_us = 1000000UL / fps;
  // If exposure doesn't fit in the frame interval, stretch the interval
  if (exposure_us + 500 > interval_us) interval_us = exposure_us + 500;
  for (unsigned long i = 0; i < n; i++) {
    unsigned long t0 = micros();
    firePulse();
    // Busy-wait the remainder of the frame interval (handles micros wrap)
    while ((unsigned long)(micros() - t0) < interval_us) updateLamp(true);
  }
}

// Warm-up: the camera pipeline discards its first few frames after
// (re)start, which would otherwise truncate the first real burst.
// Fire a short pulse train to absorb those dropped frames.
const unsigned long WARMUP_PULSES = 10;

// Pulse train: n frames at fps. Blocking (a few seconds).
void fireBurst() {
  unsigned long n = burst_ms * fps / 1000UL;
  if (n < 1) n = 1;
  firePulseTrain(n);
  updateLamp(false);
}

void startRunning() {
  running = true;
  firePulseTrain(WARMUP_PULSES);        // absorb pipeline drop-frames
  lampDelay(2000);
  lastBurstTime = millis() - period_ms; // first real burst immediately
}

// ==================== inputs ====================

void readToggle() {
  int raw = digitalRead(TOGGLE_PIN);
  unsigned long now = millis();
  if (raw != toggleRaw) {
    toggleRaw = raw;
    toggleChangedAt = now;
  }
  bool on = (toggleRaw == LOW);
  if (on != toggleOn && now - toggleChangedAt >= TOGGLE_DEBOUNCE_MS) {
    toggleOn = on;
    overrideMode = '-';          // moving the toggle ends any setup override
    faultLatched = false;    // and acknowledges a past Pi fault
  }
}

void readReadyLine() {
  unsigned long now = millis();
  lineAdc = analogRead(READY_PIN);
  lineHigh = lineAdc > READY_THRESHOLD;
  if (lineHigh) {
    if (lineHighSince == 0) lineHighSince = now ? now : 1;
    lineLowSince = 0;
  } else {
    if (lineLowSince == 0) lineLowSince = now ? now : 1;
    lineHighSince = 0;
  }

  if (!systemReady) {
    if (now >= READY_GRACE_MS && lineHigh &&
        now - lineHighSince >= READY_STABLE_MS) {
      systemReady = true;
    }
  } else if (!lineHigh && now - lineLowSince >= FAULT_LOW_MS) {
    faultLatched = true;     // a Pi is not (or no longer) recording
  }
}

// ==================== serial ====================

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

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
      overrideMode = 'S';
      Serial.println(F("OK S"));
      break;
    case 'P': case 'p':
      overrideMode = 'P';
      Serial.println(F("OK P"));
      break;
    case 'R': case 'r':
      Serial.print(F("EXP "));       Serial.print(exposure_us);
      Serial.print(F(" PERIOD "));   Serial.print(period_ms);
      Serial.print(F(" FPS "));      Serial.print(fps);
      Serial.print(F(" BURST "));    Serial.print(burst_ms);
      Serial.print(F(" RUNNING "));  Serial.print(running ? 1 : 0);
      Serial.print(F(" TOGGLE "));   Serial.print(toggleOn ? 1 : 0);
      Serial.print(F(" LINE "));     Serial.print(lineAdc);
      Serial.print(F(" READY "));    Serial.print(systemReady ? 1 : 0);
      Serial.print(F(" FAULT "));    Serial.print(faultLatched ? 1 : 0);
      Serial.print(F(" OVERRIDE ")); Serial.println(overrideMode);
      break;
    default:
      Serial.println(F("ERR unknown command"));
      break;
  }
}

// ==================== main ====================

void setup() {
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, HIGH);  // Signal normally HIGH (idle)
  pinMode(TOGGLE_PIN, INPUT_PULLUP);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  setLamp(false, false);
  Serial.begin(115200);
  loadSettings();
  toggleRaw = digitalRead(TOGGLE_PIN);
  toggleOn = (toggleRaw == LOW);  // position at power-on counts as settled
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

  readToggle();
  readReadyLine();

  bool shouldRun;
  if (overrideMode == 'S')      shouldRun = true;
  else if (overrideMode == 'P') shouldRun = false;
  else                      shouldRun = systemReady && toggleOn;

  if (shouldRun && !running) startRunning();
  else if (!shouldRun && running) running = false;

  // --- Periodic burst ---
  if (running && (millis() - lastBurstTime >= period_ms)) {
    lastBurstTime = millis();  // period measured start-to-start
    fireBurst();
  }

  updateLamp(false);
}
