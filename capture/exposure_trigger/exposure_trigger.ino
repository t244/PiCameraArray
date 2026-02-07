const int SYNC_PIN = 13;  // Output pin (using built-in LED for visual feedback)

const unsigned long PERIOD_MS = 100;  // 10 Hz = 100ms period

// Parametric exposure time in microseconds
unsigned long exposure_time = 10000;  // 10ms = 10,000 microseconds (adjust as needed)

unsigned long lastTriggerTime = 0;

void setup() {
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, HIGH);  // Start with signal normally ON
}

void loop() {
  unsigned long currentTime = millis();
  
  // Check if 100ms period has elapsed
  if (currentTime - lastTriggerTime >= PERIOD_MS) {
    // Generate sync pulse: go LOW for exposure_time
    digitalWrite(SYNC_PIN, LOW);
    delayMicroseconds(exposure_time);  // LOW period in microseconds
    digitalWrite(SYNC_PIN, HIGH);  // Return to normally ON
    
    lastTriggerTime = currentTime;
  }
}