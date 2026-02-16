const int SYNC_PIN = 13;  // Output pin (using built-in LED for visual feedback)
const unsigned long PERIOD_MS = 30000;
const unsigned long STARTUP_DELAY_MS = 90000;  // 90 seconds wait

// Parametric exposure time in microseconds
unsigned long exposure_time = 10000;
unsigned long lastTriggerTime = 0;
bool startupComplete = false;

void setup() {
  pinMode(SYNC_PIN, OUTPUT);
  digitalWrite(SYNC_PIN, HIGH);  // Start with signal normally ON
}

void loop() {
  unsigned long currentTime = millis();
  
  // Wait for startup delay before beginning triggers
  if (!startupComplete) {
    if (currentTime >= STARTUP_DELAY_MS) {
      startupComplete = true;
      lastTriggerTime = currentTime;  // Reset trigger timing
    }
    return;  // Skip trigger logic until startup complete
  }
  
  // Check if period has elapsed
  if (currentTime - lastTriggerTime >= PERIOD_MS) {
    // Generate sync pulse: go LOW for exposure_time
    digitalWrite(SYNC_PIN, LOW);
    delayMicroseconds(exposure_time);
    digitalWrite(SYNC_PIN, HIGH);  // Return to normally ON
    
    lastTriggerTime = currentTime;
  }
}