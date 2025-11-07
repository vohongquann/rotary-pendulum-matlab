#include <Arduino.h>
#include "encoder.h"
#include "control.h"
#include "common_utils.h"


// ======= Control thresholds =======
constexpr float SWING_UP_THRESHOLD = 2.8f;       // radians, ~160°
constexpr float BALANCE_THRESHOLD  = 0.3f;       // radians, ~17°

// ======= State variables =======
bool swing_up_mode = true;
unsigned long balance_enter_time = 0;
unsigned long swingup_enter_time  = 0;

// Setpoints
constexpr float sp_alpha = (float)M_PI;
constexpr float sp_theta = 0.0f;

// Low-pass filter for control signal
constexpr float ALPHA = 0.3f;
float control_signal_filtered = 0.0f;
float action_prev = 0;

bool reset_done_flag = false;

// Shared buffer for JSON message
char stateBuf[150];

// Send JSON state (with timestamp and reset flag)
void sendStateJSON(float theta, float theta_dot,
                   float alpha, float alpha_dot,
                   bool mode, float action) {
  unsigned long ts = millis(); // time since ESP32 started

  int len = snprintf(stateBuf, sizeof(stateBuf),
    "{\"mode\":%d,\"theta\":%.3f,\"theta_dot\":%.3f,"
    "\"alpha\":%.3f,\"alpha_dot\":%.3f,"
    "\"action_prev\":%.3f,"
    "\"reset_done\":%d}\n",
    mode ? 0 : 1,
    theta, theta_dot,
    alpha, alpha_dot,
    action,
    reset_done_flag ? 1 : 0
  );

  Serial.write((uint8_t*)stateBuf, len);

  // Clear reset flag after sending
  if (reset_done_flag) {
    reset_done_flag = false;
  }
}


// ---- Global variables for reset/action timing ----
float last_action = 0.0f;
bool  in_reset_mode = false;
unsigned long reset_start_time = 0;

// Loop timing (not used actively here)
unsigned long last_loop_time     = 0;
unsigned long action_timer_start = 0;
bool        action_timer_on     = false;

unsigned long last_active_time = 0;
constexpr unsigned long TIMEOUT_MS = 1.5 * 60 * 1000;  // 1.5 minutes


void processSerialInput() {
  String input = Serial.readStringUntil('\n');
  if (input.length() == 0) return;

  if (input == "RESET") {
    in_reset_mode      = true;
    reset_start_time   = millis();
  } else {
    float val = 0.0f;
    if (sscanf(input.c_str(), "%f", &val) == 1) {
      last_action      = val;
      in_reset_mode    = false;
      action_timer_on  = false;  // reset pulse state
    }
  }
}

void performReset() {
  controlMotor(0);
  noInterrupts();
    pulseCount_theta = pulseCount_alpha = 0;
  interrupts();
  last_action = control_signal_filtered = action_prev = 0.0f;
  swing_up_mode = true;
  balance_enter_time = swingup_enter_time = 0;

  // Mark reset completion
  reset_done_flag = true;
}

float receiveAction() {
  return last_action;
}

void setup() {
  Serial.begin(921600);

  pinMode(ENCODER_A_1, INPUT_PULLUP);
  pinMode(ENCODER_B_1, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A_1), readEncoder1, CHANGE);

  pinMode(ENCODER_A_2, INPUT_PULLUP);
  pinMode(ENCODER_B_2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A_2), readEncoder2, CHANGE);

  ledcAttach(pwmPin_A, pwmFreq, pwmResolution);
  ledcAttach(pwmPin_B, pwmFreq, pwmResolution);
  ledcWriteChannel(pwmChannel_A, 0);
  ledcWriteChannel(pwmChannel_B, 0);
  ledcWrite(pwmChannel_A, 0);
  ledcWrite(pwmChannel_B, 0);
}

void loop() {
  processSerialInput();

  // Read encoder pulses
  int pulse_theta = pulseCount_theta;
  int pulse_alpha = pulseCount_alpha;

  float theta     = wrapToPi(readTheta(pulse_theta, TYPE_RAD));
  float alpha     = wrapToPi(readAlpha(pulse_alpha, TYPE_RAD));
  float theta_dot = readAngleDot(1, pulse_theta, TYPE_RAD);
  float alpha_dot = readAngleDot(2, pulse_alpha, TYPE_RAD);

  float action = 0.0f; // Will be assigned to action_prev

  if (in_reset_mode) {
    unsigned long dt = millis() - reset_start_time;
    if (dt < 12000) {
      // Pull alpha to 0
      action      = 0; // placeholder, PID optional
      last_action = action;
      controlMotor(action);
      Serial.println("Resetting...");
    } else {
      // Reset completed
      performReset();
      in_reset_mode      = false;
      swingup_enter_time = millis();
      theta = theta_dot = alpha = alpha_dot = 0;
    }
  }
  else {
      action = receiveAction();
  }

  controlMotor(action); 

  // Send JSON state (action_prev = just executed action)
  sendStateJSON(theta, theta_dot, alpha, alpha_dot, swing_up_mode, action_prev);

  // Update last active time if system is active or resetting
  if (in_reset_mode || action != 0.0f) {
    last_active_time = millis();
  }

  // Timeout check
  if ((millis() - last_active_time) > TIMEOUT_MS) {
    Serial.println(" Timeout over 2 minutes. Auto-reset.");
    in_reset_mode = true;
    reset_start_time = millis();
    last_active_time = millis();  // prevent continuous reset loop
  }
}
