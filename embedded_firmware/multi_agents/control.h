#pragma once

/**
 * @file Control.h
 * @brief Khai báo lớp PID, LQR và các hằng số vật lý liên quan đến con lắc.
 *        Đồng thời cung cấp hàm get_state() & get_y() dựa vào encoder.
 */

#include <Arduino.h>
#include "Encoder.h"  // Gọi các hàm đo encoder

// ======================== Cấu trúc trạng thái ========================
struct State {
    double theta;       
    double alpha;       
    double theta_dot;   
    double alpha_dot;   
};

// ======================== Chiều quay ========================
constexpr double ClockWise = 1;  
constexpr double CounterClockWise = -1;                 

// ======================== Chu kỳ lấy mẫu ========================
constexpr double DT = 0.001;  

// ======================== Hàm hệ thống ========================
int sign(float value);
void controlMotor(float pwmValue);

// ======================== Lớp điều khiển PID ========================
class PIDController {
  float kp, ki, kd;
  float error_prev;
  float integral;

public:
  PIDController(float kp_, float ki_, float kd_) : kp(kp_), ki(ki_), kd(kd_), error_prev(0.0f), integral(0.0f) {}

  float compute(float setpoint, float measured) {
    float error = setpoint - measured;
    integral += error;
    float derivative = error - error_prev;
    error_prev = error;
    return kp * error + ki * integral + kd * derivative;
  }
};

