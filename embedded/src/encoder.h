#pragma once

#include "Arduino.h"
#include <stdio.h>
#include <math.h>

// ==================== Định nghĩa chân PWM ====================
constexpr uint8_t pwmPin_A = 18;  // GPIO18
constexpr uint8_t pwmPin_B = 19;  // GPIO19

constexpr uint8_t pwmChannel_A = 0;  // Kênh 0
constexpr uint8_t pwmChannel_B = 1;  // Kênh 1

constexpr float pwmFreq = 512;
constexpr uint8_t pwmResolution = 16;

constexpr uint8_t OUTPUT_VOLTAGE_MAX = 12;

// ==================== Định nghĩa chân Encoder ====================
constexpr uint8_t ENCODER_A_1 = 32;
constexpr uint8_t ENCODER_B_1 = 33;

constexpr uint8_t ENCODER_A_2 = 25;
constexpr uint8_t ENCODER_B_2 = 26;

// ==================== Thông số Encoder ====================
constexpr float PPR = 1024;   // Số xung mỗi vòng quay
constexpr float RevD = 360.0;  // Độ trong 1 vòng quay
constexpr float RevR = 2 * M_PI;  // Radian trong 1 vòng quay

// ==================== Hằng số chuyển đổi ====================
constexpr float pulse2DegreeRatio = RevD / PPR;
constexpr float degree2PulseRatio = PPR / RevD;

constexpr float pulse2RadianRatio = RevR / PPR;
constexpr float radian2PulseRatio = PPR / RevR;

constexpr uint8_t TYPE_RAD = 0;
constexpr uint8_t TYPE_DEG = 1;

// ==================== Biến toàn cục lưu xung Encoder ====================
extern volatile int pulseCount_theta;
extern volatile int pulseCount_alpha;

// ==================== Ngắt ngoài đọc xung Encoder ====================
void IRAM_ATTR readEncoder1();
void IRAM_ATTR readEncoder2();

// ==================== Chuyển đổi đơn vị ====================
inline float rad2Degree(float radian) { return radian * (180.0 / M_PI); }
inline float deg2Radian(float degree) { return degree * (M_PI / 180.0); }

inline float pulse2Degree(float pulse) { return pulse * pulse2DegreeRatio; }
inline float degree2Pulse(float degree) { return degree * degree2PulseRatio; }

inline float pulse2Radian(float pulse) { return pulse * pulse2RadianRatio; }
inline float radian2Pulse(float radian) { return radian * radian2PulseRatio; }

// ==================== Đọc giá trị từ Encoder ====================
float readTheta(int pulseCount, uint8_t type);
float readAlpha(int pulseCount, uint8_t type);

// ==================== Đọc tốc độ quay & vận tốc góc ====================
float readAngleDot(uint8_t motorID, int pulseCount, uint8_t type);
