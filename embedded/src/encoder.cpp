#include "encoder.h"

// ==================== Biến toàn cục lưu giá trị xung ====================
volatile int pulseCount_theta = 0;
volatile int pulseCount_alpha = 0;

// ==================== Hàm xử lý ngắt ngoài Encoder ====================
// ==================== Macro đọc encoder ====================
#define READ_ENCODER(A, B, count) count += (digitalRead(A) == digitalRead(B)) ? 1 : -1

void IRAM_ATTR readEncoder1() { 
  READ_ENCODER(ENCODER_A_1, ENCODER_B_1, pulseCount_theta); 
}

void IRAM_ATTR readEncoder2() { 
  READ_ENCODER(ENCODER_A_2, ENCODER_B_2, pulseCount_alpha); 
}

// ==================== Đọc giá trị góc từ Encoder ====================
float readTheta(int pulseCount, uint8_t type) {
    return (type == TYPE_RAD) ? pulse2Radian(pulseCount) : pulse2Degree(pulseCount);
}

float readAlpha(int pulseCount, uint8_t type) {
    return (type == TYPE_RAD) ? pulse2Radian(pulseCount) : pulse2Degree(pulseCount);
}

// ==================== Đọc vận tốc góc ====================
float readAngleDot(uint8_t motorID, int pulseCount, uint8_t type) {
    static int lastPulse_motor1 = 0, lastPulse_motor2 = 0;
    static unsigned long lastTime_motor1 = 0, lastTime_motor2 = 0;

    unsigned long now = millis();
    float deltaTime;
    int deltaPulse;

    if (motorID == 1) {
        deltaTime = (now - lastTime_motor1) / 1000.0;
        deltaPulse = pulseCount - lastPulse_motor1;
        lastTime_motor1 = now;
        lastPulse_motor1 = pulseCount;
    } else {
        deltaTime = (now - lastTime_motor2) / 1000.0;
        deltaPulse = pulseCount - lastPulse_motor2;
        lastTime_motor2 = now;
        lastPulse_motor2 = pulseCount;
    }

    float velocity = (type == TYPE_RAD) ? pulse2Radian(deltaPulse) / deltaTime : pulse2Degree(deltaPulse) / deltaTime;
    return velocity;
}
