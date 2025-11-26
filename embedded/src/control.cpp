#include "control.h"

// Hàm trả về dấu của giá trị (1 nếu >=0, -1 nếu <0)
int sign(float value) {
    return (value >= 0) ? 1 : -1;
}

// Hàm điều khiển động cơ dựa trên giá trị PWM và chiều quay
// Nếu pwmValue > 0 động cơ quay thuận, < 0 quay nghịch, = 0 dừng
void controlMotor(float volt) {
    float pwmValue = volt * 65535.0f / 12.0f; 
    int direction = sign(pwmValue);
    int pwm = abs((int)pwmValue);
    pwm = constrain(pwm, 0, 65535); 
    if (direction > 0) {
        ledcWrite(pwmPin_A, pwm);
        ledcWrite(pwmPin_B, 0);
    } else if (direction < 0) {
        ledcWrite(pwmPin_A, 0);
        ledcWrite(pwmPin_B, pwm);
    } else {
        ledcWrite(pwmPin_A, 0);
        ledcWrite(pwmPin_B, 0);
    }
}


