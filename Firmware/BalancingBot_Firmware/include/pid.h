#pragma once

// Simple PID controller.
// The derivative is passed in directly (e.g. gyro rate) instead of differentiating
// the error, which is much less noisy.
class PID
{
public:
    PID(float kp, float ki, float kd, float output_limit)
        : kp(kp), ki(ki), kd(kd), _limit(output_limit) {}

    float update(float error, float error_rate, float dt) {
        _integral += ki * error * dt;
        _integral = clamp(_integral, -_limit, _limit);  // anti windup

        float out = kp * error + _integral + kd * error_rate;
        return clamp(out, -_limit, _limit);
    }

    void reset() { _integral = 0.0f; }

    float kp, ki, kd;

private:
    static float clamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

    float _limit;
    float _integral = 0.0f;
};
