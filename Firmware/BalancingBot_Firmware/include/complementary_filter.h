#pragma once

#include <math.h>

// Fuses gyro (fast, but drifts) with accelerometer (noisy, but no drift) into one angle.
//   angle = alpha * (angle + rate * dt) + (1 - alpha) * accel_angle
// tau is the time constant in seconds: below it the gyro is trusted, above it the accelerometer.
class ComplementaryFilter
{
public:
    explicit ComplementaryFilter(float tau = 0.5f) : _tau(tau) {}

    float update(float accel_angle, float rate, float dt) {
        if (!_initialized) {
            _angle = accel_angle;
            _initialized = true;
            return _angle;
        }
        float alpha = _tau / (_tau + dt);
        _angle = alpha * (_angle + rate * dt) + (1.0f - alpha) * accel_angle;
        return _angle;
    }

    float angle() const { return _angle; }

private:
    float _tau;
    float _angle = 0.0f;
    bool _initialized = false;
};
