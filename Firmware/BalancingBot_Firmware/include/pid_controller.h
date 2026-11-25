#pragma once

#include "controller.h"
#include "pid.h"

// Pitch PID: leaning forward -> drive both wheels forward to get back under the CoM
class PidController : public Controller
{
public:
    PidController(float kp, float ki, float kd, float max_current)
        : _pid(kp, ki, kd, max_current) {}

    void reset() override { _pid.reset(); }

    Action update(const Observation& obs, float dt) override {
        float current = _pid.update(obs.pitch, obs.pitch_rate, dt);
        return {current, current};
    }

private:
    PID _pid;
};
