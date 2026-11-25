#pragma once

#include "controller.h"

// RL policy exported to weights/policy_weights.h (network outputs wheel torques)
class PolicyController : public Controller
{
public:
    // Robot at rest: the command reference starts at 0
    void reset() override { _prev_action[0] = _prev_action[1] = _v_ref = _yaw_rate_ref = 0.0f; }
    Action update(const Observation& obs, float dt) override;

    // Command range the policy was trained on (from weights/model.h): clamp the remote command to it
    static const float V_CMD_MAX;         // m/s
    static const float YAW_RATE_CMD_MAX;  // rad/s

    // Last network input (normalized observation) and output, for logging
    static constexpr int OBS_DIM = 9;
    const float* lastInput() const { return _last_input; }
    const float* lastOutput() const { return _prev_action; }

private:
    float _prev_action[2] = {0.0f, 0.0f};  // last network output, part of the observation
    float _v_ref = 0.0f;         // m/s, command ramped like in training
    float _yaw_rate_ref = 0.0f;  // rad/s
    float _last_input[OBS_DIM] = {};
};
