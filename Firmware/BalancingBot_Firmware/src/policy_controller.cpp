#include "policy_controller.h"
#include <esp_attr.h>

// Weights in RAM: from flash (via the cache, which they don't fit in) inference takes ~3.5 ms
#define POLICY_WEIGHT_ATTR DRAM_ATTR
#include "../weights/model.h"

// Older exports: no ramp, the reference is the command
#ifndef POLICY_V_CMD_ACC_MAX
#define POLICY_V_CMD_ACC_MAX 0.0f
#define POLICY_YAWRATE_CMD_ACC_MAX 0.0f
#endif

// Older exports: the BalanceRobotEnv default command range
#ifndef POLICY_V_CMD_MAX
#define POLICY_V_CMD_MAX 0.3f
#define POLICY_YAWRATE_CMD_MAX 2.0f
#endif

const float PolicyController::V_CMD_MAX = POLICY_V_CMD_MAX;
const float PolicyController::YAW_RATE_CMD_MAX = POLICY_YAWRATE_CMD_MAX;

static_assert(POLICY_OBS_DIM == PolicyController::OBS_DIM, "policy_weights.h does not match the observation below, re-export the policy");

const float TORQUE_CONSTANT = 0.059f;  // Nm/A (measured)

// Moves value toward target by at most max_step (max_step 0 = jump to the target)
static float rampToward(float value, float target, float max_step) {
    if (max_step <= 0.0f) return target;
    return value + fminf(fmaxf(target - value, -max_step), max_step);
}

Action PolicyController::update(const Observation& obs, float dt) {
    // The policy sees and tracks the command ramped with limited acceleration (BalanceRobotEnv._ramp_reference)
    _v_ref = rampToward(_v_ref, obs.v_cmd, POLICY_V_CMD_ACC_MAX * dt);
    _yaw_rate_ref = rampToward(_yaw_rate_ref, obs.yaw_rate_cmd, POLICY_YAWRATE_CMD_ACC_MAX * dt);

    // Same order as in training (BalanceRobotEnv._get_obs)
    const float raw[POLICY_OBS_DIM] = {
        obs.pitch, obs.pitch_rate, obs.yaw_rate,
        obs.wheel_vel_left, obs.wheel_vel_right,
        _v_ref, _yaw_rate_ref,
        _prev_action[0], _prev_action[1],
    };
    for (int i = 0; i < POLICY_OBS_DIM; i++) _last_input[i] = raw[i] / POLICY_OBS_SCALE[i];

    float action[POLICY_ACT_DIM];  // [-1, 1] per wheel: left, right
    policy_forward(_last_input, action);
    _prev_action[0] = action[0];
    _prev_action[1] = action[1];

    const float scale = POLICY_MAX_TORQUE / TORQUE_CONSTANT;  // action -> A
    return {action[0] * scale, action[1] * scale};
}
