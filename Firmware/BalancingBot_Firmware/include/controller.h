#pragma once

// Everything a controller may use. Signs: positive = forward / turning left.
struct Observation {
    float pitch;            // rad, positive = leaning forward, 0 = upright
    float pitch_rate;       // rad/s
    float yaw_rate;         // rad/s, positive = turning left (CCW seen from above)
    float wheel_vel_left;   // rad/s, positive = forward
    float wheel_vel_right;  // rad/s, positive = forward
    float v_cmd;            // m/s, commanded forward velocity
    float yaw_rate_cmd;     // rad/s, commanded yaw rate
};

// Wheel currents in A, positive = forward
struct Action {
    float current_left;
    float current_right;
};

// Base class for all balancing controllers (PID, RL policy, ...)
class Controller
{
public:
    virtual ~Controller() = default;
    virtual void reset() {}  // called when balancing (re)starts
    virtual Action update(const Observation& obs, float dt) = 0;
};
