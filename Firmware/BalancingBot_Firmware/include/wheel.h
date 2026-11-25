#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <SimpleFOC.h>
#include <atomic>
#include "bus_lock.h"

// AS5600 reads hold the bus lock (if the bus is shared with another core)
class LockedMagneticSensorI2C : public MagneticSensorI2C
{
public:
    using MagneticSensorI2C::MagneticSensorI2C;
    BusLock* bus_lock = nullptr;
    float getSensorAngle() override {
        BusGuard guard(bus_lock);
        return MagneticSensorI2C::getSensorAngle();
    }
};

class Wheel
{
public:
    Wheel(TwoWire& bus, int sda, int scl, int pinA, int pinB, int pinC, int pinEnable,
          bool reversed = false,
          float zero_electric_angle = NOT_SET,
          Direction sensor_direction = Direction::UNKNOWN,
          BusLock* bus_lock = nullptr);

    void init();

    // FOC core: call as fast as possible
    void update();

    // Control core: set the target, and once per control step update the velocity
    void setCurrent(float current) { _target = current; }  // A, positive = forward
    void updateVelocity();

    // All values are in the wheel's forward frame (positive = forward)
    float velocity() { return _velocity; }                          // rad/s, over the last control step
    float angle() { return _fwd_sign * (float)_prev_angle; }        // rad, at the last control step
    float current()  { return _fwd_sign * _motor.current.q; }       // A (estimated)
    float voltage()  { return _fwd_sign * _motor.voltage.q; }       // V applied

private:
    BLDCMotor _motor;
    BLDCDriver3PWM _driver;
    LockedMagneticSensorI2C _sensor;
    TwoWire& _wire;
    int _sda;
    int _scl;
    float _fwd_sign;             // +1 or -1, maps forward rotation to motor direction
    float _sensor_sign = 1.0f;   // +1 or -1, maps the raw sensor angle to motor direction (sensor_direction)
    float _zero_electric_angle;  // FOC calibration (NOT_SET = calibrate on init)
    Direction _sensor_dir;       // FOC calibration (UNKNOWN = calibrate on init)
    std::atomic<float> _target{0.0f};  // A, written by the control core

    // Angle snapshot, written by the FOC core (double: stays precise over many turns)
    portMUX_TYPE _mux = portMUX_INITIALIZER_UNLOCKED;
    double _angle = 0.0;
    uint32_t _angle_us = 0;

    // Used by the control core only
    double _prev_angle = 0.0;
    uint32_t _prev_us = 0;
    float _velocity = 0.0f;      // rad/s, forward frame

};
