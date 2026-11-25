#include "wheel.h"

Wheel::Wheel(TwoWire& bus, int sda, int scl, int pinA, int pinB, int pinC, int pinEnable,
             bool reversed, float zero_electric_angle, Direction sensor_direction,
             BusLock* bus_lock):
    _motor(11, 2.8, 140), // GM3506, 11 pole pairs, 5.6 Ohm phase resistance, 140 KV
    _driver(pinA, pinB, pinC, pinEnable),
    _sensor(AS5600_I2C),
    _wire(bus),
    _sda(sda),
    _scl(scl),
    _fwd_sign(reversed ? -1.0f : 1.0f),
    _zero_electric_angle(zero_electric_angle),
    _sensor_dir(sensor_direction) {
    _sensor.bus_lock = bus_lock;
}

void Wheel::init() {
    _wire.begin(_sda, _scl, 1000000);
    _sensor.init(&_wire);
    _motor.linkSensor(&_sensor);

    _driver.voltage_power_supply = 12.0;
    _driver.init();

    _motor.linkDriver(&_driver);

    _motor.torque_controller = TorqueControlType::estimated_current;
    _motor.controller = MotionControlType::torque;
    _motor.updateCurrentLimit(2.0);

    // Smoothing (audible jitter), none of it changes what the policy sees:
    // - back-EMF compensation uses SimpleFOC's own velocity, which is very noisy from one AS5600
    //   step per ~0.3 ms loop: measure it over >= 1 ms and filter it slowly (it only matters at speed)
    // - the current target changes in steps every control period (5 ms): ramp it over ~2 ms
    _sensor.min_elapsed_time = 0.001f;
    _motor.LPF_velocity.Tf = 0.02f;
    _motor.LPF_current_q.Tf = 0.002f;

    _motor.init();
    // Stored calibration; if left at NOT_SET / UNKNOWN, initFOC() calibrates
    _motor.zero_electric_angle = _zero_electric_angle;
    _motor.sensor_direction    = _sensor_dir;
    _motor.initFOC();

    Serial.printf("zero_electric_angle = %f, sensor_direction = %d\n",
                  _motor.zero_electric_angle, (int)_motor.sensor_direction);

    // The raw sensor angle runs against the motor's torque direction when sensor_direction is CCW,
    // SimpleFOC's motor frame is sensor_direction * raw angle: use the same frame
    _sensor_sign = (float)_motor.sensor_direction;
    _angle = _prev_angle = _sensor_sign * _sensor.getPreciseAngle();
    _angle_us = _prev_us = micros();
}

void Wheel::update() {
    _motor.loopFOC();
    // SimpleFOC does not apply current_limit in torque mode
    float target = constrain(_target.load(), -_motor.current_limit, _motor.current_limit);
    _motor.move(_fwd_sign * target);

    // Publish a consistent angle + time for the control core
    double angle = _sensor_sign * _sensor.getPreciseAngle();
    uint32_t now = micros();
    portENTER_CRITICAL(&_mux);
    _angle = angle;
    _angle_us = now;
    portEXIT_CRITICAL(&_mux);
}

void Wheel::updateVelocity() {
    portENTER_CRITICAL(&_mux);
    double angle = _angle;
    uint32_t t = _angle_us;
    portEXIT_CRITICAL(&_mux);

    float dt = (t - _prev_us) * 1e-6f;
    if (dt > 0.0f) {
        _velocity = _fwd_sign * (float)((angle - _prev_angle) / dt);
    }
    _prev_angle = angle;
    _prev_us = t;
}
