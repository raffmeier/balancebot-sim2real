#include "mpu6050.h"

namespace {
constexpr uint8_t REG_SMPLRT_DIV   = 0x19;
constexpr uint8_t REG_CONFIG       = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG  = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_PWR_MGMT_1   = 0x6B;
constexpr uint8_t REG_WHO_AM_I     = 0x75;

constexpr float ACCEL_SCALE = 9.80665f / 8192.0f;           // +-4 g
constexpr float GYRO_SCALE  = (PI / 180.0f) / 65.5f;        // +-500 deg/s
}

MPU6050::MPU6050(TwoWire& bus, uint8_t address, BusLock* bus_lock):
    _wire(bus),
    _addr(address),
    _bus_lock(bus_lock) {}

bool MPU6050::begin(int gyro_calib_samples) {
    _wire.beginTransmission(_addr);
    _wire.write(REG_WHO_AM_I);
    if (_wire.endTransmission(false) != 0 || _wire.requestFrom(_addr, (uint8_t)1) != 1) {
        return false;
    }
    _wire.read();

    writeReg(REG_PWR_MGMT_1, 0x01);    // wake up, clock from gyro X PLL
    delay(100);
    writeReg(REG_SMPLRT_DIV, 0x00);    // 1 kHz sample rate (with DLPF on)
    writeReg(REG_CONFIG, 0x02);        // DLPF ~94 Hz accel / 98 Hz gyro
    writeReg(REG_GYRO_CONFIG, 0x08);   // +-500 deg/s
    writeReg(REG_ACCEL_CONFIG, 0x08);  // +-4 g
    delay(50);

    // Gyro bias: average while the robot is still
    float sum[3] = {0, 0, 0};
    int n = 0;
    int16_t raw[7];
    for (int i = 0; i < gyro_calib_samples; i++) {
        if (readRaw(raw)) {
            sum[0] += raw[4];
            sum[1] += raw[5];
            sum[2] += raw[6];
            n++;
        }
        delay(2);
    }
    if (n > 0) {
        for (int i = 0; i < 3; i++) _gyro_bias[i] = sum[i] / n * GYRO_SCALE;
    }
    return true;
}

bool MPU6050::read(ImuData& d) {
    int16_t raw[7];
    if (!readRaw(raw)) return false;
    d.ax = raw[0] * ACCEL_SCALE;
    d.ay = raw[1] * ACCEL_SCALE;
    d.az = raw[2] * ACCEL_SCALE;
    d.temp = raw[3] / 340.0f + 36.53f;
    d.gx = raw[4] * GYRO_SCALE - _gyro_bias[0];
    d.gy = raw[5] * GYRO_SCALE - _gyro_bias[1];
    d.gz = raw[6] * GYRO_SCALE - _gyro_bias[2];
    d.timestamp_us = micros();
    return true;
}

// One burst read of accel, temp and gyro (14 bytes), keeps bus occupancy short
bool MPU6050::readRaw(int16_t raw[7]) {
    BusGuard guard(_bus_lock);
    _wire.beginTransmission(_addr);
    _wire.write(REG_ACCEL_XOUT_H);
    if (_wire.endTransmission(false) != 0) return false;
    if (_wire.requestFrom(_addr, (uint8_t)14) != 14) return false;
    for (int i = 0; i < 7; i++) {
        uint8_t hi = _wire.read();
        uint8_t lo = _wire.read();
        raw[i] = (int16_t)((hi << 8) | lo);
    }
    return true;
}

bool MPU6050::writeReg(uint8_t reg, uint8_t value) {
    BusGuard guard(_bus_lock);
    _wire.beginTransmission(_addr);
    _wire.write(reg);
    _wire.write(value);
    return _wire.endTransmission() == 0;
}
