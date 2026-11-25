#pragma once

#include <Arduino.h>
#include <Wire.h>
#include "bus_lock.h"

struct ImuData {
    float ax, ay, az;       // m/s^2
    float gx, gy, gz;       // rad/s (bias removed)
    float temp;             // deg C
    uint32_t timestamp_us;  // micros() at read
};

class MPU6050
{
public:
    MPU6050(TwoWire& bus, uint8_t address = 0x68, BusLock* bus_lock = nullptr);

    // Bus must already be started (Wire.begin). Robot must be still while the gyro bias is measured.
    bool begin(int gyro_calib_samples = 500);

    // Reads one sample, false on bus error
    bool read(ImuData& d);

private:
    bool readRaw(int16_t raw[7]);
    bool writeReg(uint8_t reg, uint8_t value);

    TwoWire& _wire;
    uint8_t _addr;
    BusLock* _bus_lock;
    float _gyro_bias[3] = {0, 0, 0};
};
