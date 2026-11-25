#pragma once

#include <Arduino.h>

// Serializes complete I2C transactions on a bus used from both cores (Wire: left wheel sensor on
// core 1, IMU on core 0). The Arduino Wire lock does not cover the read() calls after requestFrom(),
// so without this one core can overwrite the other's receive buffer: corrupted angles and IMU samples.
class BusLock
{
public:
    void begin() { _mutex = xSemaphoreCreateMutex(); }  // before the first transaction
    void lock()   { if (_mutex) xSemaphoreTake(_mutex, portMAX_DELAY); }
    void unlock() { if (_mutex) xSemaphoreGive(_mutex); }

private:
    SemaphoreHandle_t _mutex = nullptr;
};

// Holds the lock for the current scope (nullptr = no lock)
class BusGuard
{
public:
    explicit BusGuard(BusLock* lock) : _lock(lock) { if (_lock) _lock->lock(); }
    ~BusGuard() { if (_lock) _lock->unlock(); }

private:
    BusLock* _lock;
};
