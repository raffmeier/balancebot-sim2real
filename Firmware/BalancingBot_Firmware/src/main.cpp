#include <Arduino.h>
#include <SimpleFOC.h>
#include "wheel.h"
#include "mpu6050.h"
#include "complementary_filter.h"
#include "pid_controller.h"
#include "policy_controller.h"
#include "telemetry.h"

// Core 1 (Arduino loop): FOC only, as fast as possible
// Core 0 (control task): IMU, observations and controller at 200 Hz
// Core 0 (telemetry task, lower priority): serial output, see telemetry.h

// Wire is shared by the left wheel sensor (core 1) and the IMU (core 0)
BusLock wire_lock;
Wheel left_wheel(Wire, 8, 9, 14, 13, 12, 10, true, 4.86, Direction::CW, &wire_lock);
Wheel right_wheel(Wire1, 17, 18, 15, 7, 6, 5, true, 2.89, Direction::CCW);
MPU6050 imu(Wire, 0x68, &wire_lock);

// Pitch: positive = leaning forward (rad)
ComplementaryFilter pitch_filter(0.5);

// Controllers, select the active one below
// PID start: Kp ~2x the gravity torque (~3 A/rad for 600 g, CoM 6.5 cm), Kd to damp
PidController pid_controller(6.0, 0.0, 0.3, 1.0);
PolicyController policy_controller;
Controller& controller = policy_controller;

const float PITCH_OFFSET   = 0.0;  // rad, measured pitch when balanced (IMU mounting trim)
const float FALL_ANGLE     = 0.6;  // rad (~35 deg), motors off beyond this
const float RESTART_ANGLE  = 0.05; // rad (~3 deg), re-enable when held upright

// Control step rate, same as the RL policy
const uint32_t CONTROL_PERIOD_MS = 5;  // 200 Hz
const float CONTROL_DT = CONTROL_PERIOD_MS * 1e-3f;

// Torque test: right wheel holds the current typed into the serial monitor (e.g. "0.3" + Enter),
// left wheel off, no balancing
const bool TORQUE_TEST = false;

Observation obs = {};
Action action = {0, 0};
RemoteCommand remote = {0, 0, false, true};
bool balancing = false;
volatile uint32_t foc_loops = 0;  // counted on core 1, for the loop rate printout
Telemetry telemetry;

void controlTask(void*);

void setup() {
  Serial.begin(115200);
  SimpleFOCDebug::enable(&Serial);
  delay(1000);
  Serial.println("Initializing..");
  wire_lock.begin();
  left_wheel.init();
  right_wheel.init();
  Serial.println("Wheel initialized");

  // After left_wheel.init(), which starts the Wire bus. Keep the robot still.
  if (!imu.begin()) {
    Serial.println("MPU6050 not found");
  }

  // Torque test reads the serial monitor itself
  if (!TORQUE_TEST) telemetry.begin(1, 0);
  xTaskCreatePinnedToCore(controlTask, "control", 8192, nullptr, 2, nullptr, 0);
}

// Computes all observations, then the wheel currents. Records everything in pkt.
void controlStep(float dt, TelemetryPacket& pkt) {
  remote = telemetry.remoteCommand();
  ImuData d;
  if (!imu.read(d)) return;  // bus error: keep the last currents
  pkt.flags |= TELEM_IMU_OK;

  // IMU axes: x = left, y = back, z = up -> pitch is rotation about x
  float accel_pitch = atan2(d.ay, d.az);
  pkt.accel_pitch = accel_pitch;
  obs.pitch_rate = d.gx;
  obs.pitch = pitch_filter.update(accel_pitch, obs.pitch_rate, dt) - PITCH_OFFSET;
  obs.yaw_rate = d.gz;

  left_wheel.updateVelocity();
  right_wheel.updateVelocity();
  obs.wheel_vel_left = left_wheel.velocity();
  obs.wheel_vel_right = right_wheel.velocity();

  // From the Observer, 0 (balance in place) without a fresh command
  // Clamped to the policy's training range (exported into weights/model.h)
  const float v_max = PolicyController::V_CMD_MAX, yaw_rate_max = PolicyController::YAW_RATE_CMD_MAX;
  obs.v_cmd = constrain(remote.v, -v_max, v_max);
  obs.yaw_rate_cmd = constrain(remote.yaw_rate, -yaw_rate_max, yaw_rate_max);

  // Fallen over or switched off remotely: stop. Restart only once it is put back upright.
  if (fabs(obs.pitch) > FALL_ANGLE || !remote.armed) balancing = false;
  if (!balancing && remote.armed && fabs(obs.pitch) < RESTART_ANGLE) {
    balancing = true;
    controller.reset();
  }

  action = balancing ? controller.update(obs, dt) : Action{0, 0};
  left_wheel.setCurrent(action.current_left);
  right_wheel.setCurrent(action.current_right);

  if (balancing) pkt.flags |= TELEM_BALANCING;
  pkt.accel[0] = d.ax; pkt.accel[1] = d.ay; pkt.accel[2] = d.az;
  pkt.gyro[0] = d.gx;  pkt.gyro[1] = d.gy;  pkt.gyro[2] = d.gz;
}

// Everything that is not set in controlStep (also sent when the IMU read failed)
void fillPacket(TelemetryPacket& pkt) {
  pkt.pitch = obs.pitch;
  pkt.wheel_angle[0] = left_wheel.angle();
  pkt.wheel_angle[1] = right_wheel.angle();
  pkt.wheel_vel[0] = obs.wheel_vel_left;
  pkt.wheel_vel[1] = obs.wheel_vel_right;
  // Only meaningful with the policy controller active (stale while not balancing)
  memcpy(pkt.policy_in, policy_controller.lastInput(), sizeof(pkt.policy_in));
  memcpy(pkt.policy_out, policy_controller.lastOutput(), sizeof(pkt.policy_out));
  pkt.current_cmd[0] = action.current_left;
  pkt.current_cmd[1] = action.current_right;
  pkt.current_est[0] = left_wheel.current();
  pkt.current_est[1] = right_wheel.current();
  pkt.voltage[0] = left_wheel.voltage();
  pkt.voltage[1] = right_wheel.voltage();
  pkt.v_cmd = obs.v_cmd;
  pkt.yaw_rate_cmd = obs.yaw_rate_cmd;
  if (remote.fresh) pkt.flags |= TELEM_REMOTE;
  if (!remote.armed) pkt.flags |= TELEM_DISARMED;
}

// Sets the right wheel current from the serial monitor and prints what the wheel does
void torqueTest() {
  static float current = 0.0;
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) current = line.toFloat();
  }
  left_wheel.setCurrent(0.0);
  right_wheel.setCurrent(current);

  static uint32_t last_print = 0;
  if (millis() - last_print < 200) return;
  last_print = millis();
  right_wheel.updateVelocity();
  Serial.printf("right wheel: target %5.2f A | estimated %5.2f A | %5.2f V | %6.2f rad/s\n",
                current, right_wheel.current(), right_wheel.voltage(), right_wheel.velocity());
}

void controlTask(void*) {
  TickType_t last_wake = xTaskGetTickCount();
  uint32_t step = 0;
  uint32_t last_foc_loops = foc_loops;
  for (;;) {
    if (TORQUE_TEST) {
      torqueTest();
    } else {
      TelemetryPacket pkt = {};
      pkt.step = step++;
      pkt.t_us = micros();
      uint32_t loops = foc_loops;
      pkt.foc_loops = loops - last_foc_loops;
      last_foc_loops = loops;

      controlStep(CONTROL_DT, pkt);
      pkt.compute_us = micros() - pkt.t_us;
      fillPacket(pkt);
      telemetry.push(pkt);
    }
    vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(CONTROL_PERIOD_MS));
  }
}

void loop() {
  left_wheel.update();
  right_wheel.update();
  foc_loops++;
}
