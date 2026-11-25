#include <Arduino.h>
#include <SimpleFOC.h>

// ---- hardware ----
BLDCMotor motor = BLDCMotor(11);                        // GM3506: 24N22P -> 11 pole pairs
BLDCDriver3PWM driver = BLDCDriver3PWM(14, 13, 12, 11); // IN1, IN2, IN3, EN
MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);

// ---- runtime target ----
float target = 50;   // rad/s at the shaft
Commander command = Commander(Serial);
void doTarget(char* cmd) { command.scalar(&target, cmd); }
void doMotor(char* cmd)  { command.motor(&motor, cmd); }

void setup() {
  Serial.begin(115200);
  delay(2000);                       // S3 USB CDC needs a moment after reset
  SimpleFOCDebug::enable(&Serial);

    // ---- sensor ----
    // init first, then re-establish the bus ourselves: SimpleFOC's init calls
    // Wire.begin() internally, which the esp32 i2c-ng driver won't tolerate twice
  sensor.init(&Wire);
  Wire.end();
  Wire.begin(8, 9);
  Wire.setClock(400000);             // needed for a decent loop rate
  motor.linkSensor(&sensor);

    // ---- driver ----
  driver.voltage_power_supply = 12;
  driver.voltage_limit = 6;
  driver.init();
  motor.linkDriver(&driver);

    // ---- motor limits ----
    // measure phase-to-phase resistance with a multimeter and halve it for a
    // wye winding. Setting it lets SimpleFOC limit by current instead of volts.
  motor.phase_resistance = 5.6;      // ohms - MEASURE YOURS
  motor.current_limit = 0.5;         // amps - conservative, raise once stable
  motor.velocity_limit = 30;         // rad/s

    // if you'd rather not set phase_resistance, comment the two lines above
    // and use this instead:
    // motor.voltage_limit = 3;

    // ---- control ----
  motor.torque_controller = TorqueControlType::voltage;
  motor.controller = MotionControlType::velocity;

  motor.PID_velocity.P = 1;
  motor.PID_velocity.I = 0.5;
  motor.PID_velocity.D = 0.0;
  motor.LPF_velocity.Tf = 0.01;            // lower = less lag, more noise

  motor.init();
  motor.initFOC();                   // watch for "MOT: Success!"

  command.add('T', doTarget, "target velocity");
  command.add('M', doMotor,  "motor");

  Serial.println(F("ready - send T5 for 5 rad/s"));
}

void loop() {
  motor.loopFOC();                   // must run every iteration, nothing blocking
  motor.move(target);
  command.run();

    // never print every iteration - it will starve loopFOC()
  static unsigned long t = 0;
  if (millis() - t > 200) {
  t = millis();
  Serial.printf("v=%6.2f  target=%6.2f  Uq=%5.2f\n",
  motor.shaft_velocity, target, motor.voltage.q);
    }
}