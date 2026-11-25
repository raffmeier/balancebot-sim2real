#pragma once

#include <Arduino.h>
#include <WiFiUdp.h>
#include <atomic>

// One packet per control step, streamed as raw bytes (little endian) over WiFi UDP.
// RL_training/robot_log.py has the same layout (PACKET_DTYPE): change both together and bump the version.
#define TELEMETRY_VERSION 2

enum TelemetryFlags : uint8_t {
    TELEM_BALANCING = 1 << 0,  // controller active, motors on
    TELEM_IMU_OK    = 1 << 1,  // IMU read succeeded this step (otherwise the IMU fields are stale)
    TELEM_REMOTE    = 1 << 2,  // a drive command arrived within COMMAND_TIMEOUT_MS
    TELEM_DISARMED  = 1 << 3,  // motors switched off remotely ("k"), until re-armed ("a")
};

struct __attribute__((packed)) TelemetryPacket {
    uint8_t  sync[2];          // 0xAA 0x55, filled in by Telemetry
    uint8_t  version;          // TELEMETRY_VERSION, filled in by Telemetry
    uint8_t  flags;            // TelemetryFlags
    uint32_t step;             // control step counter, gaps = dropped packets
    uint32_t t_us;             // micros() at the start of the step
    uint16_t compute_us;       // time spent in the control step (IMU read + controller)
    uint16_t foc_loops;        // FOC iterations since the previous step
    float accel[3];            // m/s^2, IMU frame (x = left, y = back, z = up)
    float gyro[3];             // rad/s, IMU frame, bias removed
    float accel_pitch;         // rad, pitch from the accelerometer alone
    float pitch;               // rad, complementary filter output (as the controller sees it)
    float wheel_angle[2];      // rad, left / right, forward frame
    float wheel_vel[2];        // rad/s, left / right, forward frame
    float policy_in[9];        // normalized observation fed to the network
    float policy_out[2];       // network output in [-1, 1], left / right
    float current_cmd[2];      // A, commanded
    float current_est[2];      // A, SimpleFOC estimate
    float voltage[2];          // V, applied q voltage
    float v_cmd;               // m/s, forward velocity command used (0 without a remote command)
    float yaw_rate_cmd;        // rad/s, yaw rate command used
    uint16_t crc;              // CRC-16/XMODEM over all bytes above, filled in by Telemetry
};
static_assert(sizeof(TelemetryPacket) == 142, "update PACKET_DTYPE in robot_log.py");

// Drive command from the PC (Observer), as the control task reads it
struct RemoteCommand {
    float v;         // m/s, 0 when not fresh
    float yaw_rate;  // rad/s, 0 when not fresh
    bool fresh;      // received within COMMAND_TIMEOUT_MS
    bool armed;      // false after "k": motors stay off
};

// Output from its own low priority task, so the control loop never waits for WiFi or USB.
//   WiFi UDP: every packet, batched, to the PC that last sent "b" (robot_log.py, it repeats it
//             as a keep-alive). Stops after a "t" or when the keep-alive times out.
//   Serial:   a status line every 100 ms, readable in the serial monitor
// Commands received on the same port (one per datagram):
//   "b" / "t"             start (keep-alive) / stop streaming
//   "c" + 2 x float32 LE  drive command: forward velocity (m/s), yaw rate (rad/s). Repeated by the
//                         sender, falls back to 0 when none arrives for COMMAND_TIMEOUT_MS (dead man)
//   "k" / "a"             motors off (latched) / armed again, balancing restarts once held upright
// WiFi settings: wifi_config.h
class Telemetry
{
public:
    static const uint32_t COMMAND_TIMEOUT_MS = 300;

    void begin(UBaseType_t priority, BaseType_t core);

    // Control task: queues the packet, never blocks (dropped if the queue is full)
    void push(const TelemetryPacket& p);

    // Control task: the latest drive command
    RemoteCommand remoteCommand();

private:
    static const int BATCH = 8;  // packets per datagram: 1136 bytes, 25 datagrams/s

    static void taskEntry(void* self) { static_cast<Telemetry*>(self)->run(); }
    void run();
    void startWifi();
    bool wifiReady();
    void handleCommands();
    void send(TelemetryPacket& p);
    void printStatus(const TelemetryPacket& p);

    QueueHandle_t _queue = nullptr;
    std::atomic<uint32_t> _dropped{0};

    // WiFi
    WiFiUDP _udp;
    bool _udp_started = false;
    IPAddress _remote;
    uint16_t _remote_port = 0;
    uint32_t _last_keepalive_ms = 0;
    bool _streaming = false;
    uint8_t _batch[BATCH * sizeof(TelemetryPacket)];
    int _batch_count = 0;

    // Latest remote command, written by the telemetry task, read by the control task
    portMUX_TYPE _cmd_mux = portMUX_INITIALIZER_UNLOCKED;
    float _v_cmd = 0.0f;
    float _yaw_rate_cmd = 0.0f;
    uint32_t _cmd_ms = 0;
    bool _cmd_received = false;
    bool _armed = true;

    // Text status rate counters
    uint32_t _last_print_ms = 0;
    uint32_t _steps = 0;
    uint32_t _foc_loops = 0;
};
