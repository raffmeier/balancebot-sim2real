#include "telemetry.h"
#if __has_include("wifi_config.h")
#include "wifi_config.h"
#else
#include "wifi_config.example.h"
#endif
#include <WiFi.h>
#include <ESPmDNS.h>

namespace {
const int QUEUE_LENGTH = 64;               // 320 ms of packets at 200 Hz
const uint32_t STATUS_PERIOD_MS = 100;
const uint32_t KEEPALIVE_TIMEOUT_MS = 3000;

uint16_t crc16_xmodem(const uint8_t* data, size_t len) {
    uint16_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
    }
    return crc;
}

bool apMode() { return strlen(WIFI_SSID) == 0; }
}

void Telemetry::begin(UBaseType_t priority, BaseType_t core) {
    _queue = xQueueCreate(QUEUE_LENGTH, sizeof(TelemetryPacket));
    xTaskCreatePinnedToCore(taskEntry, "telemetry", 6144, this, priority, nullptr, core);
}

void Telemetry::push(const TelemetryPacket& p) {
    if (_queue == nullptr) return;
    if (xQueueSend(_queue, &p, 0) != pdTRUE) _dropped++;
}

void Telemetry::startWifi() {
    WiFi.setHostname(WIFI_HOSTNAME);
    if (apMode()) {
        WiFi.mode(WIFI_AP);
        WiFi.softAP(AP_SSID, AP_PASSWORD);
    } else {
        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);  // reconnects by itself
    }
    WiFi.setSleep(false);  // power save delays packets by up to ~100 ms
}

bool Telemetry::wifiReady() {
    return apMode() || WiFi.status() == WL_CONNECTED;
}

void Telemetry::run() {
    startWifi();
    TelemetryPacket p;
    for (;;) {
        handleCommands();
        if (xQueueReceive(_queue, &p, pdMS_TO_TICKS(20)) != pdTRUE) continue;
        printStatus(p);
        if (_streaming) send(p);
    }
}

RemoteCommand Telemetry::remoteCommand() {
    taskENTER_CRITICAL(&_cmd_mux);
    bool fresh = _cmd_received && millis() - _cmd_ms < COMMAND_TIMEOUT_MS;
    RemoteCommand c = {fresh ? _v_cmd : 0.0f, fresh ? _yaw_rate_cmd : 0.0f, fresh, _armed};
    taskEXIT_CRITICAL(&_cmd_mux);
    return c;
}

// Protocol: see telemetry.h
void Telemetry::handleCommands() {
    if (!wifiReady()) {
        _streaming = false;
        return;
    }
    if (!_udp_started) {
        _udp.begin(TELEMETRY_PORT);
        MDNS.begin(WIFI_HOSTNAME);
        _udp_started = true;
        IPAddress ip = apMode() ? WiFi.softAPIP() : WiFi.localIP();
        Serial.printf("Telemetry: UDP %s:%d (%s)\n", ip.toString().c_str(), TELEMETRY_PORT,
                      apMode() ? "own network " AP_SSID : "joined " WIFI_SSID);
    }
    while (_udp.parsePacket() > 0) {
        uint8_t msg[16];
        int len = _udp.read(msg, sizeof(msg));
        _udp.clear();  // discard the rest of the datagram
        if (len <= 0) continue;
        switch (msg[0]) {
        case 'b':
            if (!_streaming) _batch_count = 0;
            _streaming = true;
            _remote = _udp.remoteIP();
            _remote_port = _udp.remotePort();
            _last_keepalive_ms = millis();
            break;
        case 't':
            _streaming = false;
            break;
        case 'c': {
            float v, yaw_rate;
            if (len < 9) break;
            memcpy(&v, msg + 1, 4);
            memcpy(&yaw_rate, msg + 5, 4);
            if (!isfinite(v) || !isfinite(yaw_rate)) break;
            taskENTER_CRITICAL(&_cmd_mux);
            _v_cmd = v;
            _yaw_rate_cmd = yaw_rate;
            _cmd_ms = millis();
            _cmd_received = true;
            taskEXIT_CRITICAL(&_cmd_mux);
            break;
        }
        case 'k':
        case 'a':
            taskENTER_CRITICAL(&_cmd_mux);
            _armed = msg[0] == 'a';
            taskEXIT_CRITICAL(&_cmd_mux);
            break;
        }
    }
    if (_streaming && millis() - _last_keepalive_ms > KEEPALIVE_TIMEOUT_MS) _streaming = false;
}

void Telemetry::send(TelemetryPacket& p) {
    p.sync[0] = 0xAA;
    p.sync[1] = 0x55;
    p.version = TELEMETRY_VERSION;
    p.crc = crc16_xmodem((const uint8_t*)&p, offsetof(TelemetryPacket, crc));
    memcpy(_batch + _batch_count * sizeof(p), &p, sizeof(p));
    if (++_batch_count < BATCH) return;

    _udp.beginPacket(_remote, _remote_port);
    _udp.write(_batch, sizeof(_batch));
    _udp.endPacket();
    _batch_count = 0;
}

// Status line from the latest packet; rates are averaged since the last line
void Telemetry::printStatus(const TelemetryPacket& p) {
    _steps++;
    _foc_loops += p.foc_loops;
    uint32_t elapsed = millis() - _last_print_ms;
    if (elapsed < STATUS_PERIOD_MS) return;

    uint32_t foc_hz = _foc_loops * 1000 / elapsed;
    uint32_t control_hz = _steps * 1000 / elapsed;
    _steps = 0;
    _foc_loops = 0;
    _last_print_ms = millis();

    Serial.printf("foc %5u Hz | control %3u Hz | pitch %6.2f deg | "
                  "action L %5.2f A R %5.2f A %-3s | vel L %6.2f R %6.2f rad/s | "
                  "step %3u us | dropped %u%s%s%s\n",
                  foc_hz, control_hz, p.pitch * RAD_TO_DEG,
                  p.current_cmd[0], p.current_cmd[1], (p.flags & TELEM_BALANCING) ? "ON" : "off",
                  p.wheel_vel[0], p.wheel_vel[1], p.compute_us, _dropped.load(),
                  _streaming ? " | streaming" : "",
                  (p.flags & TELEM_REMOTE) ? " | remote" : "",
                  (p.flags & TELEM_DISARMED) ? " | DISARMED" : "");
}
