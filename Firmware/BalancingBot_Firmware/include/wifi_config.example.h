#pragma once

// Copy to wifi_config.h (not in git) and fill in your network.
// WiFi for the telemetry stream (robot_log.py).
// Empty WIFI_SSID: the robot opens its own network (AP_SSID / AP_PASSWORD), the PC joins it.
// Otherwise the robot joins this network, the PC must be on the same one.
#define WIFI_SSID     ""
#define WIFI_PASSWORD ""

#define AP_SSID       "BalancingBot"
#define AP_PASSWORD   "balancebot"   // at least 8 characters

#define TELEMETRY_PORT 4210          // UDP, same as in robot_log.py
#define WIFI_HOSTNAME  "balancingbot" // also reachable as balancingbot.local (mDNS)
