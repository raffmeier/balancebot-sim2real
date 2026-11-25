# Observer

Live web view of the robot: 3D twin, all telemetry as charts, loop rates and link quality. Drive it
from the browser, record logs and replay them in the same view.

```
python observer.py            # then open http://localhost:8080
```

The PC must be on the same WiFi as the robot (`wifi_config.h`). The bridge finds the robot by
broadcast, or use `--robot <ip>`. The robot streams to one PC only: don't run `robot_log.py` or a
second observer at the same time, they take the stream from each other.

Without the robot: `python fake_robot.py` and `python observer.py --robot 127.0.0.1`. The fake
robot answers the same protocol, including drive commands and motors off.

## Using it

- **Drive**: tick "Enable driving", then W A S D / arrow keys or drag the pad. The sliders set the
  full-scale speed and yaw rate. Commands are sent at 25 Hz only while driving is enabled, the tab is
  visible and the live view is shown. The robot clamps them to the training range, 
  the policy ramps them, and they fall back to 0 300 ms after the last one.
- **Motors off**: Space, any time (also while replaying). The robot falls over and stays off until
  "Arm"; then balancing restarts once it is held upright.
- **Charts**: window length, smoothing (centered moving average, no lag; tiles and 3D stay raw),
  F to freeze, then drag to zoom and double-click to reset. Shaded bands = controller off.
- **Record**: saves to `RL_training/logs/<date_time>.npy`, the same format as `robot_log.py`, so
  `compare_real_sim.py` works on it. Saved automatically when the bridge stops.
- **Replay**: pick a log and "Open". Play / pause (K), speed, J / L 1 s back / forward, comma /
  period one control step. The red strip under the scrub bar shows where the controller was off.
  Older logs (telemetry v1) open too.

## How it works

- `observer.py`: requests the UDP telemetry stream (protocol in `telemetry.h`, packet layout and
  parser from `RL_training/robot_log.py`), forwards validated packets to the browsers over a
  WebSocket, and drive / motor / record commands the other way. Serves the page, the model
  (`balancebot.xml` + STL meshes), `POLICY_OBS_SCALE` from `weights/model.h` (for the reference
  the policy tracks) and the logs.
- `static/js/telemetry.js`: decodes packets with the field list the bridge sends, so a new
  telemetry field only needs `telemetry.h` and `PACKET_DTYPE`. Stores one array per channel, plus
  the odometry (position from the wheels and pitch, heading from the gyro: drifts slowly).
- `static/js/source.js`: what is shown, live (60 ms behind the newest packet, they come in bursts
  of 8) or a log at the replay time.
- `static/js/twin.js`: three.js scene built from the MuJoCo model.
- `static/js/charts.js`: uPlot charts, `static/js/drive.js`: keyboard and pad.

Libraries are in `static/vendor` (three.js 0.160, uPlot 1.6.30), so it also works on the robot's
own WiFi without internet.
