"""Load any trained policy (SB3 model.zip, my PPO model.pt, mjlab model_<iter>.pt) and export it
as a C header for the firmware.

    python export_weights.py runs/<run>                     # -> runs/<run>/model.h
    python export_weights.py runs/<run>/checkpoints/model_X_steps.zip
    python export_weights.py <mjlab>/logs/rsl_rl/balancebot/<run>   # newest checkpoint
    python export_weights.py <run or model file> -o path/to/model.h

obs_scale, max_torque and the command limits come from the run's config.json (SB3, my PPO) or
params/env.yaml (mjlab), so the header matches the env the policy was trained with.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch as th


def _run_config(model_path):
    run_dir = os.path.dirname(model_path)
    if os.path.basename(run_dir) == "checkpoints":
        run_dir = os.path.dirname(run_dir)
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def _from_config(config):
    env = config["env"]
    return dict(
        obs_scale=config["obs_scale"],
        max_torque=config["max_torque"],
        v_cmd_acc_max=env.get("v_cmd_acc_max") or 0.0,
        yawrate_cmd_acc_max=env.get("yawrate_cmd_acc_max") or 0.0,
        v_cmd_max=env.get("v_cmd_max", 0.3),
        yawrate_cmd_max=env.get("yawrate_cmd_max", 2.0),
        env=env,
    )


def load_sb3(model_path):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    policy_net = model.policy.mlp_extractor.policy_net
    assert all(isinstance(m, (th.nn.Linear, th.nn.Tanh)) for m in policy_net), \
        "policy_forward() assumes tanh activations"
    layers = [m for m in policy_net if isinstance(m, th.nn.Linear)] + [model.policy.action_net]
    return dict(
        weights=[layer.weight.detach().numpy() for layer in layers],   # (out, in), row major
        biases=[layer.bias.detach().numpy() for layer in layers],
        **_from_config(_run_config(model_path)),
    )


def load_my_ppo(model_path, state_dict):
    idx = sorted(int(m.group(1)) for k in state_dict if (m := re.match(r"actor\.(\d+)\.weight$", k)))
    return dict(
        weights=[state_dict[f"actor.{i}.weight"].numpy() for i in idx],
        biases=[state_dict[f"actor.{i}.bias"].numpy() for i in idx],
        **_from_config(_run_config(model_path)),
    )


def _load_yaml(path):
    """mjlab's params/*.yaml without importing mjlab: python tags are loaded as plain data."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def construct_python_tag(loader, tag_suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", construct_python_tag)
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


def _mjlab_obs_scale(env, obs_dim):
    """mjlab stores the multiplier 1 / OBS_SCALE per observation term."""
    scale = []
    for name, term in env["observations"]["actor"]["terms"].items():
        s = term.get("scale")
        if s is None:
            dim = {"last_action": 2}.get(name)
            if dim is None:
                sys.exit(f"Unknown dimension of unscaled observation term '{name}'")
            scale += [1.0] * dim
        elif isinstance(s, (int, float)):
            sys.exit(f"Scalar scale for term '{name}': dimension unknown, use a tuple")
        else:
            scale += [1.0 / float(v) for v in s]
    if len(scale) != obs_dim:
        sys.exit(f"Observation scales ({len(scale)}) don't match the network input ({obs_dim})")
    return scale


def load_mjlab(model_path, ckpt):
    run_dir = os.path.dirname(model_path)
    env = _load_yaml(os.path.join(run_dir, "params", "env.yaml"))
    agent = _load_yaml(os.path.join(run_dir, "params", "agent.yaml"))
    if agent["actor"]["activation"] != "tanh":
        sys.exit(f"policy_forward() assumes tanh activations, actor uses {agent['actor']['activation']}")
    if agent.get("clip_actions") not in (1, 1.0):
        print(f"[WARN] clip_actions = {agent.get('clip_actions')}, the header clips to [-1, 1]")

    actor = ckpt["actor_state_dict"]
    if any(k.startswith(("obs_normalizer", "normalizer")) for k in actor):
        sys.exit("Actor uses obs_normalization: policy_forward() has no normalizer, export not supported")
    idx = sorted(int(m.group(1)) for k in actor if (m := re.match(r"mlp\.(\d+)\.weight$", k)))
    weights = [actor[f"mlp.{i}.weight"].numpy() for i in idx]
    cmd = env["commands"]["velocity"]
    limits = dict(
        v_cmd_acc_max=cmd.get("v_acc_max") or 0.0,
        yawrate_cmd_acc_max=cmd.get("yawrate_acc_max") or 0.0,
        v_cmd_max=float(cmd["v_max"]),
        yawrate_cmd_max=float(cmd["yawrate_max"]),
    )
    return dict(
        weights=weights,
        biases=[actor[f"mlp.{i}.bias"].numpy() for i in idx],
        obs_scale=_mjlab_obs_scale(env, weights[0].shape[1]),
        max_torque=float(env["actions"]["wheel_torque"]["scale"]),
        **limits,
        env=limits,   # no BalanceRobotEnv config: its defaults with the same command limits
        note=f"rsl_rl iteration {ckpt.get('iter')}",
    )


def resolve_model(path):
    """Run folder -> model.zip, model.pt or the newest model_<iter>.pt in it."""
    if os.path.isfile(path):
        return path
    for name in ("model.zip", "model.pt"):
        if os.path.isfile(os.path.join(path, name)):
            return os.path.join(path, name)
    ckpts = [f for f in os.listdir(path) if re.fullmatch(r"model_\d+\.pt", f)]
    if not ckpts:
        sys.exit(f"No model.zip, model.pt or model_<iter>.pt in {path}")
    return os.path.join(path, max(ckpts, key=lambda f: int(re.findall(r"\d+", f)[0])))


def load_policy(path):
    """Run folder or model file -> (model_path, policy dict), the format is detected from the file."""
    model_path = resolve_model(os.path.expanduser(path))
    if model_path.endswith(".zip"):
        return model_path, load_sb3(model_path)
    ckpt = th.load(model_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in ckpt:
        return model_path, load_mjlab(model_path, ckpt)
    return model_path, load_my_ppo(model_path, ckpt)


def policy_forward(policy, obs_norm):
    """Same computation as policy_forward() in the header, in float32. obs_norm: (9,) or (n, 9)."""
    h = np.asarray(obs_norm, dtype=np.float32)
    last = len(policy["weights"]) - 1
    for i, (w, b) in enumerate(zip(policy["weights"], policy["biases"])):
        h = h @ w.T.astype(np.float32) + b.astype(np.float32)
        h = np.tanh(h) if i < last else np.clip(h, -1.0, 1.0)
    return h


def env_actor(policy, env):
    """obs of a BalanceRobotEnv -> action. Rescales if the env's obs_scale differs from training."""
    rescale = np.asarray(env.obs_scale, dtype=np.float32) / np.asarray(policy["obs_scale"], dtype=np.float32)
    return lambda obs: policy_forward(policy, obs * rescale)


def c_array(name, arr):
    values = ", ".join(f"{v:.8e}f" for v in np.asarray(arr).flatten())
    return f"static const float POLICY_WEIGHT_ATTR {name}[{np.asarray(arr).size}] = {{{values}}};\n"


def policy_forward_c(sizes):
    """C version of the deterministic policy: obs (already divided by OBS_SCALE) -> action in [-1, 1]."""
    n = len(sizes) - 1
    lines = [
        "// obs_norm: raw observation divided by POLICY_OBS_SCALE (same order as BalanceRobotEnv._get_obs)",
        "// action:   wheel torques in [-1, 1], multiply by POLICY_MAX_TORQUE for Nm",
        "static inline void policy_forward(const float obs_norm[POLICY_OBS_DIM], float action[POLICY_ACT_DIM]) {",
        f"    static float buf_a[{max(sizes)}], buf_b[{max(sizes)}];",
        "    const float *in = obs_norm;",
        "    float *out = buf_a;",
    ]
    for i in range(n):
        last = i == n - 1
        dst = "action" if last else "out"
        lines += [
            f"    for (int o = 0; o < {sizes[i + 1]}; o++) {{",
            f"        float s = B{i}[o];",
            f"        for (int k = 0; k < {sizes[i]}; k++) s += W{i}[o * {sizes[i]} + k] * in[k];",
            f"        {dst}[o] = " + ("s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);" if last else "tanhf(s);"),
            "    }",
        ]
        if not last:
            lines.append("    in = out; out = (out == buf_a) ? buf_b : buf_a;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_header(p, out_path, source):
    weights, biases = p["weights"], p["biases"]
    sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    with open(out_path, "w") as f:
        f.write(f"// Auto generated by export_weights.py from {source}\n")
        if "note" in p:
            f.write(f"// ({p['note']})\n")
        f.write("#pragma once\n#include <math.h>\n\n")
        f.write(f"// Network: {' -> '.join(map(str, sizes))}, tanh hidden activations, linear output clipped to [-1, 1]\n")
        f.write(f"#define POLICY_OBS_DIM {sizes[0]}\n#define POLICY_ACT_DIM {sizes[-1]}\n")
        f.write("\n// Where the weights are stored. The firmware defines DRAM_ATTR before including this header:\n"
                "// read from flash they don't fit in the cache and inference is ~10x slower.\n"
                "#ifndef POLICY_WEIGHT_ATTR\n#define POLICY_WEIGHT_ATTR\n#endif\n\n")
        f.write(f"#define POLICY_MAX_TORQUE {p['max_torque']:.6f}f  // Nm, torque = action * POLICY_MAX_TORQUE\n")
        # The command reference must ramp like in the env (0 = no ramp, the reference jumps to the command)
        f.write(f"#define POLICY_V_CMD_ACC_MAX {p['v_cmd_acc_max']:.6f}f  // m/s^2\n")
        f.write(f"#define POLICY_YAWRATE_CMD_ACC_MAX {p['yawrate_cmd_acc_max']:.6f}f  // rad/s^2\n")
        # Command range the policy was trained on: the firmware clamps the remote command to it
        f.write(f"#define POLICY_V_CMD_MAX {p['v_cmd_max']:.6f}f  // m/s\n")
        f.write(f"#define POLICY_YAWRATE_CMD_MAX {p['yawrate_cmd_max']:.6f}f  // rad/s\n\n")
        f.write("// Observation: pitch, pitchrate, yawrate, wheel_vel_left, wheel_vel_right, v_ref, yawrate_ref,\n"
                "//              prev_action_left, prev_action_right\n")
        f.write(c_array("POLICY_OBS_SCALE", p["obs_scale"]) + "\n")
        for i in range(len(weights)):
            f.write(c_array(f"W{i}", weights[i]))
            f.write(c_array(f"B{i}", biases[i]))
        f.write("\n" + policy_forward_c(sizes))
    return sizes


def export(path, out_path=None):
    model_path, policy = load_policy(path)
    out_path = out_path or os.path.splitext(model_path)[0] + ".h"
    source = os.path.join(os.path.basename(os.path.dirname(model_path)), os.path.basename(model_path))
    sizes = write_header(policy, out_path, source)
    print(f"Exported {model_path} ({' -> '.join(map(str, sizes))}) to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="run folder or model file")
    parser.add_argument("-o", "--out", default=None, help="output .h (default: next to the model)")
    args = parser.parse_args()
    export(args.path, args.out)
