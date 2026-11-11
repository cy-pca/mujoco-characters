"""G1 with a character skin: PD stand plus a scripted routine, in the viewer or rendered.

The robot's own position servos (Menagerie G1, kp=500) track a scripted target:
settle, squat, wave, twist, then march in place with a quasi-static weight
shift.  No learned policy; the point is to show the skin following the robot.

  python examples/demo_g1.py out/unitree_g1_meshy.xml --viewer
  python examples/demo_g1.py out/unitree_g1_meshy.xml --render evidence/   # gif + contact sheet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------- routine


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _env(t: float, t0: float, t1: float, ramp: float = 0.5) -> float:
    """1 inside [t0, t1], easing in/out over `ramp` seconds."""
    return _smooth((t - t0) / ramp) * _smooth((t1 - t) / ramp)


PHASES = [  # name, start, end (seconds)
    ("settle", 0.0, 1.0),
    ("squat", 1.0, 4.5),
    ("wave", 4.5, 8.5),
    ("twist", 8.5, 11.5),
    ("march", 11.5, 19.5),
    ("settle", 19.5, 21.0),
]
DURATION = PHASES[-1][2]


MARCH_PERIOD = 2.0


def march_shift(t: float) -> float:
    """Planned CoM shift toward the right foot (+1) or left foot (-1) while marching."""
    e = _env(t, 11.5, 19.5, ramp=1.0)
    return 0.9 * e * np.sin(2 * np.pi * (t - 11.5) / MARCH_PERIOD) if e else 0.0


def routine(t: float) -> dict[str, float]:
    """Joint-target offsets (radians) from the stand keyframe at time t."""
    o: dict[str, float] = {}

    def add(j, v):
        o[j] = o.get(j, 0.0) + v

    # squat twice
    e = _env(t, 1.0, 4.5)
    if e:
        a = 0.45 * e * (0.5 - 0.5 * np.cos(2 * np.pi * (t - 1.0) / 1.75))
        for s in ("left", "right"):
            add(f"{s}_hip_pitch_joint", -a)
            add(f"{s}_knee_joint", 2 * a)
            add(f"{s}_ankle_pitch_joint", -a)
        add("left_shoulder_pitch_joint", -0.8 * a)
        add("right_shoulder_pitch_joint", -0.8 * a)
    # wave with the right arm, left arm swings
    e = _env(t, 4.5, 8.5)
    if e:
        add("right_shoulder_roll_joint", -1.25 * e)
        add("right_shoulder_yaw_joint", 0.4 * e)
        add("right_elbow_joint", e * (-0.6 + 0.45 * np.sin(2 * np.pi * 1.5 * (t - 4.5))))
        add("right_wrist_roll_joint", 0.5 * e * np.sin(2 * np.pi * 1.5 * (t - 4.5)))
        add("left_shoulder_pitch_joint", 0.35 * e * np.sin(2 * np.pi * 0.75 * (t - 4.5)))
        add("left_elbow_joint", -0.4 * e)
    # twist at the waist, head-and-shoulders turn
    e = _env(t, 8.5, 11.5)
    if e:
        w = np.sin(2 * np.pi * 0.66 * (t - 8.5))
        add("waist_yaw_joint", 0.6 * e * w)
        add("waist_pitch_joint", 0.15 * e)
        add("left_shoulder_pitch_joint", -0.5 * e * w)
        add("right_shoulder_pitch_joint", 0.5 * e * w)
    # march in place: shift weight over the stance foot, then lift the other
    e = _env(t, 11.5, 19.5, ramp=1.0)
    if e:
        ph = 2 * np.pi * (t - 11.5) / MARCH_PERIOD
        sway = 0.11 * e * np.sin(ph)  # + -> pelvis over the right foot
        for s in ("left", "right"):
            add(f"{s}_hip_roll_joint", sway)
            add(f"{s}_ankle_roll_joint", -sway)
        # lift the unloaded foot while the sway is near its peak
        lift_l = 0.35 * e * max(0.0, np.sin(ph)) ** 4
        lift_r = 0.35 * e * max(0.0, -np.sin(ph)) ** 4
        for s, lift in (("left", lift_l), ("right", lift_r)):
            add(f"{s}_hip_pitch_joint", -lift)
            add(f"{s}_knee_joint", 2 * lift)
            add(f"{s}_ankle_pitch_joint", -lift)
        add("left_shoulder_pitch_joint", 0.3 * e * np.sin(ph))
        add("right_shoulder_pitch_joint", -0.3 * e * np.sin(ph))
    return o


class Controller:
    def __init__(self, model: mujoco.MjModel, key: str | None = "stand"):
        self.model = model
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key) if key else -1
        self.key = kid
        self.base = model.key_ctrl[kid].copy() if kid >= 0 else np.zeros(model.nu)
        self.act = {}
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            self.act[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = a
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in ("left", "right")]
        self.prev_com = None
        # ankle strategy gains: rad of ankle correction per metre of CoM error (and per m/s)
        self.kp_bal, self.kd_bal = 2.0, 0.35

    def reset(self, data: mujoco.MjData):
        self.prev_com = None
        if self.key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, data, self.key)
        else:
            mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)

    def com_target(self, data: mujoco.MjData) -> np.ndarray:
        """Where the CoM should be: over the feet, shifted over the stance foot while marching."""
        lf, rf = data.site_xpos[self.feet[0]], data.site_xpos[self.feet[1]]
        mid = 0.5 * (lf + rf)
        fwd = data.site_xmat[self.feet[0]].reshape(3, 3)[:, 0]
        target = mid[:2] + 0.03 * fwd[:2]  # foot centre is ~3 cm ahead of the ankle
        target[1] += (rf[1] - lf[1]) * 0.5 * march_shift(data.time)
        return target

    def __call__(self, data: mujoco.MjData):
        ctrl = self.base.copy()
        for j, v in routine(data.time).items():
            if j in self.act:
                ctrl[self.act[j]] += v
        if self.feet[0] >= 0 and self.feet[1] >= 0:
            com = data.subtree_com[1, :2].copy()
            vel = np.zeros(2) if self.prev_com is None else (com - self.prev_com) / self.model.opt.timestep
            self.prev_com = com
            err = com - self.com_target(data)
            dpitch = self.kp_bal * err[0] + self.kd_bal * vel[0]
            droll = -(self.kp_bal * err[1] + self.kd_bal * vel[1])
            for s in ("left", "right"):
                ctrl[self.act[f"{s}_ankle_pitch_joint"]] += np.clip(dpitch, -0.3, 0.3)
                ctrl[self.act[f"{s}_ankle_roll_joint"]] += np.clip(droll, -0.2, 0.2)
        lo, hi = self.model.actuator_ctrlrange.T
        limited = self.model.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(ctrl[limited], lo[limited], hi[limited])
        data.ctrl[:] = ctrl


def phase_at(t: float) -> str:
    for name, t0, t1 in PHASES:
        if t0 <= t < t1:
            return name
    return PHASES[-1][0]


# --------------------------------------------------------------------------- runners


def load(xml: Path):
    model = mujoco.MjModel.from_xml_path(str(xml))
    meta_path = xml.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, meta


def simulate(model, ctl, seconds=DURATION, on_frame=None, fps=30):
    data = mujoco.MjData(model)
    ctl.reset(data)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    next_frame = 0.0
    stats = {"min_pelvis_z": 9.0, "max_tilt_deg": 0.0, "fell": False}
    while data.time < seconds:
        ctl(data)
        mujoco.mj_step(model, data)
        z = data.xpos[pelvis][2]
        tilt = np.degrees(np.arccos(np.clip(data.xmat[torso][8], -1, 1)))
        stats["min_pelvis_z"] = min(stats["min_pelvis_z"], z)
        stats["max_tilt_deg"] = max(stats["max_tilt_deg"], tilt)
        if z < 0.45:
            stats["fell"] = True
            stats["fell_at"] = round(data.time, 2)
            break
        if on_frame and data.time >= next_frame:
            on_frame(data)
            next_frame += 1.0 / fps
    return data, stats


def run_viewer(model, meta, ctl, hide_robot=True, seconds=None):
    import mujoco.viewer

    data = mujoco.MjData(model)
    ctl.reset(data)
    with mujoco.viewer.launch_passive(model, data) as v:
        if hide_robot:
            for g in meta.get("hide_geom_groups", [2]):
                v.opt.geomgroup[g] = 0
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.0, -12.0, 150.0
        start = time.time()
        while v.is_running():
            if seconds is not None and time.time() - start > seconds:
                break
            t_wall = time.time()
            if data.time >= DURATION:
                ctl.reset(data)
            for _ in range(int(round(1 / 60 / model.opt.timestep))):
                ctl(data)
                mujoco.mj_step(model, data)
            with v.lock():
                v.cam.lookat[:] = data.xpos[1] + np.array([0, 0, -0.1])
            v.sync()
            dt = 1 / 60 - (time.time() - t_wall)
            if dt > 0:
                time.sleep(dt)


def _ffmpeg() -> str | None:
    """ffmpeg from PATH or from the cvcpkg prefix (cvcpkg install ffmpeg-cli)."""
    import shutil
    import sys

    for cand in (shutil.which("ffmpeg"), Path(sys.prefix) / "bin" / "ffmpeg.exe", Path(sys.prefix) / "bin" / "ffmpeg"):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def render(model, meta, ctl, out_dir: Path, pane=(480, 540), fps=30, gif_fps=10, gif_scale=0.5):
    """MP4 + GIF (robot | skin side by side), a contact sheet of key moments, and one still."""
    import subprocess

    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    pw, ph = pane
    renderer = mujoco.Renderer(model, ph, pw)
    big = mujoco.Renderer(model, 720, 640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.1, -10.0, 150.0
    cam.lookat[:] = (0.0, 0.0, 0.62)
    opt_skin = mujoco.MjvOption()
    for g in meta.get("hide_geom_groups", [2]):
        opt_skin.geomgroup[g] = 0
    opt_robot = mujoco.MjvOption()
    opt_robot.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    opt_both = mujoco.MjvOption()

    ff = _ffmpeg()
    mp4 = out_dir / "g1_skin_demo.mp4"
    enc = None
    if ff:
        enc = subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * pw}x{ph}", "-r", str(fps), "-i", "pipe:0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(mp4)],
            stdin=subprocess.PIPE,
        )
    gif_frames = []
    sheet_times = [0.5, 2.0, 5.6, 6.0, 9.2, 10.0, 13.0, 14.0]
    sheet = {}
    still = {}
    count = [0]

    def view(r, data, opt):
        r.update_scene(data, cam, opt)
        return r.render().copy()

    def grab(data):
        im = Image.fromarray(np.concatenate([view(renderer, data, opt_robot), view(renderer, data, opt_skin)], axis=1))
        ImageDraw.Draw(im).text((8, 8), f"t={data.time:5.2f}s  {phase_at(data.time)}   robot | character skin", fill=(255, 255, 255))
        if enc:
            enc.stdin.write(im.tobytes())
        if count[0] % max(1, round(fps / gif_fps)) == 0:
            gif_frames.append(im.resize((int(im.width * gif_scale), int(im.height * gif_scale))))
        count[0] += 1
        for st in sheet_times:
            if st not in sheet and data.time >= st:
                sheet[st] = ([view(renderer, data, o) for o in (opt_robot, opt_both, opt_skin)], phase_at(data.time))
        if "wave" not in still and data.time >= 6.0:
            still["wave"] = np.concatenate([view(big, data, opt_robot), view(big, data, opt_skin)], axis=1)

    _, stats = simulate(model, ctl, on_frame=grab, fps=fps)
    if enc:
        enc.stdin.close()
        enc.wait()
    gif = out_dir / "g1_skin_demo.gif"
    # one shared palette and no dithering, so unchanged pixels stay unchanged between frames
    picks = gif_frames[:: max(1, len(gif_frames) // 8)]
    mosaic = Image.new("RGB", (gif_frames[0].width, gif_frames[0].height * len(picks)))
    for i, f in enumerate(picks):
        mosaic.paste(f, (0, i * f.height))
    palette = mosaic.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in gif_frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=int(1000 / gif_fps), loop=0)
    # contact sheet: two moments per row, each robot | robot + skin | skin only
    rows = []
    for st in sorted(sheet):
        row, phase = sheet[st]
        r = Image.fromarray(np.concatenate(row, axis=1))
        ImageDraw.Draw(r).text((8, 8), f"t={st:.1f}s {phase}:  robot | robot + skin | skin only", fill=(255, 255, 255))
        rows.append(r)
    if rows:
        cs = Image.new("RGB", (rows[0].width * 2, rows[0].height * ((len(rows) + 1) // 2)))
        for i, r in enumerate(rows):
            cs.paste(r, ((i % 2) * r.width, (i // 2) * r.height))
        cs = cs.resize((cs.width // 2, cs.height // 2))
        cs.save(out_dir / "g1_skin_contact_sheet.png")
    if "wave" in still:
        Image.fromarray(still["wave"]).save(out_dir / "g1_skin_wave.png")
    outputs = [gif] + ([mp4] if enc else [])
    return outputs, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml", type=Path, help="scene written by `python -m mjchar skin`")
    ap.add_argument("--viewer", action="store_true", help="open the interactive viewer")
    ap.add_argument("--show-robot", action="store_true", help="viewer: draw the robot under the skin")
    ap.add_argument("--seconds", type=float, help="viewer: close after this many seconds")
    ap.add_argument("--render", type=Path, help="write an MP4 (needs ffmpeg), a GIF, a contact sheet and a still to this folder")
    a = ap.parse_args()

    model, meta = load(a.xml)
    ctl = Controller(model, meta.get("stand_key", "stand"))
    if a.render:
        outputs, stats = render(model, meta, ctl, a.render)
        for o in outputs:
            print(f"wrote {o}")
        print(json.dumps(stats))
    if a.viewer:
        run_viewer(model, meta, ctl, hide_robot=not a.show_robot, seconds=a.seconds)
    if not a.render and not a.viewer:
        _, stats = simulate(model, ctl)
        print(json.dumps(stats))


if __name__ == "__main__":
    main()
