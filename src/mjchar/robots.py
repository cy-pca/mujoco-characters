"""Robot profiles: how canonical humanoid bones land on a Menagerie robot.

A profile names, for one robot model:

* ``body_map``: canonical bone -> robot body whose motion drives the skin
  vertices weighted to that bone.  A ``(proximal, distal)`` pair splits the
  bone's weight along its length (used for forearm twist: the G1's wrist-roll
  joint sits mid-forearm).
* ``landmarks``: canonical joint (the head of a bone) -> a point on the robot,
  given as a body name and an offset in that body's frame.  The converter
  re-proportions the character so its joints land on these points.
* ``ik_chains``: joints the converter may move to pose the robot like the
  character's bind pose (T- or A-pose), and which bone segments must line up.

Bodies, joints and offsets are read from Menagerie ``unitree_g1/g1.xml``
(MuJoCo Menagerie v2026.11.2, as packaged by cvcpkg ``mujoco-menagerie``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IKChain:
    joints: tuple[str, ...]
    segments: tuple[tuple[str, str], ...]  # canonical (from, to) joint pairs
    flat_body: str | None = None  # keep this body's world orientation fixed (feet flat)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    model_dir: str  # directory under share/mujoco-menagerie
    scene: str  # scene file inside model_dir
    body_map: dict
    landmarks: dict
    ik_chains: tuple[IKChain, ...]
    sole: dict = field(default_factory=dict)  # side -> (body, offset) of the sole under the ankle
    stand_key: str | None = None
    hide_geom_groups: tuple[int, ...] = (2,)  # visual geoms to hide when only the skin should show


def _sided(fmt: dict, side: str) -> dict:
    lo = side.lower()
    return {k.format(S=side): (v.format(s=lo) if isinstance(v, str) else tuple(x.format(s=lo) for x in v)) for k, v in fmt.items()}


_G1_SIDED_BODIES = {
    "{S}Shoulder": "torso_link",
    "{S}Arm": "{s}_shoulder_yaw_link",
    "{S}ForeArm": ("{s}_elbow_link", "{s}_wrist_roll_link"),
    "{S}Hand": "{s}_wrist_yaw_link",
    "{S}UpLeg": "{s}_hip_yaw_link",
    "{S}Leg": "{s}_knee_link",
    "{S}Foot": "{s}_ankle_roll_link",
    "{S}ToeBase": "{s}_ankle_roll_link",
}

_G1_SIDED_LANDMARKS = {
    "{S}UpLeg": "{s}_hip_pitch_link",  # hip joint centre
    "{S}Leg": "{s}_knee_link",
    "{S}Foot": "{s}_ankle_pitch_link",  # ankle joint centre
    "{S}Arm": "{s}_shoulder_roll_link",  # shoulder centre
    "{S}ForeArm": "{s}_elbow_link",
    "{S}Hand": "{s}_wrist_pitch_link",  # wrist centre (pitch/yaw axes)
}


def _g1() -> RobotProfile:
    body_map = {"Hips": "pelvis", "Spine": "torso_link", "Spine1": "torso_link", "Spine2": "torso_link", "Neck": "torso_link", "Head": "torso_link"}
    landmarks = {}
    for side in ("Left", "Right"):
        body_map.update(_sided(_G1_SIDED_BODIES, side))
        landmarks.update({k: (v, (0.0, 0.0, 0.0)) for k, v in _sided(_G1_SIDED_LANDMARKS, side).items()})
    chains = []
    for side in ("Left", "Right"):
        s = side.lower()
        chains.append(
            IKChain(
                joints=(f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint", f"{s}_elbow_joint"),
                segments=((f"{side}Arm", f"{side}ForeArm"), (f"{side}ForeArm", f"{side}Hand")),
            )
        )
        chains.append(
            IKChain(
                joints=(
                    f"{s}_hip_pitch_joint",
                    f"{s}_hip_roll_joint",
                    f"{s}_hip_yaw_joint",
                    f"{s}_knee_joint",
                    f"{s}_ankle_pitch_joint",
                    f"{s}_ankle_roll_joint",
                ),
                segments=((f"{side}UpLeg", f"{side}Leg"), (f"{side}Leg", f"{side}Foot")),
                flat_body=f"{s}_ankle_roll_link",
            )
        )
    # foot contact spheres sit at z=-0.03 (radius 0.005) in the ankle-roll frame
    sole = {side: (f"{side.lower()}_ankle_roll_link", (0.0, 0.0, -0.035)) for side in ("Left", "Right")}
    return RobotProfile(
        name="unitree_g1",
        model_dir="unitree_g1",
        scene="scene.xml",
        body_map=body_map,
        landmarks=landmarks,
        ik_chains=tuple(chains),
        sole=sole,
        stand_key="stand",
    )


PROFILES = {"g1": _g1, "unitree_g1": _g1}


def get_profile(name: str) -> RobotProfile:
    try:
        return PROFILES[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; known: {sorted(set(PROFILES))}") from None


def menagerie_root(explicit: str | os.PathLike | None = None) -> Path:
    """Find ``share/mujoco-menagerie`` from cvcpkg's ``mujoco-menagerie`` package.

    Search order: explicit argument, ``$MUJOCO_MENAGERIE``, ``$CVCPKG_PREFIX``,
    then the running interpreter's prefix (a cvcpkg prefix's own python.exe).
    """
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("MUJOCO_MENAGERIE"):
        cands.append(Path(os.environ["MUJOCO_MENAGERIE"]))
    for var in ("CVCPKG_PREFIX",):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]) / "share" / "mujoco-menagerie")
    for p in {sys.prefix, sys.base_prefix}:
        cands.append(Path(p) / "share" / "mujoco-menagerie")
    for c in cands:
        if (c / "unitree_g1").is_dir():
            return c
    raise FileNotFoundError(
        "mujoco-menagerie not found; install it into your cvcpkg prefix "
        "(cvcpkg install mujoco-menagerie --prefix <prefix>) or set MUJOCO_MENAGERIE"
    )
