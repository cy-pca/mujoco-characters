"""Humanoid bone naming: map exporter-specific bone names to canonical names.

Canonical names are Mixamo's, without the ``mixamorig:`` prefix (``Hips``,
``Spine``, ``Spine1``, ``Spine2``, ``Neck``, ``Head``, ``LeftShoulder``,
``LeftArm``, ``LeftForeArm``, ``LeftHand``, ``LeftUpLeg``, ``LeftLeg``,
``LeftFoot``, ``LeftToeBase`` and the right-side twins).

Meshy can rig with several skeleton templates.  Its API default, ``meshyrig``,
is a 24-bone skeleton with Mixamo-like names (``Spine01``/``Spine02``,
``neck``, ``head_end``, ``headfront``); its ``mixamo`` template uses
``mixamorig:``-prefixed names; ``ue5_mannequin`` and ``unity_humanoid`` follow
those engines.  The alias table below covers all four.  Anything else can be
mapped with an explicit ``{"source bone": "CanonicalName"}`` JSON override.
"""

from __future__ import annotations

import re

SIDES = ("Left", "Right")

# Core bones that robot profiles bind to.  Order is parent-before-child.
CORE = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    *[f"{s}{b}" for s in SIDES for b in ("Shoulder", "Arm", "ForeArm", "Hand")],
    *[f"{s}{b}" for s in SIDES for b in ("UpLeg", "Leg", "Foot", "ToeBase")],
]

# Canonical parent of each core bone (used when a skeleton skips a bone).
CANONICAL_PARENT = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
}
for _s in SIDES:
    CANONICAL_PARENT.update(
        {
            f"{_s}Shoulder": "Spine2",
            f"{_s}Arm": f"{_s}Shoulder",
            f"{_s}ForeArm": f"{_s}Arm",
            f"{_s}Hand": f"{_s}ForeArm",
            f"{_s}UpLeg": "Hips",
            f"{_s}Leg": f"{_s}UpLeg",
            f"{_s}Foot": f"{_s}Leg",
            f"{_s}ToeBase": f"{_s}Foot",
        }
    )

# side-free aliases: normalised base name -> canonical base name
_BASE_ALIASES = {
    # trunk
    "hips": "Hips",
    "hip": "Hips",
    "pelvis": "Hips",
    "root": None,  # explicit non-bone (UE5 'root' sits at the floor)
    "spine": "Spine",
    "spine1": "Spine1",
    "spine2": "Spine2",
    "spine3": "Spine2",  # UE5 spine_03 (closest)
    "spine4": "Spine2",
    "spine5": "Spine2",
    "chest": "Spine1",  # Unity
    "upperchest": "Spine2",  # Unity
    "neck": "Neck",
    "neck1": "Neck",  # UE5 neck_01
    "neck2": "Neck",
    "head": "Head",
    # arms
    "shoulder": "Shoulder",
    "clavicle": "Shoulder",
    "arm": "Arm",
    "upperarm": "Arm",
    "forearm": "ForeArm",
    "lowerarm": "ForeArm",
    "hand": "Hand",
    # legs
    "upleg": "UpLeg",
    "upperleg": "UpLeg",
    "thigh": "UpLeg",
    "leg": "Leg",
    "lowerleg": "Leg",
    "calf": "Leg",
    "shin": "Leg",
    "foot": "Foot",
    "toebase": "ToeBase",
    "toes": "ToeBase",
    "toe": "ToeBase",
    "ball": "ToeBase",
}

_SIDED = {"Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase"}

_PREFIX_RE = re.compile(r"^(mixamorig\d*[:_]?|armature[|:_/]|.*[:|/])", re.IGNORECASE)


def strip_namespace(name: str) -> str:
    """Drop exporter namespaces such as ``mixamorig:`` or ``Armature|``."""
    prev = None
    out = name.strip()
    while out != prev:
        prev = out
        out = _PREFIX_RE.sub("", out)
    return out


def _split_side(name: str) -> tuple[str | None, str]:
    """Return (side, remainder) for names like LeftArm, upperarm_l, L_Arm, Arm.L."""
    n = name
    low = n.lower()
    for side, long, short in (("Left", "left", "l"), ("Right", "right", "r")):
        if low.startswith(long):
            return side, n[len(long) :]
        if low.endswith(long):
            return side, n[: -len(long)]
        m = re.match(rf"^{short}[_.\-\s](.+)$", n, re.IGNORECASE)
        if m:
            return side, m.group(1)
        m = re.match(rf"^(.+?)[_.\-\s]{short}$", n, re.IGNORECASE)
        if m:
            return side, m.group(1)
    return None, n


def _normalise_base(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]", "", s).lower()
    return re.sub(r"0+(\d)", r"\1", s)  # spine01 -> spine1, neck_01 -> neck1


def canonical_name(name: str) -> str | None:
    """Canonical core-bone name for an exporter bone name, or None.

    Finger, twist and end bones return None; they are bound through their
    nearest mapped ancestor.
    """
    raw = strip_namespace(name)
    side, rest = _split_side(raw)
    base = _normalise_base(rest)
    if base not in _BASE_ALIASES:
        return None
    canon = _BASE_ALIASES[base]
    if canon is None:
        return None
    if canon in _SIDED:
        return f"{side}{canon}" if side else None
    if side:  # e.g. "LeftHips" is not a thing
        return None
    return canon


def map_skeleton(bone_names: list[str], parents: list[int], overrides: dict[str, str] | None = None) -> dict[int, str]:
    """Map skeleton bones (by index) to canonical names.

    ``parents[i]`` is the index of bone i's parent inside the same list, or -1.
    When two bones claim the same canonical name (UE5 spine_01..05 all near the
    chest), the one closest to the root keeps it.
    """
    overrides = overrides or {}
    claimed: dict[str, int] = {}
    depth = []
    for i in range(len(bone_names)):
        d, p = 0, parents[i]
        while p >= 0:
            d, p = d + 1, parents[p]
        depth.append(d)
    order = sorted(range(len(bone_names)), key=lambda i: depth[i])
    for i in order:
        name = bone_names[i]
        canon = overrides.get(name, canonical_name(name))
        if canon is None or canon not in CANONICAL_PARENT:
            continue
        if canon in claimed:
            continue
        claimed[canon] = i
    # UE5-style spine_01..spine_05: if Spine was not claimed but Spine1 was by a
    # bone directly under Hips, shift the trunk names down one.
    if "Spine" not in claimed and "Spine1" in claimed:
        hips = claimed.get("Hips")
        s1 = claimed["Spine1"]
        if hips is not None and parents[s1] == hips:
            claimed["Spine"] = claimed.pop("Spine1")
            if "Spine2" in claimed:
                # leave Spine2 on the bone it found; fill Spine1 from its parent
                p = parents[claimed["Spine2"]]
                if p >= 0 and p != claimed["Spine"]:
                    claimed["Spine1"] = p
    return {i: c for c, i in claimed.items()}
