"""Level 1: bind a rigged humanoid mesh to a robot as a MuJoCo ``<skin>``.

Pipeline
--------
1. ``load_character``: read the GLB, rebuild the mesh in its bind pose in world
   space (metres), convert glTF axes (Y up, facing +Z) to MuJoCo's (Z up,
   facing +X), and map bone names to canonical names.
2. ``solve_reference_pose``: pose the robot like the character's bind pose
   (T- or A-pose) with a small per-limb least-squares IK on segment directions.
3. ``fit_character``: scale the character to the robot's leg length and
   re-proportion each limb so the character's joints sit on the robot's joints
   (per-bone affine maps blended by the skin weights).
4. ``bind_skin``: merge the character's bone weights onto robot bodies and take
   each body's pose in the reference configuration as its ``bindpos`` /
   ``bindquat``.  MuJoCo then skins the mesh every frame with
   ``v = X_body(t) * X_body(bind)^-1 * v_bind``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import bones as B
from . import gltf
from .robots import RobotProfile

# glTF (x left, y up, z forward) -> MuJoCo (x forward, y left, z up)
GLTF_TO_MJ = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# --------------------------------------------------------------------------- data


@dataclass
class Character:
    name: str
    vertices: np.ndarray  # (n, 3) bind pose, MuJoCo axes, metres
    faces: np.ndarray  # (m, 3)
    texcoords: np.ndarray | None  # (n, 2), glTF convention (v down)
    bone_names: list[str]
    bone_parents: list[int]
    bone_heads: np.ndarray  # (nb, 3) joint positions in bind pose, MuJoCo axes
    joints: np.ndarray  # (n, k) bone indices
    weights: np.ndarray  # (n, k) normalised
    canonical: dict[int, str]  # bone index -> canonical name
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    image: gltf.Image | None = None
    notes: list[str] = field(default_factory=list)

    def bone(self, canon: str) -> int | None:
        for i, c in self.canonical.items():
            if c == canon:
                return i
        return None

    def head(self, canon: str) -> np.ndarray | None:
        i = self.bone(canon)
        return None if i is None else self.bone_heads[i]

    def canonical_ancestor(self, i: int) -> str | None:
        """Canonical name of bone i or of its nearest mapped ancestor."""
        while i >= 0:
            if i in self.canonical:
                return self.canonical[i]
            i = self.bone_parents[i]
        return None

    def height(self) -> float:
        return float(self.vertices[:, 2].max() - self.vertices[:, 2].min())


@dataclass
class SkinBinding:
    vertices: np.ndarray  # (n, 3) skin-space vertices (= world, reference pose)
    faces: np.ndarray
    texcoords: np.ndarray | None
    bodies: list[str]
    bindpos: np.ndarray  # (nbody, 3)
    bindquat: np.ndarray  # (nbody, 4) wxyz
    vertid: list[np.ndarray]
    vertweight: list[np.ndarray]
    qpos_ref: np.ndarray
    report: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- math helpers


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- loading


def load_character(path, overrides: dict[str, str] | None = None, name: str | None = None) -> Character:
    g = gltf.load(path)
    notes: list[str] = []
    mesh_nodes = g.skinned_mesh_nodes()
    if not mesh_nodes:
        raise gltf.GltfError(f"{path}: no skinned mesh (is the character rigged?)")
    skin_idx = mesh_nodes[0].skin
    if any(n.skin != skin_idx for n in mesh_nodes):
        notes.append("several skins found; only meshes using the first one are bound")
        mesh_nodes = [n for n in mesh_nodes if n.skin == skin_idx]
    skin = g.skins[skin_idx]
    G = g.global_transforms()
    J = skin.joints
    K = np.einsum("jab,jbc->jac", G[J], skin.inverse_bind)  # joint world * IBM; = mesh->world at bind
    spread = float(np.abs(K - K[0]).max() / max(1e-9, np.abs(K[0]).max()))
    if spread < 1e-3:
        mesh_to_world = K.mean(axis=0)
        heads = G[J][:, :3, 3]
    else:
        notes.append(f"stored node pose differs from the bind pose (spread {spread:.3g}); using inverse bind matrices")
        nodeset = set(J)
        roots = [k for k, j in enumerate(J) if g.nodes[j].parent not in nodeset]
        r = roots[0]
        mesh_to_world = G[J[r]] @ skin.inverse_bind[r]
        heads = np.array([(mesh_to_world @ np.linalg.inv(ibm))[:3, 3] for ibm in skin.inverse_bind])

    verts, faces, uvs, jidx, wts = [], [], [], [], []
    kmax = max(p.joints.shape[1] for n in mesh_nodes for p in g.meshes[n.mesh] if p.joints is not None)
    base = 0
    material = None
    has_uv = all(p.texcoords is not None for n in mesh_nodes for p in g.meshes[n.mesh])
    for n in mesh_nodes:
        for p in g.meshes[n.mesh]:
            if p.joints is None:
                notes.append(f"mesh '{n.name}' has a primitive without skin weights; skipped")
                continue
            v = np.c_[p.positions, np.ones(len(p.positions))] @ mesh_to_world.T
            verts.append(v[:, :3])
            faces.append(p.faces + base)
            if has_uv:
                uvs.append(p.texcoords)
            pj = np.zeros((len(p.positions), kmax), dtype=np.int64)
            pw = np.zeros((len(p.positions), kmax))
            pj[:, : p.joints.shape[1]] = p.joints
            pw[:, : p.weights.shape[1]] = p.weights
            jidx.append(pj)
            wts.append(pw)
            base += len(p.positions)
            if material is None and p.material is not None:
                material = g.materials[p.material]
    V = np.concatenate(verts) @ GLTF_TO_MJ.T
    heads = heads @ GLTF_TO_MJ.T
    W = np.concatenate(wts)
    W = np.where(W > 0, W, 0.0)
    s = W.sum(axis=1, keepdims=True)
    if (s <= 0).any():
        notes.append(f"{int((s <= 0).sum())} vertices had no skin weight; bound to the root bone")
        W[(s <= 0)[:, 0], 0] = 1.0
        s = W.sum(axis=1, keepdims=True)
    W = W / s

    names = [g.nodes[j].name for j in J]
    pos_in_skin = {j: k for k, j in enumerate(J)}
    parents = []
    for j in J:
        p = g.nodes[j].parent
        while p >= 0 and p not in pos_in_skin:
            p = g.nodes[p].parent
        parents.append(pos_in_skin.get(p, -1))
    canonical = B.map_skeleton(names, parents, overrides)

    ch = Character(
        name=name or (g.path.stem if g.path else "character"),
        vertices=V,
        faces=np.concatenate(faces),
        texcoords=np.concatenate(uvs) if has_uv else None,
        bone_names=names,
        bone_parents=parents,
        bone_heads=heads,
        joints=np.concatenate(jidx),
        weights=W,
        canonical=canonical,
        notes=notes,
    )
    if material is not None:
        ch.base_color = tuple(material.base_color)
        if material.base_color_image is not None:
            ch.image = g.images[material.base_color_image]
    _face_forward(ch)
    return ch


def _face_forward(ch: Character) -> None:
    """Rotate about +Z so the character's left (+Y) side is where its left bones are."""
    pairs = [("LeftUpLeg", "RightUpLeg"), ("LeftArm", "RightArm"), ("LeftShoulder", "RightShoulder")]
    for l, r in pairs:
        hl, hr = ch.head(l), ch.head(r)
        if hl is not None and hr is not None:
            lat = hl - hr
            lat[2] = 0.0
            if np.linalg.norm(lat) < 1e-6:
                continue
            yaw = np.arctan2(lat[1], lat[0]) - np.pi / 2  # angle of lateral from +Y
            if abs(yaw) > np.radians(1.0):
                R = rot_z(-yaw)
                ch.vertices = ch.vertices @ R.T
                ch.bone_heads = ch.bone_heads @ R.T
                ch.notes.append(f"character faced {np.degrees(yaw):+.1f} deg off glTF +Z; rotated to face +X")
            return
    ch.notes.append("no left/right bone pair found; assuming the glTF +Z facing convention")


# --------------------------------------------------------------------------- robot pose


class RobotKinematics:
    def __init__(self, model: mujoco.MjModel, profile: RobotProfile):
        self.model = model
        self.profile = profile
        self.data = mujoco.MjData(model)
        self.qpos = model.qpos0.copy()

    def forward(self, qpos=None):
        if qpos is not None:
            self.qpos = np.array(qpos, dtype=float)
        self.data.qpos[:] = self.qpos
        mujoco.mj_kinematics(self.model, self.data)

    def body(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(f"robot has no body {name!r}")
        return i

    def point(self, body: str, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        b = self.body(body)
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ np.asarray(offset, dtype=float)

    def landmark(self, canon: str) -> np.ndarray:
        body, off = self.profile.landmarks[canon]
        return self.point(body, off)


def solve_reference_pose(kin: RobotKinematics, ch: Character, iters: int = 60) -> tuple[np.ndarray, list[str]]:
    """Joint angles that make the robot's limb segments parallel to the character's."""
    m = kin.model
    report = []
    kin.forward(m.qpos0)
    q = m.qpos0.copy()
    flat_ref = {}
    for chain in kin.profile.ik_chains:
        if chain.flat_body:
            flat_ref[chain.flat_body] = kin.data.xmat[kin.body(chain.flat_body)].reshape(3, 3).copy()

    for chain in kin.profile.ik_chains:
        segs = []
        for a, b in chain.segments:
            ha, hb = ch.head(a), ch.head(b)
            if ha is None or hb is None:
                continue
            segs.append((a, b, _unit(hb - ha)))
        if not segs:
            report.append(f"IK chain {chain.joints[0]}..: no matching character bones, left at zero")
            continue
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in chain.joints]
        if min(jids) < 0:
            missing = [j for j, i in zip(chain.joints, jids) if i < 0]
            raise KeyError(f"robot has no joints {missing}")
        adr = np.array([m.jnt_qposadr[j] for j in jids])
        lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jids])
        hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jids])
        x0 = np.clip(q[adr], lo, hi)

        def residual(x):
            qq = q.copy()
            qq[adr] = x
            kin.forward(qq)
            r = []
            for a, b, dc in segs:
                dg = _unit(kin.landmark(b) - kin.landmark(a))
                r.extend(dg - dc)
            if chain.flat_body:
                R = kin.data.xmat[kin.body(chain.flat_body)].reshape(3, 3)
                r.extend(0.5 * (R - flat_ref[chain.flat_body]).ravel())
            r.extend(0.02 * (x - x0))
            return np.array(r)

        x = x0.copy()
        lam = 1e-3
        f = residual(x)
        for _ in range(iters):
            Jm = np.empty((len(f), len(x)))
            for k in range(len(x)):
                dx = np.zeros_like(x)
                dx[k] = 1e-5
                Jm[:, k] = (residual(x + dx) - f) / 1e-5
            step = np.linalg.solve(Jm.T @ Jm + lam * np.eye(len(x)), -Jm.T @ f)
            xn = np.clip(x + step, lo, hi)
            fn = residual(xn)
            if fn @ fn < f @ f:
                x, f, lam = xn, fn, max(lam * 0.3, 1e-7)
                if np.abs(step).max() < 1e-7:
                    break
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        q[adr] = x
        kin.forward(q)
        worst = 0.0
        for a, b, dc in segs:
            dg = _unit(kin.landmark(b) - kin.landmark(a))
            worst = max(worst, np.degrees(np.arccos(np.clip(dg @ dc, -1, 1))))
        report.append(f"IK {chain.joints[0].rsplit('_', 2)[0]}: max segment angle error {worst:.1f} deg")
    kin.forward(q)
    return q, report


# --------------------------------------------------------------------------- fitting


@dataclass
class _Affine:
    M: np.ndarray
    t: np.ndarray
    R: np.ndarray | None = None  # rotation part, when the map is rotation * scale

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return v @ self.M.T + self.t


def _segment_map(c0, c1, g0, g1, axial: bool = True) -> _Affine:
    ac, ag = _unit(c1 - c0), _unit(g1 - g0)
    R = rotation_between(ac, ag)
    sig = np.linalg.norm(g1 - g0) / max(np.linalg.norm(c1 - c0), 1e-9) if axial else 1.0
    M = R @ (np.eye(3) + (sig - 1.0) * np.outer(ac, ac))
    return _Affine(M, g0 - M @ c0, R)


def _rigid_at(c0, g0, R, S=None) -> _Affine:
    M = R if S is None else R @ S
    return _Affine(M, g0 - M @ c0, R)


def fit_character(ch: Character, kin: RobotKinematics, fit_limbs: bool = True) -> tuple[np.ndarray, dict[str, _Affine], list[str]]:
    """Scale and re-proportion the character onto the robot in its current pose.

    Returns fitted vertices, the per-canonical-bone maps, and report lines.
    """
    rep = []
    need = ["LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]
    missing = [n for n in need if ch.head(n) is None]
    if missing:
        raise ValueError(f"character skeleton lacks {missing}; add a --bone-map override")

    def lm(n):
        return kin.landmark(n)

    # 1. uniform scale from leg length, translation from the hip centre
    def leg(h, side):
        return np.linalg.norm(h(f"{side}UpLeg") - h(f"{side}Leg")) + np.linalg.norm(h(f"{side}Leg") - h(f"{side}Foot"))

    lc = 0.5 * (leg(ch.head, "Left") + leg(ch.head, "Right"))
    lg = 0.5 * (leg(lm, "Left") + leg(lm, "Right"))
    s = lg / lc
    mid_c = 0.5 * (ch.head("LeftUpLeg") + ch.head("RightUpLeg"))
    mid_g = 0.5 * (lm("LeftUpLeg") + lm("RightUpLeg"))
    rep.append(f"character height {ch.height():.3f} m, leg {lc:.3f} m -> scale {s:.3f} (robot leg {lg:.3f} m)")

    V = s * (ch.vertices - mid_c) + mid_g
    H = s * (ch.bone_heads - mid_c) + mid_g

    def hc(n):
        i = ch.bone(n)
        return None if i is None else H[i]

    maps: dict[str, _Affine] = {}
    ident = _Affine(np.eye(3), np.zeros(3))
    if not fit_limbs:
        for c in B.CANONICAL_PARENT:
            maps[c] = ident
        return V, maps, rep

    # 2. trunk: one frame from hip centre to shoulder centre, lateral scale
    #    blended from the hip-width ratio to the shoulder-width ratio
    sh_c = 0.5 * (hc("LeftArm") + hc("RightArm"))
    sh_g = 0.5 * (lm("LeftArm") + lm("RightArm"))
    up_c, up_g = sh_c - mid_g, sh_g - mid_g
    Rt = rotation_between(up_c, up_g)
    sv = np.linalg.norm(up_g) / np.linalg.norm(up_c)
    u = _unit(up_c)

    def lateral(a, b):
        lat = a - b
        lat = lat - (lat @ u) * u
        return lat

    hipw_c, hipw_g = np.linalg.norm(lateral(hc("LeftUpLeg"), hc("RightUpLeg"))), np.linalg.norm(lateral(lm("LeftUpLeg"), lm("RightUpLeg")))
    shw_c, shw_g = np.linalg.norm(lateral(hc("LeftArm"), hc("RightArm"))), np.linalg.norm(lateral(lm("LeftArm"), lm("RightArm")))
    lat_u = _unit(lateral(hc("LeftArm"), hc("RightArm")))
    r_hip, r_sh = hipw_g / hipw_c, shw_g / shw_c
    rep.append(f"trunk: height x{sv:.3f}, hip width x{r_hip:.3f}, shoulder width x{r_sh:.3f}")

    def trunk_map(frac: float) -> _Affine:
        sl = r_hip + (r_sh - r_hip) * float(np.clip(frac, 0.0, 1.0))
        M = Rt @ (np.eye(3) + (sv - 1.0) * np.outer(u, u) + (sl - 1.0) * np.outer(lat_u, lat_u))
        return _Affine(M, mid_g - M @ mid_g)

    def frac_of(n):
        h = hc(n)
        return 0.0 if h is None else float((h - mid_g) @ u / np.linalg.norm(up_c))

    maps["Hips"] = trunk_map(0.0)
    for n in ("Spine", "Spine1"):
        maps[n] = trunk_map(frac_of(n))
    maps["Spine2"] = trunk_map(1.0)
    for side in ("Left", "Right"):
        maps[f"{side}Shoulder"] = trunk_map(1.0)
    neck = hc("Neck") if hc("Neck") is not None else hc("Head")
    if neck is not None:
        maps["Neck"] = _rigid_at(neck, maps["Spine2"](neck), Rt)
        maps["Head"] = maps["Neck"]
    else:
        maps["Neck"] = maps["Head"] = maps["Spine2"]

    # 3. limbs: each segment maps the character's joint pair onto the robot's
    for side in ("Left", "Right"):
        for a, b in (("UpLeg", "Leg"), ("Leg", "Foot"), ("Arm", "ForeArm"), ("ForeArm", "Hand")):
            maps[f"{side}{a}"] = _segment_map(hc(f"{side}{a}"), hc(f"{side}{b}"), lm(f"{side}{a}"), lm(f"{side}{b}"))
        # hand: rigid at the wrist, turned like the forearm
        maps[f"{side}Hand"] = _rigid_at(hc(f"{side}Hand"), lm(f"{side}Hand"), maps[f"{side}ForeArm"].R)
        # foot: rigid at the ankle, squashed vertically so the sole meets the robot's sole
        ankle_c, ankle_g = hc(f"{side}Foot"), lm(f"{side}Foot")
        foot_ids = [i for i, c in ch.canonical.items() if c in (f"{side}Foot", f"{side}ToeBase")]
        foot_ids += [i for i in range(len(ch.bone_names)) if i not in ch.canonical and ch.canonical_ancestor(i) in (f"{side}Foot", f"{side}ToeBase")]
        dom = ch.joints[np.arange(len(V)), np.argmax(ch.weights, axis=1)]
        fv = V[np.isin(dom, foot_ids)]
        Rs = rotation_between(_unit(hc(f"{side}Foot") - hc(f"{side}Leg")), _unit(lm(f"{side}Foot") - lm(f"{side}Leg")))
        S = np.eye(3)
        if len(fv) and kin.profile.sole:
            body, off = kin.profile.sole[side]
            sole_g = kin.point(body, off)[2]
            hcf = ankle_c[2] - fv[:, 2].min()
            hgf = ankle_g[2] - sole_g
            if hcf > 1e-4 and hgf > 1e-4:
                S = np.diag([1.0, 1.0, hgf / hcf])
                rep.append(f"{side.lower()} foot: ankle height {hcf:.3f} -> {hgf:.3f} m")
        maps[f"{side}Foot"] = _rigid_at(ankle_c, ankle_g, Rs, S)
        maps[f"{side}ToeBase"] = maps[f"{side}Foot"]
        for a, b in (("Arm", "ForeArm"), ("ForeArm", "Hand"), ("UpLeg", "Leg"), ("Leg", "Foot")):
            lc_ = np.linalg.norm(hc(f"{side}{b}") - hc(f"{side}{a}"))
            lg_ = np.linalg.norm(lm(f"{side}{b}") - lm(f"{side}{a}"))
            if side == "Left":
                rep.append(f"{a.lower()}: {lc_:.3f} -> {lg_:.3f} m (x{lg_ / lc_:.2f})")

    # 4. blend the maps with the skin weights
    Vf = np.zeros_like(V)
    nb = len(ch.bone_names)
    bone_map = []
    for i in range(nb):
        c = ch.canonical_ancestor(i)
        bone_map.append(maps.get(c, maps["Hips"]) if c else maps["Hips"])
    for i in range(nb):
        w = (ch.weights * (ch.joints == i)).sum(axis=1)
        sel = w > 0
        if sel.any():
            Vf[sel] += w[sel, None] * bone_map[i](V[sel])
    return Vf, maps, rep


# --------------------------------------------------------------------------- binding


def bind_skin(ch: Character, kin: RobotKinematics, V: np.ndarray, qpos_ref: np.ndarray) -> SkinBinding:
    """Weights per robot body, with bind frames from the reference pose."""
    prof = kin.profile
    kin.forward(qpos_ref)
    n = len(V)
    nb = len(ch.bone_names)
    body_index: dict[str, int] = {}
    cols: list[np.ndarray] = []

    def col(body: str) -> np.ndarray:
        if body not in body_index:
            kin.body(body)  # validates
            body_index[body] = len(cols)
            cols.append(np.zeros(n))
        return cols[body_index[body]]

    unmapped = set()
    for i in range(nb):
        w = (ch.weights * (ch.joints == i)).sum(axis=1)
        if not (w > 0).any():
            continue
        c = ch.canonical_ancestor(i) or "Hips"
        target = prof.body_map.get(c)
        cc = c
        while target is None and cc is not None:  # e.g. Spine1 missing from the map
            cc = B.CANONICAL_PARENT.get(cc)
            target = prof.body_map.get(cc) if cc else None
        if target is None:
            target = prof.body_map["Hips"]
            unmapped.add(ch.bone_names[i])
        if isinstance(target, tuple):
            # split along the bone: proximal body near this joint, distal near the next
            side = next((sd for sd in B.SIDES if c.startswith(sd)), "")
            child = {"ForeArm": "Hand", "Leg": "Foot", "Arm": "ForeArm", "UpLeg": "Leg"}[c[len(side) :]]
            p0 = kin.landmark(c)
            p1 = kin.landmark(f"{side}{child}")
            ax = p1 - p0
            t = np.clip(((V - p0) @ ax) / (ax @ ax), 0.0, 1.0)
            col(target[0])[:] += w * (1.0 - t)
            col(target[1])[:] += w * t
        else:
            col(target)[:] += w
    Wb = np.stack(cols, axis=1)
    Wb = Wb / Wb.sum(axis=1, keepdims=True)
    Wb[Wb < 1e-4] = 0.0
    Wb = Wb / Wb.sum(axis=1, keepdims=True)

    bodies = sorted(body_index, key=lambda b: kin.body(b))
    bindpos, bindquat, vertid, vertweight = [], [], [], []
    for b in bodies:
        k = body_index[b]
        bid = kin.body(b)
        ids = np.nonzero(Wb[:, k] > 0)[0]
        if len(ids) == 0:
            continue
        bindpos.append(kin.data.xpos[bid].copy())
        bindquat.append(kin.data.xquat[bid].copy())
        vertid.append(ids.astype(np.int32))
        vertweight.append(Wb[ids, k].astype(np.float32))
    used = [b for b in bodies if (Wb[:, body_index[b]] > 0).any()]
    report = [f"skin: {n} vertices, {len(ch.faces)} faces, {len(used)} robot bodies"]
    if unmapped:
        report.append(f"bones with no mapped ancestor (bound to pelvis): {sorted(unmapped)}")
    return SkinBinding(
        vertices=V,
        faces=ch.faces,
        texcoords=ch.texcoords,
        bodies=used,
        bindpos=np.array(bindpos),
        bindquat=np.array(bindquat),
        vertid=vertid,
        vertweight=vertweight,
        qpos_ref=qpos_ref,
        report=report,
    )


def skin_vertices(model: mujoco.MjModel, data: mujoco.MjData, binding: SkinBinding) -> np.ndarray:
    """Reference implementation of MuJoCo's skinning (for tests and diagnostics)."""
    out = np.zeros_like(binding.vertices)
    for b, body in enumerate(binding.bodies):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, binding.bindquat[b])
        Rb = Rb.reshape(3, 3)
        Rn = data.xmat[bid].reshape(3, 3)
        R = Rn @ Rb.T
        t = data.xpos[bid] - R @ binding.bindpos[b]
        ids = binding.vertid[b]
        out[ids] += binding.vertweight[b][:, None] * (binding.vertices[ids] @ R.T + t)
    return out
