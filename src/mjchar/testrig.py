"""Synthetic rigged humanoid GLB, for development until a real Meshy export exists.

Builds a 1.7 m character from lofted tubes and ellipsoids (torso, neck, head
with a nose, arms, hands with thumbs, legs, feet), a 512x512 texture (striped
shirt and trousers, a face, shoes), distance-based skin weights (4 influences
per vertex), and a skeleton named in one of two conventions:

* ``mixamo``   - ``mixamorig:Hips`` ... with finger and end bones, metres.
* ``meshyrig`` - the 24-bone set Meshy's API documents as its default
  (``Hips, Spine, Spine01, Spine02, neck, Head, head_end, headfront`` plus
  limbs), in centimetres under an ``Armature`` node scaled by 0.01.

The exact meshyrig bone list is inferred from Meshy's docs (24 bones, those
end-bone names); confirm it against a real export.

Everything is generated here, so the files carry no third-party licence.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .gltf import GlbWriter

# --------------------------------------------------------------------------- skeleton

# name, parent, head (glTF axes: x = character's left, y = up, z = forward), metres
_SKELETON = [
    ("Hips", None, (0.0, 0.95, 0.0)),
    ("Spine", "Hips", (0.0, 1.05, -0.005)),
    ("Spine1", "Spine", (0.0, 1.17, -0.01)),
    ("Spine2", "Spine1", (0.0, 1.29, -0.01)),
    ("Neck", "Spine2", (0.0, 1.46, -0.01)),
    ("Head", "Neck", (0.0, 1.56, 0.0)),
    ("HeadTop_End", "Head", (0.0, 1.75, 0.0)),
]
for _sx, _side in ((1.0, "Left"), (-1.0, "Right")):
    _SKELETON += [
        (f"{_side}Shoulder", "Spine2", (_sx * 0.05, 1.40, -0.01)),
        (f"{_side}Arm", f"{_side}Shoulder", (_sx * 0.18, 1.40, -0.01)),
        (f"{_side}ForeArm", f"{_side}Arm", (_sx * 0.46, 1.40, -0.01)),
        (f"{_side}Hand", f"{_side}ForeArm", (_sx * 0.71, 1.40, -0.01)),
        (f"{_side}HandThumb1", f"{_side}Hand", (_sx * 0.735, 1.39, 0.03)),
        (f"{_side}HandMiddle1", f"{_side}Hand", (_sx * 0.80, 1.40, -0.005)),
        (f"{_side}UpLeg", "Hips", (_sx * 0.09, 0.91, 0.0)),
        (f"{_side}Leg", f"{_side}UpLeg", (_sx * 0.09, 0.50, 0.01)),
        (f"{_side}Foot", f"{_side}Leg", (_sx * 0.09, 0.09, -0.02)),
        (f"{_side}ToeBase", f"{_side}Foot", (_sx * 0.09, 0.02, 0.11)),
        (f"{_side}Toe_End", f"{_side}ToeBase", (_sx * 0.09, 0.02, 0.19)),
    ]

MESHYRIG_NAMES = {
    "Spine1": "Spine01",
    "Spine2": "Spine02",
    "Neck": "neck",
    "HeadTop_End": "head_end",
}
MESHYRIG_DROP = {"LeftHandThumb1", "LeftHandMiddle1", "RightHandThumb1", "RightHandMiddle1", "LeftToe_End", "RightToe_End"}


def _skeleton(pose: str):
    bones = [(n, p, np.array(h, dtype=float)) for n, p, h in _SKELETON]
    if pose.upper() == "A":
        heads = {n: h for n, _, h in bones}
        for side, sx in (("Left", 1.0), ("Right", -1.0)):
            pivot = heads[f"{side}Arm"].copy()
            ang = -sx * np.radians(45.0)  # rotate the arm down about +z (forward)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            for n in list(heads):
                if n.startswith(side) and any(k in n for k in ("ForeArm", "Hand")):
                    heads[n] = pivot + R @ (heads[n] - pivot)
        bones = [(n, p, heads[n]) for n, p, _ in bones]
    return bones


# --------------------------------------------------------------------------- geometry


class _Mesh:
    def __init__(self):
        self.v, self.uv, self.f, self.allowed = [], [], [], []
        self.n = 0

    def add(self, verts, uvs, faces, allowed):
        self.v.append(np.asarray(verts, float))
        self.uv.append(np.asarray(uvs, float))
        self.f.append(np.asarray(faces, np.int64) + self.n)
        self.allowed.append((self.n, self.n + len(verts), allowed))
        self.n += len(verts)


def _frame(t):
    t = t / np.linalg.norm(t)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(t, ref)
    a /= np.linalg.norm(a)
    b = np.cross(t, a)
    return a, b


def _loft(path, radii, band, nring=24, caps=True):
    """Tube along `path` (k,3) with elliptical radii (k,2); UVs inside texture band (v0, v1)."""
    path = np.asarray(path, float)
    radii = np.asarray(radii, float)
    k = len(path)
    verts, uvs = [], []
    v0, v1 = band
    for i in range(k):
        t = path[min(i + 1, k - 1)] - path[max(i - 1, 0)]
        a, b = _frame(t)
        for j in range(nring + 1):
            th = 2 * np.pi * j / nring
            verts.append(path[i] + radii[i, 0] * np.cos(th) * a + radii[i, 1] * np.sin(th) * b)
            uvs.append((j / nring, v0 + (v1 - v0) * (0.02 + 0.96 * i / (k - 1))))
    faces = []
    for i in range(k - 1):
        for j in range(nring):
            p = i * (nring + 1) + j
            q = p + nring + 1
            faces += [(p, p + 1, q), (p + 1, q + 1, q)]
    if caps:
        for end, ring0, flip in ((0, 0, True), (k - 1, (k - 1) * (nring + 1), False)):
            c = len(verts)
            verts.append(path[end])
            uvs.append((0.5, v0 + (v1 - v0) * (0.02 if end == 0 else 0.98)))
            for j in range(nring):
                tri = (c, ring0 + j + 1, ring0 + j)
                faces.append(tri if flip else tri[::-1])
    faces = np.array(faces)
    # orient outward: make sure normals point away from the path
    vv = np.array(verts)
    tri = vv[faces[:20]]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = tri.mean(axis=1) - path[0]
    if (np.einsum("ij,ij->i", nrm, out) < 0).sum() > 10:
        faces = faces[:, ::-1]
    return vv, np.array(uvs), faces


def _ellipsoid(center, radii, band, nlon=24, nlat=16, front_u=True, rot=None):
    verts, uvs = [], []
    v0, v1 = band
    for i in range(nlat + 1):
        phi = np.pi * i / nlat  # 0 top .. pi bottom
        for j in range(nlon + 1):
            lam = 2 * np.pi * j / nlon - np.pi  # -pi..pi, 0 = front (+z)
            d = np.array([np.sin(phi) * np.sin(lam), np.cos(phi), np.sin(phi) * np.cos(lam)])
            p = d * radii
            if rot is not None:
                p = rot @ p
            verts.append(np.asarray(center) + p)
            uvs.append((j / nlon, v0 + (v1 - v0) * (0.02 + 0.96 * i / nlat)))
    faces = []
    for i in range(nlat):
        for j in range(nlon):
            p = i * (nlon + 1) + j
            q = p + nlon + 1
            faces += [(p, q, p + 1), (p + 1, q, q + 1)]
    return np.array(verts), np.array(uvs), np.array(faces)


BANDS = {
    "torso": (0.000, 0.125),
    "arm": (0.125, 0.250),
    "head": (0.250, 0.375),
    "skin": (0.375, 0.500),
    "leg": (0.500, 0.625),
    "shoe": (0.625, 0.750),
}


def _build_mesh(bones):
    H = {n: h for n, _, h in bones}
    m = _Mesh()
    trunk = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "LeftShoulder", "RightShoulder", "LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm"]

    ys = np.array([0.86, 0.95, 1.08, 1.25, 1.36, 1.43, 1.47])
    rx = np.array([0.15, 0.165, 0.14, 0.165, 0.185, 0.13, 0.065])
    rz = np.array([0.10, 0.11, 0.095, 0.11, 0.10, 0.08, 0.06])
    yy = np.linspace(ys[0], ys[-1], 30)
    path = np.c_[np.zeros_like(yy), yy, np.interp(yy, ys, [0, 0, -0.005, -0.01, -0.01, -0.01, -0.01])]
    radii = np.c_[np.interp(yy, ys, rx), np.interp(yy, ys, rz)]
    # loft frame for a vertical path: a = x (lateral), b = z (depth) after _frame
    m.add(*_loft(path, radii, BANDS["torso"], nring=32), trunk)

    neck = np.c_[np.zeros(8), np.linspace(1.42, 1.59, 8), np.full(8, -0.005)]
    m.add(*_loft(neck, np.full((8, 2), 0.05), BANDS["skin"], nring=16), ["Spine2", "Neck", "Head"])
    m.add(*_ellipsoid((0.0, 1.645, 0.01), (0.085, 0.11, 0.10), BANDS["head"]), ["Head"])
    m.add(*_ellipsoid((0.0, 1.625, 0.105), (0.018, 0.025, 0.02), BANDS["skin"], 10, 8), ["Head"])

    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        sh, el, wr = H[f"{side}Arm"], H[f"{side}ForeArm"], H[f"{side}Hand"]
        start = sh + (sh - el) / np.linalg.norm(sh - el) * 0.04
        ts = np.linspace(0, 1, 32)
        pts = np.array([start + (el - start) * (t / 0.5) if t <= 0.5 else el + (wr - el) * ((t - 0.5) / 0.5) for t in ts])
        r = np.interp(ts, [0, 0.5, 1.0], [0.052, 0.042, 0.032])
        m.add(*_loft(pts, np.c_[r, r], BANDS["arm"], nring=20), [f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand", "Spine2"])

        hand_dir = (wr - el) / np.linalg.norm(wr - el)
        R = np.c_[np.cross(hand_dir, [0, 0, 1.0]), hand_dir, [0, 0, 1.0]]  # right-handed, local y along the hand
        R[:, 0] /= np.linalg.norm(R[:, 0])
        hc = wr + hand_dir * 0.075
        m.add(*_ellipsoid(hc, (0.022, 0.08, 0.045), BANDS["skin"], 16, 12, rot=R), [f"{side}Hand", f"{side}HandMiddle1"])
        th = H[f"{side}HandThumb1"]
        m.add(*_ellipsoid(th + hand_dir * 0.02 + np.array([0, -0.005, 0.015]), (0.013, 0.03, 0.013), BANDS["skin"], 10, 8, rot=R), [f"{side}Hand", f"{side}HandThumb1"])

        up, kn, an = H[f"{side}UpLeg"], H[f"{side}Leg"], H[f"{side}Foot"]
        ys2 = np.array([0.98, 0.91, 0.75, 0.50, 0.36, 0.10, 0.07])
        rr = np.array([0.085, 0.085, 0.072, 0.05, 0.056, 0.036, 0.034])
        yy2 = np.linspace(ys2[0], ys2[-1], 36)
        x2 = np.full_like(yy2, sx * 0.09)
        z2 = np.interp(-yy2, -np.array([0.98, 0.91, 0.50, 0.09, 0.07]), [0, 0, 0.01, -0.02, -0.02])
        m.add(*_loft(np.c_[x2, yy2, z2], np.c_[np.interp(-yy2, -ys2, rr), np.interp(-yy2, -ys2, rr)], BANDS["leg"], nring=20), ["Hips", f"{side}UpLeg", f"{side}Leg", f"{side}Foot"])

        fv, fuv, ff = _ellipsoid((sx * 0.09, 0.05, 0.05), (0.048, 0.055, 0.135), BANDS["shoe"], 20, 12)
        fv[:, 1] = np.maximum(fv[:, 1], 0.0)
        m.add(fv, fuv, ff, [f"{side}Foot", f"{side}ToeBase", f"{side}Leg"])
    return m


def _segments(bones):
    H = {n: h for n, _, h in bones}
    kids: dict[str, list[str]] = {}
    for n, p, _ in bones:
        if p:
            kids.setdefault(p, []).append(n)
    seg = {}
    for n, p, h in bones:
        ch = kids.get(n, [])
        primary = {"Hips": "Spine", "Spine2": "Neck", "Hand": None}.get(n, ch[0] if ch else None)
        if n.endswith("Hand"):
            primary = next((c for c in ch if "Middle" in c), None)
        if primary and primary in H:
            seg[n] = (h, H[primary])
        else:
            d = h - H[p] if p else np.array([0, 0.1, 0])
            seg[n] = (h, h + 0.05 * d / max(np.linalg.norm(d), 1e-9))
    return seg


def _weights(m: _Mesh, bones, nmax=4, power=6.0):
    names = [n for n, _, _ in bones]
    seg = _segments(bones)
    V = np.concatenate(m.v)
    J = np.zeros((len(V), nmax), dtype=np.uint16)
    W = np.zeros((len(V), nmax), dtype=np.float32)
    for a, b, allowed in m.allowed:
        allowed = [x for x in allowed if x in seg]
        P = V[a:b]
        D = np.zeros((len(P), len(allowed)))
        for k, n in enumerate(allowed):
            p0, p1 = seg[n]
            d = p1 - p0
            t = np.clip(((P - p0) @ d) / max(d @ d, 1e-12), 0, 1)
            D[:, k] = np.linalg.norm(P - (p0 + t[:, None] * d), axis=1)
        w = 1.0 / np.maximum(D, 1e-3) ** power
        order = np.argsort(-w, axis=1)[:, :nmax]
        top = np.take_along_axis(w, order, axis=1)
        top /= top.sum(axis=1, keepdims=True)
        top[top < 0.01] = 0
        top /= top.sum(axis=1, keepdims=True)
        J[a:b, : order.shape[1]] = np.array([names.index(allowed[i]) for i in range(len(allowed))])[order]
        W[a:b, : order.shape[1]] = top
    return J, W


# --------------------------------------------------------------------------- texture


def _texture(size=512) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size] / size

    def band(name):
        v0, v1 = BANDS[name]
        return (yy >= v0) & (yy < v1), (yy - v0) / (v1 - v0)

    skin = np.array([224, 172, 140])
    shirt = np.array([40, 90, 170])
    shirt2 = np.array([235, 235, 240])
    pants = np.array([120, 105, 70])
    pants2 = np.array([90, 78, 50])
    img[:] = (128, 128, 128)
    sel, t = band("torso")
    stripe = (np.floor(t * 10) % 2 == 0)
    img[sel] = np.where(stripe[sel][:, None], shirt, shirt2)
    sel, t = band("arm")
    sleeve = t < 0.55
    img[sel] = np.where(sleeve[sel][:, None], np.where((np.floor(t * 12) % 2 == 0)[sel][:, None], shirt, shirt2), skin)
    sel, t = band("skin")
    img[sel] = skin
    sel, t = band("head")
    img[sel] = skin
    hair = sel & ((t < 0.32) | ((t < 0.62) & ((xx < 0.25) | (xx > 0.75))))
    img[hair] = (60, 40, 25)
    # face at u = 0.5 (front); eyes, brows and mouth
    for ex in (0.455, 0.545):
        eye = sel & ((xx - ex) ** 2 / 0.012**2 + (t - 0.47) ** 2 / 0.05**2 < 1)
        img[eye] = (250, 250, 250)
        pupil = sel & ((xx - ex) ** 2 / 0.005**2 + (t - 0.48) ** 2 / 0.025**2 < 1)
        img[pupil] = (30, 30, 60)
        brow = sel & (np.abs(xx - ex) < 0.018) & (np.abs(t - 0.39) < 0.012)
        img[brow] = (60, 40, 25)
    mouth = sel & (np.abs(xx - 0.5) < 0.03) & (np.abs(t - 0.66) < 0.012)
    img[mouth] = (170, 60, 60)
    sel, t = band("leg")
    img[sel] = np.where((np.floor(t * 8) % 2 == 0)[sel][:, None], pants, pants2)
    sel, t = band("shoe")
    img[sel] = (45, 35, 30)
    img[sel & (t > 0.85)] = (230, 230, 230)
    return img


def png_bytes(img: np.ndarray) -> bytes:
    """Encode an (h, w, 3|4) uint8 image as PNG with the standard library."""
    h, w, c = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    color = {3: 2, 4: 6}[c]
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


# --------------------------------------------------------------------------- writer


def make_test_rig(path, template: str = "mixamo", pose: str = "T") -> Path:
    path = Path(path)
    bones = _skeleton(pose)
    mesh = _build_mesh(bones)
    J, W = _weights(mesh, bones)
    V = np.concatenate(mesh.v)
    UV = np.concatenate(mesh.uv)
    F = np.concatenate(mesh.f)

    if template == "meshyrig":
        keep = [i for i, (n, _, _) in enumerate(bones) if n not in MESHYRIG_DROP]
        remap = {old: new for new, old in enumerate(keep)}
        parent_of = {n: p for n, p, _ in bones}
        # fold dropped bones' weights into their parents
        names_all = [n for n, _, _ in bones]
        for d in MESHYRIG_DROP:
            di = names_all.index(d)
            J[J == di] = names_all.index(parent_of[d])
        bones = [bones[i] for i in keep]
        J = np.vectorize(remap.get)(J).astype(np.uint16)
        bones.append(("headfront", "Head", np.array([0.0, 1.63, 0.11])))
        names = [MESHYRIG_NAMES.get(n, n) for n, _, _ in bones]
        parents = [MESHYRIG_NAMES.get(p, p) if p else None for _, p, _ in bones]
        unit = 100.0
    elif template == "mixamo":
        names = [f"mixamorig:{n}" for n, _, _ in bones]
        parents = [f"mixamorig:{p}" if p else None for _, p, _ in bones]
        unit = 1.0
    else:
        raise ValueError("template must be 'mixamo' or 'meshyrig'")
    heads = np.array([h for _, _, h in bones]) * unit

    w = GlbWriter(generator="mjchar testrig")
    idx = {n: i for i, n in enumerate(names)}
    first_joint = 2  # node 0 = Armature, node 1 = mesh
    nodes = [
        {"name": "Armature", "children": [1, first_joint]},
        {"name": "Character", "mesh": 0, "skin": 0},
    ]
    if unit != 1.0:
        nodes[0]["scale"] = [1.0 / unit] * 3
    for i, n in enumerate(names):
        pi = idx.get(parents[i]) if parents[i] else None
        t = heads[i] - (heads[pi] if pi is not None else 0.0)
        nodes.append({"name": n, "translation": [float(x) for x in t]})
    for i, n in enumerate(names):
        if parents[i]:
            nodes[first_joint + idx[parents[i]]].setdefault("children", []).append(first_joint + i)
    for nd in nodes:
        w.add("nodes", nd)

    ibm = np.repeat(np.eye(4)[None], len(names), axis=0)
    ibm[:, :3, 3] = -heads
    acc = {
        "POSITION": w.accessor((V * unit).astype(np.float32), "VEC3", 5126, 34962, minmax=True),
        "TEXCOORD_0": w.accessor(UV.astype(np.float32), "VEC2", 5126, 34962),
        "JOINTS_0": w.accessor(J, "VEC4", 5123, 34962),
        "WEIGHTS_0": w.accessor(W, "VEC4", 5126, 34962),
    }
    ind = w.accessor(F.reshape(-1).astype(np.uint32), "SCALAR", 5125, 34963)
    img = w.image(png_bytes(_texture()), "image/png", "basecolor")
    w.add("samplers", {"magFilter": 9729, "minFilter": 9987})
    w.add("textures", {"source": img, "sampler": 0})
    w.add("materials", {"name": "body", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.8}})
    w.add("meshes", {"name": "Character", "primitives": [{"attributes": acc, "indices": ind, "material": 0}]})
    w.add("skins", {"name": "skin", "joints": list(range(first_joint, first_joint + len(names))), "inverseBindMatrices": w.accessor(ibm, "MAT4", 5126), "skeleton": first_joint})
    w.add("scenes", {"nodes": [0]})
    w.doc["scene"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    w.save(path)
    return path
