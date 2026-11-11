"""Minimal glTF 2.0 / GLB reader and writer for skinned characters.

Only what a rigged character needs: node hierarchy, one or more skinned mesh
primitives, skins (joints + inverse bind matrices), base-colour textures and
animation clips.  No Draco or meshopt decompression; files that use them are
rejected with a clear error.

Pure numpy; no third-party glTF library, so it runs from a bare cvcpkg prefix.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67  # b"glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

_COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

UNSUPPORTED_EXTENSIONS = {
    "KHR_draco_mesh_compression": "Draco-compressed meshes",
    "EXT_meshopt_compression": "meshopt-compressed buffers",
}


class GltfError(ValueError):
    pass


# --------------------------------------------------------------------------- math


def quat_to_mat3(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def trs_to_mat4(t=None, r=None, s=None) -> np.ndarray:
    m = np.eye(4)
    rot = quat_to_mat3(r) if r is not None else np.eye(3)
    scl = np.asarray(s, dtype=float) if s is not None else np.ones(3)
    m[:3, :3] = rot * scl[None, :]
    if t is not None:
        m[:3, 3] = t
    return m


# --------------------------------------------------------------------------- model


@dataclass
class Node:
    index: int
    name: str
    children: list[int]
    local: np.ndarray  # 4x4 rest transform
    parent: int = -1
    mesh: int | None = None
    skin: int | None = None


@dataclass
class Primitive:
    positions: np.ndarray  # (n, 3) float
    faces: np.ndarray  # (m, 3) int
    texcoords: np.ndarray | None = None  # (n, 2)
    normals: np.ndarray | None = None
    joints: np.ndarray | None = None  # (n, k) int, indices into skin.joints
    weights: np.ndarray | None = None  # (n, k) float
    material: int | None = None


@dataclass
class Skin:
    name: str
    joints: list[int]  # node indices
    inverse_bind: np.ndarray  # (njoint, 4, 4)
    skeleton: int | None = None


@dataclass
class Image:
    name: str
    mime: str
    data: bytes


@dataclass
class Material:
    name: str
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    base_color_image: int | None = None  # index into Gltf.images


@dataclass
class AnimationChannel:
    node: int
    path: str  # translation | rotation | scale
    times: np.ndarray
    values: np.ndarray
    interpolation: str = "LINEAR"


@dataclass
class Animation:
    name: str
    channels: list[AnimationChannel] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((float(c.times[-1]) for c in self.channels if len(c.times)), default=0.0)


@dataclass
class Gltf:
    nodes: list[Node]
    meshes: list[list[Primitive]]
    skins: list[Skin]
    materials: list[Material]
    images: list[Image]
    animations: list[Animation]
    scene_roots: list[int]
    asset: dict
    path: Path | None = None

    # -- hierarchy helpers -------------------------------------------------
    def global_transforms(self, local_override: dict[int, np.ndarray] | None = None) -> np.ndarray:
        """World 4x4 transforms of every node in the rest pose (or with overrides)."""
        out = np.zeros((len(self.nodes), 4, 4))
        done = np.zeros(len(self.nodes), dtype=bool)

        def visit(i: int, parent: np.ndarray):
            local = self.nodes[i].local
            if local_override and i in local_override:
                local = local_override[i]
            out[i] = parent @ local
            done[i] = True
            for c in self.nodes[i].children:
                visit(c, out[i])

        for i, n in enumerate(self.nodes):
            if n.parent < 0:
                visit(i, np.eye(4))
        if not done.all():  # pragma: no cover - cyclic graphs are invalid glTF
            raise GltfError("node graph is not a forest")
        return out

    def skinned_mesh_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.mesh is not None and n.skin is not None]

    def node_by_name(self, name: str) -> Node | None:
        for n in self.nodes:
            if n.name == name:
                return n
        return None


# --------------------------------------------------------------------------- reader


def _read_accessor(doc: dict, buffers: list[bytes], index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise GltfError(f"accessor {index}: sparse accessors are not supported")
    dtype = np.dtype(_COMPONENT[acc["componentType"]]).newbyteorder("<")
    ncomp = _NCOMP[acc["type"]]
    count = acc["count"]
    if "bufferView" not in acc:
        arr = np.zeros((count, ncomp), dtype=dtype)
    else:
        bv = doc["bufferViews"][acc["bufferView"]]
        buf = buffers[bv["buffer"]]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        elem = dtype.itemsize * ncomp
        stride = bv.get("byteStride", elem)
        if stride == elem:
            arr = np.frombuffer(buf, dtype=dtype, count=count * ncomp, offset=offset).reshape(count, ncomp)
        else:
            raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + elem, offset=offset)
            rows = np.lib.stride_tricks.as_strided(raw, shape=(count, elem), strides=(stride, 1))
            arr = np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)
    if acc.get("normalized") and dtype.kind in "iu":
        info = np.iinfo(dtype)
        arr = arr.astype(np.float64) / info.max
        if dtype.kind == "i":
            arr = np.maximum(arr, -1.0)
    if acc["type"] == "MAT4":
        arr = arr.reshape(count, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
    elif acc["type"] == "SCALAR":
        arr = arr.reshape(count)
    return np.array(arr)


def _load_uri(uri: str, base: Path | None) -> bytes:
    if uri.startswith("data:"):
        _, b64 = uri.split(",", 1)
        return base64.b64decode(b64)
    if base is None:
        raise GltfError(f"external resource {uri!r} needs a file path")
    return (base / uri).read_bytes()


def load(path) -> Gltf:
    """Load a .glb or .gltf file."""
    path = Path(path)
    data = path.read_bytes()
    bin_chunk = None
    if data[:4] == b"glTF":
        magic, version, length = struct.unpack_from("<III", data, 0)
        if version != 2:
            raise GltfError(f"{path}: glTF version {version} is not supported")
        pos = 12
        doc = None
        while pos < length:
            clen, ctype = struct.unpack_from("<II", data, pos)
            chunk = data[pos + 8 : pos + 8 + clen]
            if ctype == CHUNK_JSON:
                doc = json.loads(chunk.decode("utf-8"))
            elif ctype == CHUNK_BIN and bin_chunk is None:
                bin_chunk = bytes(chunk)
            pos += 8 + clen
        if doc is None:
            raise GltfError(f"{path}: no JSON chunk")
    else:
        doc = json.loads(data.decode("utf-8"))
    return parse(doc, bin_chunk, path)


def parse(doc: dict, bin_chunk: bytes | None, path: Path | None = None) -> Gltf:
    used = set(doc.get("extensionsUsed", [])) | set(doc.get("extensionsRequired", []))
    for ext, what in UNSUPPORTED_EXTENSIONS.items():
        if ext in used:
            raise GltfError(f"{ext} ({what}) is not supported; re-export without compression")

    base = path.parent if path else None
    buffers = []
    for i, b in enumerate(doc.get("buffers", [])):
        if "uri" in b:
            buffers.append(_load_uri(b["uri"], base))
        else:
            if bin_chunk is None:
                raise GltfError(f"buffer {i} refers to a missing GLB BIN chunk")
            buffers.append(bin_chunk)

    nodes = []
    for i, n in enumerate(doc.get("nodes", [])):
        if "matrix" in n:
            local = np.array(n["matrix"], dtype=float).reshape(4, 4).T
        else:
            local = trs_to_mat4(n.get("translation"), n.get("rotation"), n.get("scale"))
        nodes.append(
            Node(
                index=i,
                name=n.get("name", f"node{i}"),
                children=list(n.get("children", [])),
                local=local,
                mesh=n.get("mesh"),
                skin=n.get("skin"),
            )
        )
    for n in nodes:
        for c in n.children:
            nodes[c].parent = n.index

    meshes = []
    for m in doc.get("meshes", []):
        prims = []
        for p in m["primitives"]:
            if p.get("mode", 4) != 4:
                raise GltfError("only triangle primitives (mode 4) are supported")
            if any(e in p.get("extensions", {}) for e in UNSUPPORTED_EXTENSIONS):
                raise GltfError("compressed primitives are not supported")
            attr = p["attributes"]
            pos = _read_accessor(doc, buffers, attr["POSITION"]).astype(np.float64)
            if "indices" in p:
                faces = _read_accessor(doc, buffers, p["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                faces = np.arange(len(pos), dtype=np.int64).reshape(-1, 3)
            prim = Primitive(positions=pos, faces=faces, material=p.get("material"))
            if "TEXCOORD_0" in attr:
                prim.texcoords = _read_accessor(doc, buffers, attr["TEXCOORD_0"]).astype(np.float64)
            if "NORMAL" in attr:
                prim.normals = _read_accessor(doc, buffers, attr["NORMAL"]).astype(np.float64)
            jsets, wsets = [], []
            k = 0
            while f"JOINTS_{k}" in attr and f"WEIGHTS_{k}" in attr:
                jsets.append(_read_accessor(doc, buffers, attr[f"JOINTS_{k}"]).astype(np.int64))
                wsets.append(_read_accessor(doc, buffers, attr[f"WEIGHTS_{k}"]).astype(np.float64))
                k += 1
            if jsets:
                prim.joints = np.concatenate(jsets, axis=1)
                prim.weights = np.concatenate(wsets, axis=1)
            prims.append(prim)
        meshes.append(prims)

    skins = []
    for i, s in enumerate(doc.get("skins", [])):
        joints = list(s["joints"])
        if "inverseBindMatrices" in s:
            ibm = _read_accessor(doc, buffers, s["inverseBindMatrices"]).astype(np.float64)
        else:
            ibm = np.repeat(np.eye(4)[None], len(joints), axis=0)
        skins.append(Skin(name=s.get("name", f"skin{i}"), joints=joints, inverse_bind=ibm, skeleton=s.get("skeleton")))

    images = []
    for i, im in enumerate(doc.get("images", [])):
        if "bufferView" in im:
            bv = doc["bufferViews"][im["bufferView"]]
            buf = buffers[bv["buffer"]]
            off = bv.get("byteOffset", 0)
            blob = buf[off : off + bv["byteLength"]]
        else:
            blob = _load_uri(im["uri"], base)
        mime = im.get("mimeType") or _sniff_mime(blob)
        images.append(Image(name=im.get("name", f"image{i}"), mime=mime, data=bytes(blob)))

    textures = doc.get("textures", [])
    materials = []
    for i, m in enumerate(doc.get("materials", [])):
        pbr = m.get("pbrMetallicRoughness", {})
        mat = Material(name=m.get("name", f"material{i}"), base_color=tuple(pbr.get("baseColorFactor", (1, 1, 1, 1))))
        tex = pbr.get("baseColorTexture")
        if tex is not None and tex["index"] < len(textures):
            src = textures[tex["index"]].get("source")
            if src is None:  # KHR_texture_basisu / EXT_texture_webp keep the source in an extension
                for ext in textures[tex["index"]].get("extensions", {}).values():
                    src = ext.get("source", src)
            mat.base_color_image = src
        materials.append(mat)

    animations = []
    for i, a in enumerate(doc.get("animations", [])):
        anim = Animation(name=a.get("name", f"animation{i}"))
        for ch in a.get("channels", []):
            tgt = ch["target"]
            if "node" not in tgt or tgt["path"] not in ("translation", "rotation", "scale"):
                continue
            smp = a["samplers"][ch["sampler"]]
            times = _read_accessor(doc, buffers, smp["input"]).astype(np.float64).reshape(-1)
            vals = _read_accessor(doc, buffers, smp["output"]).astype(np.float64)
            interp = smp.get("interpolation", "LINEAR")
            if interp == "CUBICSPLINE":  # keep the spline vertices, drop tangents
                vals = vals.reshape(len(times), 3, -1)[:, 1, :]
            anim.channels.append(AnimationChannel(tgt["node"], tgt["path"], times, vals.reshape(len(times), -1), interp))
        animations.append(anim)

    scenes = doc.get("scenes", [])
    scene_idx = doc.get("scene", 0)
    roots = list(scenes[scene_idx]["nodes"]) if scenes else [n.index for n in nodes if n.parent < 0]
    return Gltf(nodes, meshes, skins, materials, images, animations, roots, doc.get("asset", {}), path)


def _sniff_mime(blob: bytes) -> str:
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if blob[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


# --------------------------------------------------------------------------- writer


class GlbWriter:
    """Tiny GLB builder used by the synthetic test rig and by tests."""

    def __init__(self, generator: str = "mjchar"):
        self.doc: dict = {"asset": {"version": "2.0", "generator": generator}}
        self.bin = bytearray()

    def _list(self, key: str) -> list:
        return self.doc.setdefault(key, [])

    def _view(self, blob: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(blob)}
        if target is not None:
            view["target"] = target
        self.bin.extend(blob)
        views = self._list("bufferViews")
        views.append(view)
        return len(views) - 1

    def accessor(self, array, kind: str, ctype: int, target: int | None = None, minmax: bool = False, normalized=False) -> int:
        arr = np.ascontiguousarray(array, dtype=np.dtype(_COMPONENT[ctype]).newbyteorder("<"))
        if kind == "MAT4":
            flat = arr.reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, 16)
            count = flat.shape[0]
            blob = np.ascontiguousarray(flat).tobytes()
        else:
            count = arr.shape[0]
            blob = arr.tobytes()
        acc = {"bufferView": self._view(blob, target), "componentType": ctype, "count": int(count), "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            a2 = arr.reshape(count, -1)
            acc["min"] = a2.min(axis=0).tolist()
            acc["max"] = a2.max(axis=0).tolist()
        accs = self._list("accessors")
        accs.append(acc)
        return len(accs) - 1

    def image(self, blob: bytes, mime: str, name: str = "image") -> int:
        ims = self._list("images")
        ims.append({"bufferView": self._view(blob), "mimeType": mime, "name": name})
        return len(ims) - 1

    def add(self, key: str, obj: dict) -> int:
        lst = self._list(key)
        lst.append(obj)
        return len(lst) - 1

    def save(self, path) -> None:
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.doc, separators=(",", ":")).encode("utf-8")
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", GLB_MAGIC, 2, total))
            f.write(struct.pack("<II", len(js), CHUNK_JSON))
            f.write(js)
            f.write(struct.pack("<II", len(self.bin), CHUNK_BIN))
            f.write(bytes(self.bin))
