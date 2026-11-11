"""Command line: ``mjchar <command> ...`` (or ``python -m mjchar <command> ...``)

Commands
  skin      GLB character -> MJCF robot scene with the character as a skin
  inspect   print a GLB's skeleton and how its bones map to canonical names
  demo      run the G1 routine on a converted scene: viewer, render or headless
  testrig   write the synthetic Mixamo/meshyrig-named test character
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_skin(a) -> int:
    from .mjcf import convert
    from .robots import get_profile

    bone_map = json.loads(Path(a.bone_map).read_text(encoding="utf-8")) if a.bone_map else None
    xml, _b, report = convert(a.glb, a.out, get_profile(a.robot), a.menagerie, bone_map, fit_limbs=not a.no_fit, name=a.name)
    for line in report:
        print(line)
    print(f"wrote {xml}")
    return 0


def _cmd_inspect(a) -> int:
    from . import bones, gltf

    g = gltf.load(a.glb)
    print(f"{a.glb}: generator={g.asset.get('generator', '?')}")
    nodes = g.skinned_mesh_nodes()
    if not nodes:
        print("no skinned mesh")
        return 1
    skin = g.skins[nodes[0].skin]
    names = [g.nodes[j].name for j in skin.joints]
    pos = {j: k for k, j in enumerate(skin.joints)}
    parents = []
    for j in skin.joints:
        p = g.nodes[j].parent
        while p >= 0 and p not in pos:
            p = g.nodes[p].parent
        parents.append(pos.get(p, -1))
    canon = bones.map_skeleton(names, parents)
    G = g.global_transforms()
    kids: dict[int, list[int]] = {}
    for k, p in enumerate(parents):
        kids.setdefault(p, []).append(k)

    def show(k: int, depth: int) -> None:
        head = G[skin.joints[k]][:3, 3]
        label = f"{'  ' * depth}{names[k]}"
        print(f"{label:<40s} -> {canon.get(k, '-'):<14s} head=({head[0]:.3f}, {head[1]:.3f}, {head[2]:.3f}) (glTF world)")
        for c in kids.get(k, []):
            show(c, depth + 1)

    for root in kids.get(-1, []):
        show(root, 0)
    missing = [c for c in bones.CORE if c not in canon.values()]
    if missing:
        print("missing canonical bones:", ", ".join(missing))
    for m in g.meshes:
        for p in m:
            print(f"primitive: {len(p.positions)} verts, {len(p.faces)} faces, influences/vertex={0 if p.joints is None else p.joints.shape[1]}")
    for im in g.images:
        print(f"image {im.name}: {im.mime}, {len(im.data)} bytes")
    for an in g.animations:
        print(f"animation '{an.name}': {len(an.channels)} channels, {an.duration:.2f} s")
    return 0


def add_demo_arguments(ap: argparse.ArgumentParser) -> None:
    """``mjchar demo`` options (here, so parsing the command line does not import MuJoCo)."""
    ap.add_argument("xml", type=Path, help="scene written by `mjchar skin`")
    ap.add_argument("--viewer", action="store_true", help="open the interactive viewer")
    ap.add_argument("--show-robot", action="store_true", help="viewer: draw the robot under the skin")
    ap.add_argument("--seconds", type=float, help="viewer: close after this many seconds")
    ap.add_argument("--render", type=Path, help="write an MP4 (needs ffmpeg), a GIF, a contact sheet and a still to this folder")


def _cmd_demo(a) -> int:
    from . import demo

    return demo.run(a)


def _cmd_testrig(a) -> int:
    from .testrig import make_test_rig

    p = make_test_rig(a.out, a.template, a.pose)
    print(f"wrote {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mjchar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("skin", help="GLB character -> robot scene with the character as a skin")
    s.add_argument("glb")
    s.add_argument("-o", "--out", required=True, help="output folder (gets the XML, robot meshes, texture)")
    s.add_argument("--robot", default="g1", help="robot profile (default: g1)")
    s.add_argument("--menagerie", help="share/mujoco-menagerie folder (default: from the cvcpkg prefix)")
    s.add_argument("--bone-map", help="JSON {source bone name: canonical name} overrides")
    s.add_argument("--name", help="character name (default: GLB file stem)")
    s.add_argument("--no-fit", action="store_true", help="scale only; do not re-proportion limbs to the robot")
    s.set_defaults(func=_cmd_skin)

    i = sub.add_parser("inspect", help="print a GLB's skeleton and bone mapping")
    i.add_argument("glb")
    i.set_defaults(func=_cmd_inspect)

    d = sub.add_parser("demo", help="G1 stand + scripted routine on a converted scene (viewer, render or headless)")
    add_demo_arguments(d)
    d.set_defaults(func=_cmd_demo)

    t = sub.add_parser("testrig", help="write the synthetic test character")
    t.add_argument("out")
    t.add_argument("--template", choices=["mixamo", "meshyrig"], default="mixamo")
    t.add_argument("--pose", choices=["T", "A"], default="T")
    t.set_defaults(func=_cmd_testrig)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
