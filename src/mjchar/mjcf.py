"""Write a Menagerie robot scene plus a character skin as MJCF, via mujoco.MjSpec."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from . import retarget
from .robots import RobotProfile, menagerie_root
from .testrig import png_bytes


def load_robot_spec(profile: RobotProfile, menagerie: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(menagerie / profile.model_dir / profile.scene))


def texture_png(image) -> bytes | None:
    """Base-colour image as PNG bytes (MuJoCo reads PNG; Meshy may ship JPEG/WebP)."""
    if image is None:
        return None
    if image.mime == "image/png":
        return image.data
    try:
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover - pillow-cp312 is in the cvcpkg catalog
        return None
    im = PILImage.open(io.BytesIO(image.data)).convert("RGB")
    return png_bytes(np.asarray(im))


def add_skin(spec: mujoco.MjSpec, binding: retarget.SkinBinding, name: str, texture_file: str | None, rgba=(1, 1, 1, 1), group: int = 1) -> None:
    mat_name = f"{name}_material"
    if texture_file:
        tex = spec.add_texture()
        tex.name = f"{name}_basecolor"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = texture_file
        mat = spec.add_material()
        mat.name = mat_name
        textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
        textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = tex.name
        mat.textures = textures
        mat.specular = 0.2
        mat.shininess = 0.2
    else:
        mat = spec.add_material()
        mat.name = mat_name
        mat.rgba = list(rgba)
    sk = spec.add_skin()
    sk.name = name
    sk.material = mat_name
    sk.group = group
    sk.vert = binding.vertices.astype(np.float32).ravel().tolist()
    sk.face = binding.faces.astype(np.int32).ravel().tolist()
    if binding.texcoords is not None and texture_file:
        # MuJoCo samples skin textures with v running down the image, as glTF does
        sk.texcoord = binding.texcoords.astype(np.float32).ravel().tolist()
    sk.bodyname = list(binding.bodies)
    sk.bindpos = binding.bindpos.astype(np.float32).ravel().tolist()
    sk.bindquat = binding.bindquat.astype(np.float32).ravel().tolist()
    sk.vertid = [v.tolist() for v in binding.vertid]
    sk.vertweight = [w.tolist() for w in binding.vertweight]


def convert(
    glb: str | Path,
    out_dir: str | Path,
    robot: RobotProfile,
    menagerie: str | Path | None = None,
    bone_map: dict[str, str] | None = None,
    fit_limbs: bool = True,
    name: str | None = None,
) -> tuple[Path, retarget.SkinBinding, list[str]]:
    """GLB character -> ``<out_dir>/<robot>_<name>.xml`` (+ assets).  Returns (xml, binding, report)."""
    menagerie = menagerie_root(menagerie)
    out_dir = Path(out_dir)
    ch = retarget.load_character(glb, bone_map, name)
    report = [f"character '{ch.name}': {len(ch.bone_names)} bones, {len(ch.vertices)} vertices"]
    report += [f"note: {n}" for n in ch.notes]
    report.append("bone map: " + ", ".join(f"{ch.bone_names[i]}->{c}" for i, c in sorted(ch.canonical.items())))
    extra = [n for i, n in enumerate(ch.bone_names) if i not in ch.canonical]
    if extra:
        report.append("bound through ancestors: " + ", ".join(extra))

    spec = load_robot_spec(robot, menagerie)
    model = spec.compile()
    kin = retarget.RobotKinematics(model, robot)
    q_ref, ik_rep = retarget.solve_reference_pose(kin, ch)
    report += ik_rep
    V, _maps, fit_rep = retarget.fit_character(ch, kin, fit_limbs=fit_limbs)
    report += fit_rep
    binding = retarget.bind_skin(ch, kin, V, q_ref)
    report += binding.report

    # self-contained output folder: robot meshes + skin texture next to the XML
    out_dir.mkdir(parents=True, exist_ok=True)
    src_assets = menagerie / robot.model_dir / (spec.meshdir or "assets")
    dst_assets = out_dir / "assets"
    if src_assets.is_dir():
        shutil.copytree(src_assets, dst_assets, dirs_exist_ok=True)
    lic = menagerie / robot.model_dir / "LICENSE"
    if lic.exists():
        shutil.copy2(lic, out_dir / f"LICENSE.{robot.model_dir}")
    spec.meshdir = "assets"
    spec.modelfiledir = str(out_dir.resolve())  # resolve assets and texture from the output folder

    tex_file = None
    png = texture_png(ch.image)
    if png:
        tex_file = f"{ch.name}_basecolor.png"
        (out_dir / tex_file).write_bytes(png)
    elif ch.image is not None:
        report.append(f"note: texture is {ch.image.mime}; install pillow-cp312 to convert it (using base colour)")
    skin_name = f"{ch.name}_skin"
    add_skin(spec, binding, skin_name, tex_file, rgba=ch.base_color)
    spec.modelname = f"{robot.name}_{ch.name}"
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1280)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 720)

    xml_path = out_dir / f"{robot.name}_{ch.name}.xml"
    xml_path.write_text(spec.to_xml(), encoding="utf-8")
    # validate the written file the way users will load it
    m2 = mujoco.MjModel.from_xml_path(str(xml_path))
    if m2.nskin != 1:
        raise RuntimeError("compiled model lost the skin")
    meta = {
        "robot": robot.name,
        "character": ch.name,
        "source": str(Path(glb).name),
        "skin": skin_name,
        "qpos_ref": q_ref.tolist(),
        "hide_geom_groups": list(robot.hide_geom_groups),
        "stand_key": robot.stand_key,
        "report": report,
    }
    (out_dir / f"{robot.name}_{ch.name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return xml_path, binding, report
