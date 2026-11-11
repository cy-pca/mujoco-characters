"""Tests: run with  <prefix>/python -m unittest discover -s tests  (from the repo root)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjchar import bones, gltf, retarget  # noqa: E402
from mjchar.mjcf import convert  # noqa: E402
from mjchar.robots import get_profile, menagerie_root  # noqa: E402
from mjchar.testrig import make_test_rig  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="mjchar-test-"))


def rig(template="mixamo", pose="T") -> Path:
    p = TMP / f"{template}_{pose}.glb"
    if not p.exists():
        make_test_rig(p, template, pose)
    return p


class BoneNames(unittest.TestCase):
    def test_conventions(self):
        cases = {
            "mixamorig:Hips": "Hips",
            "mixamorig:LeftUpLeg": "LeftUpLeg",
            "mixamorig1:RightForeArm": "RightForeArm",
            "mixamorig_LeftToeBase": "LeftToeBase",
            "Spine01": "Spine1",  # meshyrig
            "Spine02": "Spine2",
            "neck": "Neck",
            "head_end": None,
            "headfront": None,
            "pelvis": "Hips",  # UE5 mannequin
            "thigh_l": "LeftUpLeg",
            "calf_r": "RightLeg",
            "upperarm_l": "LeftArm",
            "lowerarm_r": "RightForeArm",
            "clavicle_l": "LeftShoulder",
            "ball_r": "RightToeBase",
            "upperarm_twist_01_l": None,
            "LeftUpperLeg": "LeftUpLeg",  # Unity humanoid
            "RightLowerArm": "RightForeArm",
            "UpperChest": "Spine2",
            "LeftHandIndex1": None,
            "Armature|mixamorig:Head": "Head",
        }
        for name, want in cases.items():
            self.assertEqual(bones.canonical_name(name), want, name)

    def test_ue5_spine_shift(self):
        names = ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head"]
        parents = [-1, 0, 1, 2, 3, 4]
        m = bones.map_skeleton(names, parents)
        self.assertEqual(m[0], "Hips")
        self.assertEqual(m[1], "Spine")
        self.assertEqual(m[4], "Neck")

    def test_override(self):
        m = bones.map_skeleton(["root_bone", "weird_thigh"], [-1, 0], {"root_bone": "Hips", "weird_thigh": "LeftUpLeg"})
        self.assertEqual(m, {0: "Hips", 1: "LeftUpLeg"})


class Loading(unittest.TestCase):
    def test_glb_roundtrip(self):
        g = gltf.load(rig())
        self.assertEqual(len(g.skins), 1)
        self.assertEqual(len(g.images), 1)
        p = g.meshes[0][0]
        self.assertEqual(p.joints.shape[1], 4)
        np.testing.assert_allclose(p.weights.sum(axis=1), 1.0, atol=1e-5)

    def test_centimetre_rig_matches_metre_rig(self):
        a = retarget.load_character(rig("mixamo"))
        b = retarget.load_character(rig("meshyrig"))
        np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-5)
        for c in ("Hips", "LeftHand", "RightFoot", "Spine2"):
            np.testing.assert_allclose(a.head(c), b.head(c), atol=1e-5)
        # MuJoCo axes: up is +z, the character's left is +y, it faces +x (nose)
        self.assertGreater(a.head("Head")[2], 1.4)
        self.assertGreater(a.head("LeftHand")[1], 0.5)
        nose = a.vertices[a.vertices[:, 2] > 1.55][:, 0].max()
        self.assertGreater(nose, 0.1)

    def test_turned_character_is_faced_forward(self):
        g = gltf.load(rig())
        # rotate the whole rig 90 degrees about glTF +Y and save it again
        p = TMP / "turned.glb"
        data = rig().read_bytes()
        import json
        import struct

        jlen = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20 : 20 + jlen])
        doc["nodes"][0]["rotation"] = [0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)]
        js = json.dumps(doc).encode()
        js += b" " * (-len(js) % 4)
        rest = data[20 + jlen :]
        p.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + len(rest)) + struct.pack("<II", len(js), 0x4E4F534A) + js + rest)
        a = retarget.load_character(rig())
        b = retarget.load_character(p)
        self.assertTrue(any("rotated to face +X" in n for n in b.notes), b.notes)
        np.testing.assert_allclose(a.head("LeftHand"), b.head("LeftHand"), atol=1e-4)
        del g


class G1Skin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.menagerie = menagerie_root()
        cls.out = TMP / "out_T"
        cls.xml, cls.binding, cls.report = convert(rig(), cls.out, get_profile("g1"))
        cls.model = mujoco.MjModel.from_xml_path(str(cls.xml))

    def _mujoco_skin(self, qpos):
        d = mujoco.MjData(self.model)
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        scn = mujoco.MjvScene(self.model, maxgeom=2000)
        opt = mujoco.MjvOption()
        mujoco.mjv_updateScene(self.model, d, opt, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
        n = self.model.skin_vertnum[0]
        return np.array(scn.skinvert[: 3 * n]).reshape(n, 3), d

    def test_output_is_self_contained(self):
        self.assertTrue((self.out / "assets").is_dir())
        self.assertTrue(any(self.out.glob("*_basecolor.png")))
        self.assertTrue((self.out / "LICENSE.unitree_g1").exists())
        self.assertEqual(self.model.nskin, 1)
        self.assertGreater(self.model.nskintexvert, 0)

    def test_reference_pose_reproduces_bind_mesh(self):
        v, _ = self._mujoco_skin(self.binding.qpos_ref)
        err = np.abs(v - self.binding.vertices).max()
        self.assertLess(err, 1e-4)

    def test_stand_pose_matches_reference_skinning(self):
        k = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        v, d = self._mujoco_skin(self.model.key_qpos[k])
        ref = retarget.skin_vertices(self.model, d, self.binding)
        self.assertLess(np.abs(v - ref).max(), 1e-4)
        # hand vertices follow the wrist; head vertices stay above the torso
        hand = self.binding.bodies.index("left_wrist_yaw_link")
        ids = self.binding.vertid[hand][self.binding.vertweight[hand] > 0.9]
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.assertLess(np.linalg.norm(v[ids] - d.xpos[bid], axis=1).max(), 0.2)
        self.assertGreater(v[:, 2].max(), 1.15)
        self.assertGreater(v[:, 2].min(), -0.03)  # feet not sunk into the floor

    def test_ik_matches_t_pose(self):
        arm = [r for r in self.report if r.startswith("IK left_shoulder")][0]
        self.assertLess(float(arm.split()[-2]), 2.0)


class APose(unittest.TestCase):
    def test_a_pose_meshyrig(self):
        xml, b, report = convert(rig("meshyrig", "A"), TMP / "out_A", get_profile("g1"))
        m = mujoco.MjModel.from_xml_path(str(xml))
        self.assertEqual(m.nskin, 1)
        arm = [r for r in report if r.startswith("IK left_shoulder")][0]
        self.assertLess(float(arm.split()[-2]), 2.0)
        # A-pose: the reference pose lowers the arms, so the shoulder roll is well under 90 degrees
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")
        roll = b.qpos_ref[m.jnt_qposadr[jid]]
        self.assertTrue(0.3 < roll < 1.2, roll)


if __name__ == "__main__":
    unittest.main()
