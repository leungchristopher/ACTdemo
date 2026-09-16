import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TABLE_Z = 0.0
SHELF_Z = 0.105
FLUTE_HEIGHT = 0.185
GRASP_Z = 0.055
SOURCE = np.array([0.0, -0.255])
TARGET = np.array([0.29, -0.225])


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def build_scene(path: Path | None = None) -> str:
    root = ET.parse(ROOT / "assets/so100/so_arm100.xml").getroot()
    root.set("model", "SO-100 champagne flute: invert from shelf to table")
    root.find("compiler").set("meshdir", str(ROOT / "assets/so100/assets"))
    root.remove(root.find("keyframe"))
    option = root.find("option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    option.set("iterations", "100")
    option.set("gravity", "0 0 -9.81")
    element(root, "size", njmax="4000", nconmax="1000")
    visual = element(root, "visual")
    element(visual, "global", offwidth=960, offheight=720, azimuth=135, elevation=-25)
    element(visual, "quality", shadowsize=2048)
    element(visual, "headlight", diffuse=".5 .5 .5", ambient=".3 .3 .3")
    assets = root.find("asset")
    element(
        assets,
        "texture",
        name="floor_tex",
        type="2d",
        builtin="checker",
        rgb1=".15 .19 .23",
        rgb2=".19 .23 .27",
        width=512,
        height=512,
    )
    element(
        assets, "material", name="floor_mat", texture="floor_tex", texrepeat="5 5", reflectance=".1"
    )
    element(assets, "material", name="oak", rgba=".53 .32 .16 1")
    element(assets, "material", name="glass", rgba=".57 .85 .94 .5", specular=".9", shininess=".9")
    element(assets, "material", name="rim", rgba=".69 .92 .97 .8", specular=".9", shininess=".9")
    world = root.find("worldbody")
    element(world, "light", pos="0 -.3 1.5", dir="0 0 -1", directional="true", diffuse=".85 .85 .8")
    element(world, "light", pos="-.8 .2 .8", dir="1 -1 -1", diffuse=".4 .5 .6")
    element(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 .1",
        pos="0 0 -.085",
        material="floor_mat",
    )
    element(
        world,
        "geom",
        name="table",
        type="box",
        size=".48 .4 .025",
        pos="0 -.17 -.025",
        material="oak",
        friction=".8 .01 .001",
    )
    # Narrow elevated shelf, open on all sides within the arm's reach.
    element(
        world,
        "geom",
        name="shelf",
        type="box",
        size=".09 .065 .009",
        pos=f"0 -.29 {SHELF_Z - 0.009}",
        rgba=".28 .34 .40 1",
        friction=".8 .01 .001",
    )
    for x in [-0.073, 0.073]:
        element(
            world,
            "geom",
            name=f"shelf_leg_{x}",
            type="box",
            size=".009 .009 .048",
            pos=f"{x} -.335 .048",
            rgba=".18 .22 .28 1",
        )
    element(
        world,
        "site",
        name="target",
        type="cylinder",
        size=".043 .0004",
        pos=f"{TARGET[0]} {TARGET[1]} .0006",
        rgba=".2 .8 .55 .4",
    )
    element(
        world,
        "camera",
        name="overview",
        pos=".67 -.85 .60",
        xyaxes=".79 .61 0 -.31 .40 .86",
        fovy=43,
    )
    element(
        world, "camera", name="side", pos="-.65 -.50 .35", xyaxes=".45 -.89 0 .30 .15 .94", fovy=46
    )
    for body in world.iter("body"):
        body.set("gravcomp", "1")
    fixed = world.find(".//body[@name='Fixed_Jaw']")
    element(fixed, "site", name="grasp", pos=".003 -.1014 0", size=".004", rgba="1 .4 .1 0")
    element(
        fixed, "camera", name="wrist", pos="0 -.045 -.035", xyaxes="1 0 0 0 -.45 -.893", fovy=75
    )
    # Slightly compliant rubber pads, same geometry as the supplied model.
    for pad in world.iter("geom"):
        if "pad_" in pad.get("name", ""):
            pad.set("friction", "2 .02 .002")
            pad.set("condim", "6")
            pad.set("solref", ".004 1")
            pad.set("solimp", ".95 .99 .001")
    flute = element(
        world,
        "body",
        name="flute",
        pos=f"{SOURCE[0]} {SOURCE[1]} {SHELF_Z + FLUTE_HEIGHT}",
        quat="0 1 0 0",
    )
    element(flute, "freejoint", name="flute_free")
    element(flute, "inertial", pos="0 0 .098", mass=".065", diaginertia=".00019 .00019 .000018")
    common = dict(material="glass", friction="1 .01 .001", solref=".004 1", solimp=".95 .99 .001")
    element(
        flute,
        "geom",
        name="flute_base",
        type="cylinder",
        size=".030 .003",
        pos="0 0 .003",
        **common,
    )
    element(
        flute,
        "geom",
        name="flute_stem",
        type="cylinder",
        size=".004 .041",
        pos="0 0 .047",
        **common,
    )
    # Hollow bowl: individual wall segments avoid the solid convex hull of a mesh.
    profile = [(0.087, 0.004), (0.105, 0.018), (0.14, 0.025), (0.184, 0.023)]
    n = 20
    for level, ((z0, r0), (z1, r1)) in enumerate(zip(profile, profile[1:])):
        for i in range(n):
            a = i * 2 * np.pi / n
            p = [r0 * np.cos(a), r0 * np.sin(a), z0, r1 * np.cos(a), r1 * np.sin(a), z1]
            element(
                flute,
                "geom",
                name=f"flute_wall_{level}_{i}",
                type="capsule",
                fromto=" ".join(map(str, p)),
                size=".001",
                rgba="0 0 0 0",
                **common,
            )
    for i in range(n):
        a, b = i * 2 * np.pi / n, (i + 1) * 2 * np.pi / n
        p = [
            0.023 * np.cos(a),
            0.023 * np.sin(a),
            0.184,
            0.023 * np.cos(b),
            0.023 * np.sin(b),
            0.184,
        ]
        element(
            flute,
            "geom",
            name=f"flute_rim_{i}",
            type="capsule",
            fromto=" ".join(map(str, p)),
            size=".001",
            **{**common, "material": "rim"},
        )
    element(flute, "site", name="flute_grasp", pos=f"0 0 {GRASP_Z}", size=".003", rgba="0 1 0 0")
    # A continuous transparent visual shell (collision remains the hollow segments).
    vertices, faces = [], []
    for z, r in profile:
        for i in range(n):
            a = i * 2 * np.pi / n
            vertices.extend([r * np.cos(a), r * np.sin(a), z])
    for j in range(len(profile) - 1):
        for i in range(n):
            a = j * n + i
            b = j * n + (i + 1) % n
            c = (j + 1) * n + i
            d = (j + 1) * n + (i + 1) % n
            faces.extend([a, b, c, b, d, c])
    element(
        assets,
        "mesh",
        name="bowl_visual",
        vertex=" ".join(map(str, vertices)),
        face=" ".join(map(str, faces)),
    )
    element(
        flute,
        "geom",
        name="flute_shell",
        type="mesh",
        mesh="bowl_visual",
        material="glass",
        contype=0,
        conaffinity=0,
        density=0,
        group=2,
    )
    xml = ET.tostring(root, encoding="unicode")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        root.find("compiler").set(
            "meshdir", os.path.relpath(ROOT / "assets/so100/assets", path.resolve().parent)
        )
        ET.indent(root)
        path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return xml
