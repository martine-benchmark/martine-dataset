# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Evaluation camera set of a view subset (martine-convert subsets ... --write-eval-sfm DIR): the selected views of
gt/eval_cameras.sfm (MVS) or of the MVPS sfm.json (one view per pose) and their silhouettes, in the layout read by
the sparse-view evaluation (DIR/<preset>.sfm or DIR/views<N>.sfm, DIR/masks/<view stem>.png). No watertight mask is written."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import DATA
from martine_converters import cli as convert
from martine_converters import subsets
from martine_converters.subsets import load_table, preset_entry
from test_subsets import no_deps, ring, write_sfm

OBJ = "rock"  # an object with presets and precomputed subsets
N = 84


def _transform(c, k):
    a = 0.01 * k
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return {"rotation": [repr(float(x)) for x in rot.ravel()], "center": [repr(float(x)) for x in c]}


def _mask(path: Path, value: int, shape=(12, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros(shape, np.uint8)
    m[2:-2, 2:-2] = value
    assert cv2.imwrite(str(path), m)


def write_eval_cameras(root: Path, obj: str = OBJ, watertight: bool = False) -> Path:
    """Published-like gt/: eval_cameras.sfm (views masks_eval/NNN.png, hash-like ids), masks_eval/,
    masks_watertight/."""
    gt = root / obj / "gt"
    intr = {"intrinsicId": "11", "width": "960", "height": "640", "sensorWidth": "36", "type": "pinhole",
            "focalLength": "42", "pixelRatio": "1", "principalPoint": ["-1.5", "2.5"], "distortionType": "none"}
    unused = dict(intr, intrinsicId="12", focalLength="50")
    poses, views = [], []
    for k, c in enumerate(ring(N)):
        pid = str(900000 - 13 * k)
        poses.append({"poseId": pid, "pose": {"transform": _transform(c, k), "locked": "false"}})
        views.append({"viewId": str(70000 + k), "poseId": pid, "intrinsicId": "11", "path": f"masks_eval/{k:03d}.png",
                      "width": "960", "height": "640", "metadata": {"Exif:FocalLength": "42"}})
        _mask(gt / "masks_eval" / f"{k:03d}.png", 1 + k)
        if watertight:
            _mask(gt / "masks_watertight" / f"{k:03d}.png", 255 - k)
    poses.reverse()  # pose list not in view order
    sfm = {"version": ["1", "2", "12"], "featuresFolders": ["../../FeatureExtraction/x"],
           "matchesFolders": ["../../FeatureMatching/y"], "views": views, "intrinsics": [unused, intr],
           "poses": poses, "surveys": ""}
    (gt / "eval_cameras.sfm").write_text(json.dumps(sfm))
    return gt / "eval_cameras.sfm"


def write_mvps(root: Path, obj: str = OBJ, tier: str = "d2", lights: int = 4, duplicate: bool = True) -> Path:
    """Published-like MVPS sfm.json with one pose entry per view (the duplicated poseIds of 16 published objects),
    an unused intrinsic, LED images not in order, and masks/pose_KK.png."""
    path = root / obj / "mvps" / tier / "sfm.json"
    write_sfm(path, ring(N, phase=0.01), "mvps", scale=2, lights=lights)
    d = json.loads(path.read_text())
    d["intrinsics"].append(dict(d["intrinsics"][0], intrinsicId="2", width="9568", height="6376"))
    for v in d["views"]:
        v.update(width=d["intrinsics"][0]["width"], height=d["intrinsics"][0]["height"])
    d["views"].sort(key=lambda v: (v["path"].split("/")[1], not v["path"].endswith("/lighting_02.exr"), v["path"]))
    by_id = {p["poseId"]: p for p in d["poses"]}
    if duplicate:
        d["poses"] = [json.loads(json.dumps(by_id[v["poseId"]])) for v in d["views"]]
    path.write_text(json.dumps(d))
    for k in range(N):
        _mask(path.parent / "masks" / f"pose_{k:02d}.png", 3 + k)
    return path


def eval_cameras(sfm_path: Path):
    """Camera order and mask names as the evaluation loads an SfMData: views with a pose and an intrinsic, the first
    view of each poseId, mask = basename stem of the view path + .png."""
    d = json.loads(Path(sfm_path).read_text())
    poses = {p["poseId"]: p["pose"]["transform"] for p in d["poses"]}
    intr = {i["intrinsicId"] for i in d["intrinsics"]}
    seen, out = set(), []
    for v in d["views"]:
        if v["poseId"] in poses and v["intrinsicId"] in intr and v["poseId"] not in seen:
            seen.add(v["poseId"])
            out.append((os.path.splitext(os.path.basename(v["path"]))[0] + ".png", poses[v["poseId"]], v))
    return out


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def written_sfm(out: Path) -> Path:
    """The camera file of a --write-eval-sfm folder: the one named in its manifest."""
    (sfm,) = Path(out).glob("*.sfm")
    assert json.loads((Path(out) / "eval_sfm.json").read_text())["sfm"] == sfm.name
    return sfm


def run(root, out, *extra):
    return convert.main(["subsets", "--object", OBJ, "--data", str(root), *extra, "--write-eval-sfm", str(out)])


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(subsets, "_deps", no_deps)
    root = tmp_path / "data"
    write_eval_cameras(root)
    write_mvps(root)
    return root


@pytest.mark.parametrize("preset", ["front3", "360_6"])
def test_mvs_eval_sfm_keeps_the_preset_views_in_source_order(data, tmp_path, preset):
    out = tmp_path / "ev"
    assert run(data, out, "--preset", preset) == 0
    src = json.loads((data / OBJ / "gt" / "eval_cameras.sfm").read_text())
    sfm = json.loads((written_sfm(out)).read_text())
    names = sorted(preset_entry(OBJ, preset)["mvs"])  # source order of the published views is name order
    assert [Path(v["path"]).stem for v in sfm["views"]] == names
    assert sfm["version"] == src["version"] and [i["intrinsicId"] for i in sfm["intrinsics"]] == ["11"]
    assert "featuresFolders" not in sfm and "matchesFolders" not in sfm
    src_views = {Path(v["path"]).stem: v for v in src["views"]}
    src_poses = {p["poseId"]: p for p in src["poses"]}
    for v in sfm["views"]:
        s = src_views[Path(v["path"]).stem]
        assert {k: x for k, x in v.items() if k != "path"} == {k: x for k, x in s.items() if k != "path"}
        assert v["path"] == f"masks/{Path(s['path']).stem}.png"
    assert sfm["poses"] == [src_poses[v["poseId"]] for v in sfm["views"]]
    cams = eval_cameras(written_sfm(out))
    assert [c[0] for c in cams] == [f"{n}.png" for n in names]
    for mask, _, _ in cams:
        assert md5(out / "masks" / mask) == md5(data / OBJ / "gt" / "masks_eval" / mask)
    assert sorted(os.listdir(out / "masks")) == [c[0] for c in cams]
    assert not (out / "masks_watertight").exists()
    m = json.loads((out / "eval_sfm.json").read_text())
    assert m["object"] == OBJ and m["preset"] == preset and m["modality"] == "mvs" and m["views"] == names
    assert m["source"] == str(data / OBJ / "gt" / "eval_cameras.sfm")
    assert m["source_md5"] == md5(data / OBJ / "gt" / "eval_cameras.sfm")
    assert m["masks_source"] == str(data / OBJ / "gt" / "masks_eval") and "watertight" not in m
    assert sorted(os.listdir(out)) == sorted([f"{preset}.sfm", "eval_sfm.json", "masks"])
    assert m["sfm"] == f"{preset}.sfm"


def test_mvs_watertight_masks_are_not_written(tmp_path, monkeypatch):
    """The watertight masks belong to the usual cleanup: none in DIR, even for an object that has them."""
    monkeypatch.setattr(subsets, "_deps", no_deps)
    root = tmp_path / "data"
    write_eval_cameras(root, watertight=True)
    out = tmp_path / "ev"
    assert run(root, out, "--preset", "360_6") == 0
    assert sorted(os.listdir(out)) == ["360_6.sfm", "eval_sfm.json", "masks"]
    assert len(eval_cameras(written_sfm(out))) == 6
    assert not any("watertight" in k for k in json.loads((out / "eval_sfm.json").read_text()))


def test_watertight_masks_option_is_gone(data, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(data, tmp_path / "ev", "--preset", "front3", "--watertight-masks", str(tmp_path))
    assert "unrecognized arguments" in capsys.readouterr().err


def test_mvs_n_views(data, tmp_path):
    out = tmp_path / "ev"
    assert run(data, out, "--n-views", "5") == 0
    names = sorted(load_table()["objects"][OBJ]["5"]["mvs"])
    assert [c[0] for c in eval_cameras(written_sfm(out))] == [f"{n}.png" for n in names]
    m = json.loads((out / "eval_sfm.json").read_text())
    assert m["n_views"] == 5 and "preset" not in m and m["views"] == names
    assert m["sfm"] == "views5.sfm" and (out / "views5.sfm").is_file()   # the evaluation tag: @views5


def test_mvps_eval_sfm_one_view_per_pose_despite_duplicated_pose_ids(data, tmp_path):
    out = tmp_path / "ev"
    assert run(data, out, "--preset", "360_6", "--modality", "mvps") == 0
    e = preset_entry(OBJ, "360_6")
    src_path = data / OBJ / "mvps" / "d2" / "sfm.json"
    src = json.loads(src_path.read_text())
    sfm = json.loads((written_sfm(out)).read_text())
    folder_order = list(dict.fromkeys(v["path"].split("/")[1] for v in src["views"]))
    names = [p for p in folder_order if p in set(e["mvps"])]
    assert [Path(v["path"]).stem for v in sfm["views"]] == names and len(names) == 6
    src_views = {v["viewId"]: v for v in src["views"]}
    for v in sfm["views"]:
        s = src_views[v["viewId"]]
        assert s["path"] == f"images/{Path(v['path']).stem}/lighting_00.exr"  # lighting_00 preferred
        assert v["path"] == f"masks/{Path(v['path']).stem}.png"
    assert len(sfm["poses"]) == 6 == len({p["poseId"] for p in sfm["poses"]})
    first = {}
    for p in src["poses"]:
        first.setdefault(p["poseId"], p)
    assert sfm["poses"] == [first[v["poseId"]] for v in sfm["views"]]
    assert [i["intrinsicId"] for i in sfm["intrinsics"]] == ["1"]  # the tier intrinsic only
    cams = eval_cameras(written_sfm(out))
    assert [c[0] for c in cams] == [f"{n}.png" for n in names]
    for mask, _, _ in cams:
        assert md5(out / "masks" / mask) == md5(src_path.parent / "masks" / mask)
    assert not (out / "masks_watertight").exists()
    m = json.loads((out / "eval_sfm.json").read_text())
    assert m["modality"] == "mvps" and m["tier"] == "d2" and m["views"] == names
    assert m["source"] == str(src_path) and m["source_md5"] == md5(src_path)
    assert m["masks_source"] == str(src_path.parent / "masks")
    assert "watertight" not in m
    pair = dict(zip(e["mvps"], zip(e["mvs"], e["distance_mm"])))
    assert m["paired_mvs_views"] == [pair[n][0] for n in names]
    assert m["distance_mm"] == [pair[n][1] for n in names]


def test_mvps_tier_option(data, tmp_path):
    write_mvps(data, tier="d8", duplicate=False)
    out = tmp_path / "ev"
    assert run(data, out, "--preset", "front3", "--modality", "mvps", "--tier", "d8") == 0
    m = json.loads((out / "eval_sfm.json").read_text())
    assert m["tier"] == "d8" and m["source"] == str(data / OBJ / "mvps" / "d8" / "sfm.json")


def test_mvps_pose_without_lighting_00_takes_its_first_image(data, tmp_path):
    src_path = data / OBJ / "mvps" / "d2" / "sfm.json"
    d = json.loads(src_path.read_text())
    d["views"] = [v for v in d["views"] if v["path"] != "images/pose_09/lighting_00.exr"]
    src_path.write_text(json.dumps(d))
    out = tmp_path / "ev"
    assert run(data, out, "--preset", "front3", "--modality", "mvps") == 0
    sfm = json.loads((written_sfm(out)).read_text())
    vid = next(v["viewId"] for v in sfm["views"] if v["path"] == "masks/pose_09.png")
    assert {v["viewId"]: v["path"] for v in d["views"]}[vid] == "images/pose_09/lighting_01.exr"


def test_mvps_pose_with_two_transforms_is_an_error(data, tmp_path, capsys):
    src_path = data / OBJ / "mvps" / "d2" / "sfm.json"
    d = json.loads(src_path.read_text())
    pid = next(v["poseId"] for v in d["views"] if v["path"].startswith("images/pose_17/"))
    dup = [p for p in d["poses"] if p["poseId"] == pid]
    assert len(dup) > 1
    dup[-1]["pose"]["transform"]["center"][0] = repr(float(dup[-1]["pose"]["transform"]["center"][0]) + 1.0)
    src_path.write_text(json.dumps(d))
    out = tmp_path / "ev"
    assert run(data, out, "--preset", "360_6", "--modality", "mvps") == 2
    err = capsys.readouterr().err
    assert "pose_17" in err and "transform" in err
    assert not out.exists() or not any(out.iterdir())


def test_mvps_pose_folder_with_two_pose_ids_and_two_transforms_is_an_error(data, tmp_path, capsys):
    src_path = data / OBJ / "mvps" / "d2" / "sfm.json"
    d = json.loads(src_path.read_text())
    v = next(v for v in d["views"] if v["path"] == "images/pose_26/lighting_01.exr")
    other = next(x for x in d["views"] if x["path"].startswith("images/pose_27/"))
    v["poseId"] = other["poseId"]  # one image of pose_26 points at the camera of pose_27
    src_path.write_text(json.dumps(d))
    assert run(data, tmp_path / "ev", "--preset", "front3", "--modality", "mvps") == 2
    assert "pose_26" in capsys.readouterr().err


def test_mvps_two_selected_poses_sharing_a_pose_id_is_an_error(data, tmp_path, capsys):
    src_path = data / OBJ / "mvps" / "d2" / "sfm.json"
    d = json.loads(src_path.read_text())
    pid = next(v["poseId"] for v in d["views"] if v["path"].startswith("images/pose_09/"))
    for v in d["views"]:
        if v["path"].startswith("images/pose_17/"):
            v["poseId"] = pid
    d["poses"] = [p for p in d["poses"] if p["poseId"] != pid] + [p for p in d["poses"] if p["poseId"] == pid][:1]
    src_path.write_text(json.dumps(d))
    assert run(data, tmp_path / "ev", "--preset", "front3", "--modality", "mvps") == 2
    err = capsys.readouterr().err
    assert "pose_09" in err and "pose_17" in err


def test_missing_mask_is_an_error(data, tmp_path, capsys):
    (data / OBJ / "gt" / "masks_eval" / "017.png").unlink()
    assert run(data, tmp_path / "ev", "--preset", "360_6") == 2
    assert "017.png" in capsys.readouterr().err


def test_non_empty_folder_needs_overwrite(data, tmp_path, capsys):
    out = tmp_path / "ev"
    out.mkdir()
    (out / "notes.txt").write_text("x")
    assert run(data, out, "--preset", "front3") == 2
    assert "--overwrite" in capsys.readouterr().err
    assert run(data, out, "--preset", "front3", "--overwrite") == 2  # not a previous eval-sfm output
    assert (out / "notes.txt").is_file()
    (out / "notes.txt").unlink()
    assert run(data, out, "--preset", "360_6") == 0
    assert run(data, out, "--preset", "front3") == 2
    assert run(data, out, "--preset", "front3", "--overwrite") == 0
    assert len(os.listdir(out / "masks")) == 3
    assert json.loads((out / "eval_sfm.json").read_text())["preset"] == "front3"


def test_never_writes_under_the_data_root(data, capsys):
    for out in (data / "ev", data / OBJ / "gt" / "ev", data):
        assert run(data, out, "--preset", "front3") == 2
        assert "data root" in capsys.readouterr().err
        assert not list(out.glob("*.sfm"))
    assert not (data / "ev").exists()


def test_write_eval_sfm_needs_data(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(subsets, "_deps", no_deps)
    assert convert.main(["subsets", "--object", OBJ, "--preset", "front3", "--write-eval-sfm",
                         str(tmp_path / "ev")]) == 2
    assert "--data" in capsys.readouterr().err


@pytest.mark.data
@pytest.mark.parametrize("preset", ["front3", "360_6"])
def test_published_vase_mvs(tmp_path, preset):
    out = tmp_path / "ev"
    assert convert.main(["subsets", "--object", "vase", "--preset", preset, "--data", str(DATA),
                         "--write-eval-sfm", str(out)]) == 0
    src = DATA / "vase" / "gt" / "eval_cameras.sfm"
    ref = {c[0]: c[1] for c in eval_cameras(src)}
    cams = eval_cameras(written_sfm(out))
    names = sorted(preset_entry("vase", preset)["mvs"])
    assert [c[0] for c in cams] == [f"{n}.png" for n in names]
    for mask, tr, _ in cams:
        assert tr == ref[mask]
        assert md5(out / "masks" / mask) == md5(DATA / "vase" / "gt" / "masks_eval" / mask)
    assert json.loads((out / "eval_sfm.json").read_text())["source_md5"] == md5(src)
    assert not (out / "masks_watertight").exists()


@pytest.mark.data
def test_published_vase_mvps(tmp_path):
    out = tmp_path / "ev"
    assert convert.main(["subsets", "--object", "vase", "--preset", "360_6", "--modality", "mvps", "--data",
                         str(DATA), "--write-eval-sfm", str(out)]) == 0
    src = DATA / "vase" / "mvps" / "d2" / "sfm.json"
    d = json.loads(src.read_text())
    poses = {p["poseId"]: p["pose"]["transform"] for p in d["poses"]}
    ref = {v["path"].split("/")[1]: poses[v["poseId"]] for v in d["views"]}
    cams = eval_cameras(written_sfm(out))
    names = [n for n in dict.fromkeys(v["path"].split("/")[1] for v in d["views"])
             if n in set(preset_entry("vase", "360_6")["mvps"])]
    assert [c[0] for c in cams] == [f"{n}.png" for n in names]
    for mask, tr, v in cams:
        assert tr == ref[mask[:-4]] and (int(v["width"]), int(v["height"])) == (4784, 3188)
        assert md5(out / "masks" / mask) == md5(src.parent / "masks" / mask)
    m = json.loads((out / "eval_sfm.json").read_text())
    assert "watertight" not in m and m["source_md5"] == md5(src) and len(m["distance_mm"]) == 6


@pytest.mark.data
def test_published_sneakers_mvs_has_no_watertight_copy(tmp_path):
    out = tmp_path / "ev"
    assert convert.main(["subsets", "--object", "sneakers", "--preset", "front3", "--data", str(DATA),
                         "--write-eval-sfm", str(out)]) == 0
    assert (DATA / "sneakers" / "gt" / "masks_watertight").is_dir()
    cams = eval_cameras(written_sfm(out))
    assert [c[0] for c in cams] == ["003.png", "009.png", "072.png"]  # preset order is 072, 003, 009
    assert sorted(os.listdir(out)) == ["eval_sfm.json", "front3.sfm", "masks"]


@pytest.mark.data
def test_published_sneakers_mvps_has_duplicated_pose_ids(tmp_path):
    out = tmp_path / "ev"
    assert convert.main(["subsets", "--object", "sneakers", "--preset", "front3", "--modality", "mvps", "--data",
                         str(DATA), "--write-eval-sfm", str(out)]) == 0
    sfm = json.loads((written_sfm(out)).read_text())
    assert len(sfm["poses"]) == len(sfm["views"]) == 3 and len(sfm["intrinsics"]) == 1
    assert "watertight" not in json.loads((out / "eval_sfm.json").read_text())
