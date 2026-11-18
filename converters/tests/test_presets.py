# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Sparse-view presets (--preset front3 | 360_6): fixed MVS views per object, read from the package
(view_presets.json), with the MVPS pose nearest to each of them; no scikit-learn, scipy or MVS sfm.json needed."""
from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest
from conftest import DATA
from martine_converters import cli as convert
from martine_converters import subsets
from martine_converters.export import MANIFEST
from martine_converters.subsets import PRESETS, load_presets, preset_entry, preset_names
from test_subsets import no_deps, ring, write_sfm

GT_OBJECTS = 26  # the objects with a GT (coin and pillow have none)


def test_presets_cover_the_objects_with_a_gt():
    table = load_presets()
    assert PRESETS == ("front3", "360_6") and sorted(table["presets"]) == sorted(PRESETS)
    assert len(table["objects"]) == GT_OBJECTS and not {"coin", "pillow"} & set(table["objects"])
    for obj, by_preset in table["objects"].items():
        assert sorted(by_preset) == sorted(PRESETS), obj
        for name, n in (("front3", 3), ("360_6", 6)):
            e = by_preset[name]
            assert sorted(e) == ["distance_mm", "mvps", "mvs"], (obj, name)
            assert len(e["mvs"]) == len(set(e["mvs"])) == len(set(e["mvps"])) == len(e["distance_mm"]) == n, obj
            assert all(len(v) == 3 and v.isdigit() for v in e["mvs"]), obj
            assert all(p.startswith("pose_") for p in e["mvps"]), obj


def test_presets_values():
    """A few entries as given (the 360 lists keep their order; five objects differ from the common set)."""
    o = load_presets()["objects"]
    assert o["bowl"]["front3"]["mvs"] == ["043", "052", "061"]
    assert o["rock"]["360_6"]["mvs"] == ["017", "061", "077", "034", "016", "060"]
    for obj in ("plant", "spine", "bowl", "grid"):
        assert o[obj]["360_6"]["mvs"][-1] == "033", obj
    assert o["vase"]["360_6"]["mvs"][-1] == "047"


def test_presets_file_holds_only_the_views():
    table = json.loads(files("martine_converters").joinpath("view_presets.json").read_text())
    assert sorted(table) == ["description", "objects", "presets"]


@pytest.mark.parametrize("modality", ["mvs", "mvps"])
def test_converters_preset_selects_the_views(tmp_path, monkeypatch, modality):
    """MVS or MVPS images alone: the preset needs neither the other modality nor scikit-learn or scipy."""
    monkeypatch.setattr(subsets, "_deps", no_deps)
    write_sfm(tmp_path / "rock" / modality / "d8" / "sfm.json", ring(84), modality)
    out = tmp_path / "out"
    assert convert.main(["mvs", "--data", str(tmp_path), "--object", "rock", "--tier", "d8", "--modality", modality,
                         "--preset", "360_6", "--format", "colmap", "--points", "none", "--no-images",
                         "--out", str(out)]) == 0
    expected = preset_entry("rock", "360_6")[modality]
    assert json.loads((out / MANIFEST).read_text())["units"] == sorted(expected)
    assert preset_names(tmp_path / "rock" / modality / "d8" / "sfm.json", "360_6") == sorted(expected)


def test_preset_and_views_are_exclusive(tmp_path):
    write_sfm(tmp_path / "rock" / "mvs" / "d8" / "sfm.json", ring(84), "mvs")
    for other in (["--views", "0:3"], ["--n-views", "5"]):
        with pytest.raises(SystemExit):
            convert.main(["mvs", "--data", str(tmp_path), "--object", "rock", "--tier", "d8", "--preset", "front3",
                          *other, "--format", "colmap", "--out", str(tmp_path / "x")])


def test_unknown_preset_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        convert.main(["subsets", "--object", "rock", "--preset", "front4"])


def test_object_without_preset_is_a_clear_error(tmp_path, capsys):
    write_sfm(tmp_path / "coin" / "mvs" / "d8" / "sfm.json", ring(84), "mvs")
    assert convert.main(["mvs", "--data", str(tmp_path), "--object", "coin", "--tier", "d8", "--preset", "front3",
                         "--format", "colmap", "--points", "none", "--no-images", "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "front3" in err and "coin" in err


def test_preset_view_missing_from_the_sfm_is_an_error(tmp_path, capsys):
    write_sfm(tmp_path / "rock" / "mvs" / "d8" / "sfm.json", ring(40), "mvs")  # views 000..039 only
    assert convert.main(["mvs", "--data", str(tmp_path), "--object", "rock", "--tier", "d8", "--preset", "360_6",
                         "--format", "colmap", "--points", "none", "--no-images", "--out", str(tmp_path / "o")]) == 2
    assert "not in it" in capsys.readouterr().err


def test_cli_subsets_prints_the_preset(capsys, monkeypatch):
    monkeypatch.setattr(subsets, "_deps", no_deps)
    assert convert.main(["subsets", "--object", "rock", "--preset", "front3"]) == 0
    e = preset_entry("rock", "front3")
    text = capsys.readouterr().out
    assert f"--views {','.join(sorted(e['mvs']))}" in text and f"--views {','.join(sorted(e['mvps']))}" in text


@pytest.mark.data
def test_presets_have_a_pose_in_every_tier_and_match_the_nearest_mvps_pose():
    from martine_converters.sfm import read_sfm

    for obj, by_preset in load_presets()["objects"].items():
        tiers = {m: sorted(p.parent.name for p in (DATA / obj / m).glob("*/sfm.json")) for m in ("mvs", "mvps")}
        names = {(m, t): set(read_sfm(DATA / obj / m / t / "sfm.json").names) for m in tiers for t in tiers[m]}
        ev = json.loads((DATA / obj / "gt" / "eval_cameras.sfm").read_text())
        posed = {p["poseId"] for p in ev["poses"]}
        eval_names = {v["path"].rsplit("/", 1)[-1][:-4] for v in ev["views"] if v["poseId"] in posed}
        mvs, mvps = (read_sfm(subsets.first_sfm(DATA, obj, m)) for m in ("mvs", "mvps"))
        for name, e in by_preset.items():
            for (m, t), have in names.items():
                assert set(e[m]) <= have, (obj, name, m, t)
            assert set(e["mvs"]) <= eval_names, (obj, name)
            sel = subsets.centres(mvs)[[mvs.names.index(v) for v in e["mvs"]]]
            idx, dist = subsets.match_views(sel, subsets.centres(mvps))
            assert [mvps.units[i].name for i in idx] == e["mvps"], (obj, name)
            assert e["distance_mm"] == pytest.approx(np.round(dist, 2), abs=0.006), (obj, name)
