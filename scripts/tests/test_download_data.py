# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baptiste Brument and the Martine authors
"""Offline tests of scripts/download_data.py: selection (objects, modalities, tiers), dry run, extraction of the TAR
archives (resume marker, unsafe members). The Dataverse API is replaced by a fake file list."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "download_data.py"


@pytest.fixture()
def dd():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tar(path: Path, members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    path.write_bytes(buf.getvalue())
    return buf.getvalue()


def _entry(label: str, name: str, payload: bytes, fid: int) -> dict:
    return {"directoryLabel": label, "dataFile": {"id": fid, "filename": name, "filesize": len(payload),
                                                  "checksum": {"type": "MD5",
                                                               "value": hashlib.md5(payload).hexdigest()}}}


@pytest.fixture()
def remote(tmp_path, dd, monkeypatch):
    """A fake dataset: rock and vase, MVS d1/d8, MVPS d2/d8, no gt/ archive, one calibration archive."""
    src = tmp_path / "remote"
    src.mkdir()
    entries, blobs = [], {}
    for k, label in enumerate(["rock/mvs/d1", "rock/mvs/d8", "rock/mvps/d2", "rock/mvps/d8", "vase/mvs/d8",
                               "calibration/lights"], start=1):
        name = label.replace("/", "_") + "_part0001.tar"
        blobs[k] = _tar(src / name, {f"{label}/sfm.json": b"{}", f"{label}/images/000.exr": b"x" * (100 * k)})
        entries.append(_entry(label, name, blobs[k], k))
    monkeypatch.setattr(dd, "list_dataset_files", lambda doi, version: entries)

    def fake_fetch(entry, dst, retries):
        Path(dst).write_bytes(blobs[entry["dataFile"]["id"]])

    monkeypatch.setattr(dd, "fetch", fake_fetch)
    return entries


def _selected(dd, capsys, argv):
    assert dd._main(argv + ["--dry-run"]) == 0
    return sorted(line.strip().rsplit("/", 1)[0] for line in capsys.readouterr().out.splitlines()
                  if line.startswith("   "))


# --- tiers -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("argv", [["--objects", "rock"], ["--mvs"], ["--mvps", "--groundtruth"]])
def test_tiers_are_required_for_images(dd, remote, capsys, argv):
    with pytest.raises(SystemExit) as e:
        dd._main(argv + ["--dry-run"])
    assert e.value.code == 2
    assert "--tiers" in capsys.readouterr().err


def test_ground_truth_and_calibration_need_no_tier(dd, remote, capsys):
    assert _selected(dd, capsys, ["--objects", "rock", "--calibration"]) == ["calibration/lights"]


def test_everything_at_the_given_tiers(dd, remote, capsys):
    assert _selected(dd, capsys, ["--objects", "rock", "--tiers", "d8"]) == [
        "calibration/lights", "rock/mvps/d8", "rock/mvs/d8"]


def test_tiers_replace_the_defaults(dd, remote, capsys):
    assert _selected(dd, capsys, ["--objects", "rock", "--mvs", "--tiers", "d8"]) == ["rock/mvs/d8"]
    assert _selected(dd, capsys, ["--objects", "rock", "vase", "--mvs", "--mvps", "--tiers", "d8"]) == [
        "rock/mvps/d8", "rock/mvs/d8", "vase/mvs/d8"]


def test_tiers_d1_has_no_mvps(dd, remote, capsys):
    assert _selected(dd, capsys, ["--objects", "rock", "--mvs", "--mvps", "--tiers", "d1"]) == ["rock/mvs/d1"]


@pytest.mark.parametrize("argv", [["--tiers", "d8", "--d4"], ["--mvs", "--d8"], ["--tiers", "d3"], ["--tiers"],
                                  ["--tiers", "d8", "--objects"], ["--tiers", "d8", "--retries", "-1"]])
def test_bad_arguments(dd, remote, argv):
    with pytest.raises(SystemExit):
        dd._main(argv + ["--dry-run"])


def test_missing_ground_truth_is_reported(dd, remote, capsys):
    assert dd._main(["--objects", "rock", "--groundtruth", "--dry-run"]) == 0
    assert "no gt/ archive" in capsys.readouterr().err


# --- download and extraction ----------------------------------------------------------------------------------------
def test_download_extracts_and_resumes(dd, remote, tmp_path, capsys):
    out = tmp_path / "data"
    argv = ["--objects", "rock", "--mvs", "--tiers", "d8", "--out", str(out)]
    assert dd._main(argv) == 0
    assert (out / "rock" / "mvs" / "d8" / "sfm.json").read_bytes() == b"{}"
    assert (out / "rock" / "mvs" / "d8" / "images" / "000.exr").stat().st_size == 200
    assert not list((out / ".archives").glob("*.tar"))  # archives deleted after extraction
    capsys.readouterr()
    assert dd._main(argv) == 0
    assert ": done" in capsys.readouterr().out  # marker found: nothing fetched again


def test_out_must_be_a_directory(dd, remote, tmp_path):
    (tmp_path / "f").write_text("")
    with pytest.raises(SystemExit):
        dd._main(["--objects", "rock", "--mvs", "--tiers", "d8", "--out", str(tmp_path / "f")])


def test_verify_repairs_a_corrupted_file(dd, remote, tmp_path, capsys):
    out = tmp_path / "data"
    argv = ["--objects", "rock", "--mvs", "--tiers", "d8", "--out", str(out)]
    assert dd._main(argv) == 0
    img = out / "rock" / "mvs" / "d8" / "images" / "000.exr"
    img.write_bytes(b"y" * 200)  # same size, other content
    capsys.readouterr()
    assert dd._main(argv) == 0
    assert ": done" in capsys.readouterr().out  # size check alone misses it
    assert dd._main(argv + ["--verify"]) == 0
    assert "corrupted: rock/mvs/d8/images/000.exr" in capsys.readouterr().out
    assert img.read_bytes() == b"x" * 200
    assert dd._main(argv + ["--verify"]) == 0
    assert ": verified" in capsys.readouterr().out


def test_marker_without_file_md5_is_extracted_again(dd, remote, tmp_path, capsys):
    out = tmp_path / "data"
    argv = ["--objects", "rock", "--mvs", "--tiers", "d8", "--out", str(out)]
    assert dd._main(argv) == 0
    marker = next((out / ".archives").glob("*.done"))
    old = json.loads(marker.read_text())
    marker.write_text(json.dumps({"md5": old["md5"], "members": [m[:2] for m in old["members"]]}))
    capsys.readouterr()
    assert dd._main(argv) == 0
    assert "extracted 2 files" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["rock/mvs/d8/../../../evil", "/etc/evil", "vase/mvs/d8/sfm.json"])
def test_unsafe_members_are_refused(dd, tmp_path, name):
    tar = tmp_path / "a.tar"
    _tar(tar, {name: b"x"})
    with pytest.raises(RuntimeError, match="unsafe"):
        dd.extract(str(tar), "rock/mvs/d8", str(tmp_path / "out"))
    assert not (tmp_path / "out").exists() or not any(os.scandir(tmp_path / "out"))


# --- calibration -----------------------------------------------------------------------------------------------------
CALIBRATION = {  # folders and archive names of the published calibration (Dataverse version 1.0)
    "calibration/lights": "calibration_lights_part0001.tar",
    "calibration/spheres/lambertian": "calibration_spheres_lambertian_part0002.tar",
    "calibration/spheres/specular": "calibration_spheres_specular_part0001.tar",
    "calibration/color_checker": "calibration_color_checker_part0001.tar",
}


@pytest.mark.parametrize("label", CALIBRATION)
def test_calibration_selects_lights_spheres_and_color_checker(dd, label):
    rel = f"{label}/{CALIBRATION[label]}"
    no_images = {"mvs": [], "mvps": []}
    assert dd.wanted(rel, ["rock"], [], no_images, False, True)
    assert not dd.wanted(rel, ["rock"], ["mvs"], {"mvs": ["d8"], "mvps": []}, True, False)


@pytest.mark.parametrize("label", CALIBRATION)
def test_calibration_archives_extract_into_their_folder(dd, tmp_path, label):
    tar = tmp_path / CALIBRATION[label]
    _tar(tar, {f"{label}/README.md": b"r", f"{label}/d1/images/view_00/lighting_00.exr": b"x"})
    dd.extract(str(tar), label, str(tmp_path / "out"))
    assert (tmp_path / "out" / label / "README.md").read_bytes() == b"r"
