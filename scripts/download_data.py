#!/usr/bin/env python3
"""
Martine dataset downloader (Harvard Dataverse).

Selective download by modality (MVS / MVPS / Ground Truth / Calibration),
resolution tier (d1/d2/d4/d8), and object subset.

With no modality flag it downloads ALL modalities (MVS, MVPS, Ground Truth and
Calibration). --tiers is required whenever MVS or MVPS images are selected.

    python download_data.py --tiers d8                 # everything at d8
    python download_data.py --mvs --tiers d8           # only MVS d8 (about 1 GB)
    python download_data.py --mvs --mvps --tiers d2 d4 # MVS and MVPS at d2 and d4
    python download_data.py --calibration              # no image: no tier needed
    python download_data.py --objects knife vase --mvs --tiers d1

------------------------------------------------------------------------------
ARCHIVES
The dataset is stored on Dataverse as TAR archives, one series per object x
modality x tier, each at most ~2 GB, e.g.
    bowl/mvps/d2/bowl_mvps_d2_part0001.tar
Each archive holds paths relative to the dataset root (bowl/mvps/d2/...). For
each selected archive the downloader
  1. downloads it into <out>/.archives/ (a partial download resumes where it
     stopped, network errors are retried),
  2. checks its size and the MD5 that Dataverse records for it,
  3. extracts it into <out>, refusing any member that is not a regular file or
     that lies outside the archive's own folder (no path traversal),
  4. deletes it (--keep-archives keeps it in <out>/.archives/).
An archive already extracted (marker <out>/.archives/<name>.done, files still
present with their sizes) is skipped, so an interrupted run can be restarted
with the same command. The marker records the MD5 of every extracted file:
--verify re-reads them and downloads again any archive with a corrupted file.
Extra disk space needed: about one archive.

DATAVERSE CONFIG
The downloader lists the dataset's files via the native API and filters on
each file's `directoryLabel` (its folder inside the dataset), which mirrors the
published layout:
    <object>/mvs/<tier>/   <object>/mvps/<tier>/   <object>/gt/   calibration/
No hard-coded file ids, so it keeps working if files are added/renamed.
Only the standard library is used. Before publication (draft), export an API
token with access to the dataset as DATAVERSE_API_TOKEN.
------------------------------------------------------------------------------
"""
import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

# ----------------------------------- CONFIG ---------------------------------
DATAVERSE_BASE = os.environ.get("DATAVERSE_BASE", "https://dataverse.harvard.edu")
DATASET_DOI = os.environ.get("MARTINE_DOI", "doi:10.7910/DVN/2BEPEH")
API_TOKEN = os.environ.get("DATAVERSE_API_TOKEN")  # only before publication
# Harvard Dataverse rejects the default "Python-urllib" User-Agent (HTTP 403).
USER_AGENT = "martine-dataset-downloader (+https://github.com/martine-benchmark/martine-dataset)"

ALL_OBJECTS = [
    "bowl", "can", "candle", "coin", "colander", "glass", "grid",
    "hammer_head", "jam_pot", "knife", "lego", "moka_pot", "paw_pad",
    "petanque_ball", "pillow", "plant", "plate", "rock", "sneakers",
    "spikes_ball", "spine", "straw_hat", "surgery_saw", "surgery_scissors",
    "vase", "warrior_head", "whisk", "wood",
]
TIERS = {"mvs": ("d1", "d2", "d4", "d8"), "mvps": ("d2", "d4", "d8")}

ARCHIVE_DIR = ".archives"
CHUNK = 1 << 20


# ------------------------------ Dataverse API -------------------------------

def _req(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    if API_TOKEN:
        # Not forwarded on redirects: file downloads are redirected to S3.
        req.add_unredirected_header("X-Dataverse-key", API_TOKEN)
    return urllib.request.urlopen(req, timeout=120)


def list_dataset_files(doi, version):
    """Return the list of file entries of a dataset version (:latest by default)."""
    url = (f"{DATAVERSE_BASE}/api/datasets/:persistentId/versions/{version}"
           f"?persistentId={urllib.parse.quote(doi)}")
    try:
        with _req(url) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        hint = "" if API_TOKEN else " (unpublished dataset? set DATAVERSE_API_TOKEN)"
        raise RuntimeError(f"Dataverse API error {e.code} for {doi}{hint}") from None
    if data.get("status") != "OK":
        raise RuntimeError(f"Dataverse API error: {data}")
    return data["data"]["files"]


def file_rel_path(entry):
    """directoryLabel/label -> the path we mirror locally."""
    df = entry["dataFile"]
    d = entry.get("directoryLabel", "")
    name = df["filename"]
    return f"{d}/{name}" if d else name


def _md5_of(path, h=None):
    h = h or hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h


def fetch(entry, dst, retries):
    """Download a data file to dst, resuming <dst>.part; check size and MD5."""
    df = entry["dataFile"]
    size, md5 = df["filesize"], df["checksum"]["value"].lower()
    if df["checksum"]["type"].upper() != "MD5":
        raise RuntimeError(f"unsupported checksum {df['checksum']['type']}")
    if os.path.exists(dst):
        if os.path.getsize(dst) == size and _md5_of(dst).hexdigest() == md5:
            return
        os.remove(dst)
    part = dst + ".part"
    url = f"{DATAVERSE_BASE}/api/access/datafile/{df['id']}"
    for attempt in range(retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        h = _md5_of(part) if have else hashlib.md5()
        try:
            if have < size:
                with _req(url, {"Range": f"bytes={have}-"}) as r:
                    if have and r.status != 206:  # range ignored: start over
                        have, h = 0, hashlib.md5()
                    with open(part, "ab" if have else "wb") as f:
                        for chunk in iter(lambda: r.read(CHUNK), b""):
                            f.write(chunk)
                            h.update(chunk)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            permanent = (isinstance(e, urllib.error.HTTPError)
                         and 400 <= e.code < 500 and e.code not in (408, 429))
            if permanent or attempt == retries:
                raise RuntimeError(f"download failed: {e}") from None
            print(f"    retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        got = os.path.getsize(part)
        if got == size and h.hexdigest() == md5:
            os.replace(part, dst)
            return
        os.remove(part)
        if attempt == retries:
            raise RuntimeError(f"bad download ({got} bytes, md5 {h.hexdigest()}; "
                               f"expected {size} bytes, md5 {md5})")
        print(f"    retry {attempt + 1}/{retries}: size or MD5 mismatch")
    raise RuntimeError("download failed")


# --------------------------------- archives ---------------------------------

def _checked_members(tf, label):
    """All members, refusing non-regular files and paths outside label/."""
    prefix = f"{label}/" if label else ""
    members = tf.getmembers()
    for m in members:
        parts = m.name.split("/")
        if (not m.isreg() or m.name.startswith("/") or "\\" in m.name
                or any(p in ("", ".", "..") for p in parts)
                or not m.name.startswith(prefix)):
            raise RuntimeError(f"unsafe archive member {m.name!r} (type {m.type!r})")
    return members


def extract(tar_path, label, out_root):
    """Extract a TAR into out_root; return [[name, size, md5], ...] of its members."""
    done = []
    with tarfile.open(tar_path, "r:") as tf:
        members = _checked_members(tf, label)
        for m in members:
            dst = os.path.join(out_root, *m.name.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            h = hashlib.md5()
            with tf.extractfile(m) as src, open(tmp, "wb") as f:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    f.write(chunk)
                    h.update(chunk)
            if os.path.getsize(tmp) != m.size:
                raise RuntimeError(f"short member {m.name}")
            os.replace(tmp, dst)
            done.append([m.name, m.size, h.hexdigest()])
    return done


def is_done(marker, md5, out_root, verify=False):
    """True if the archive was extracted and its files are still in place (and, with verify, intact)."""
    try:
        with open(marker) as f:
            done = json.load(f)
    except (OSError, ValueError):
        return False
    if done.get("md5") != md5 or not done.get("members"):
        return False
    for member in done["members"]:
        if len(member) != 3:  # marker written before per-file MD5s: cannot be verified, extract again
            return False
        name, size, file_md5 = member
        p = os.path.join(out_root, *name.split("/"))
        if not os.path.isfile(p) or os.path.getsize(p) != size:
            return False
        if verify and _md5_of(p).hexdigest() != file_md5:
            print(f"    corrupted: {name}")
            return False
    return True


def process(entry, out_root, keep, retries, verify=False):
    """Fetch one data file (and extract it if it is a TAR). Return a status."""
    label = entry.get("directoryLabel", "").strip("/")
    name = entry["dataFile"]["filename"]
    md5 = entry["dataFile"]["checksum"]["value"].lower()
    if not name.endswith(".tar"):
        dst = os.path.join(out_root, *file_rel_path(entry).split("/"))
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        cached = os.path.exists(dst)
        fetch(entry, dst, retries)
        return "cached" if cached else "ok"
    adir = os.path.join(out_root, ARCHIVE_DIR)
    os.makedirs(adir, exist_ok=True)
    tar_path, marker = os.path.join(adir, name), os.path.join(adir, name + ".done")
    if is_done(marker, md5, out_root, verify):
        return "verified" if verify else "done"
    fetch(entry, tar_path, retries)
    members = extract(tar_path, label, out_root)
    with open(marker + ".tmp", "w") as f:
        json.dump({"md5": md5, "members": members}, f)
    os.replace(marker + ".tmp", marker)
    if not keep:
        os.remove(tar_path)
    return f"extracted {len(members)} files"


# ------------------------------- selection ----------------------------------

def wanted(rel, objects, modalities, tiers, want_gt, want_calib):
    parts = rel.split("/")
    top = parts[0]
    if top == "calibration":
        return want_calib
    if top not in objects:
        return False
    if len(parts) < 2:
        return False
    sub = parts[1]
    if sub.lower() == "gt":
        return want_gt
    if sub in ("mvs", "mvps"):
        if sub not in modalities:
            return False
        tier = parts[2] if len(parts) > 2 else None
        return tier in tiers.get(sub, [])
    return False


def select_tiers(tiers):
    """Tiers per modality: the ``--tiers`` that the modality publishes."""
    return {m: sorted(t for t in set(tiers or ()) if t in TIERS[m]) for m in TIERS}


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="martine_dataset", help="download destination root")
    ap.add_argument("--mvs", action="store_true")
    ap.add_argument("--mvps", action="store_true")
    ap.add_argument("--groundtruth", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--tiers", nargs="+", choices=TIERS["mvs"], metavar="TIER",
                    help="resolution tiers of the MVS / MVPS images, required with them: d1 (9568x6376), "
                         "d2 (4784x3188), d4 (2392x1594), d8 (1196x797); MVPS has no d1")
    ap.add_argument("--objects", nargs="+", metavar="OBJECT", help="subset of object ids (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    ap.add_argument("--keep-archives", action="store_true",
                    help=f"keep the TAR archives in <out>/{ARCHIVE_DIR}/ after extraction")
    ap.add_argument("--version", default=":latest",
                    help="dataset version (:latest, :draft with a token, or e.g. 1.0)")
    ap.add_argument("--retries", type=int, default=5, help="retries per archive on network errors")
    ap.add_argument("--verify", action="store_true",
                    help="re-check the MD5 of every extracted file; a corrupted archive is downloaded again")
    args = ap.parse_args(argv)
    if args.retries < 0:
        ap.error("--retries must be >= 0")
    if os.path.exists(args.out) and not os.path.isdir(args.out):
        ap.error(f"--out {args.out} exists and is not a directory")

    # If no modality flag: fetch everything.
    any_mod = args.mvs or args.mvps or args.groundtruth or args.calibration
    want_mvs = args.mvs or not any_mod
    want_mvps = args.mvps or not any_mod
    want_gt = args.groundtruth or not any_mod
    want_calib = args.calibration or not any_mod

    modalities = set()
    if want_mvs:
        modalities.add("mvs")
    if want_mvps:
        modalities.add("mvps")

    if modalities and not args.tiers:
        ap.error(f"--tiers is required with the {' and '.join(sorted(m.upper() for m in modalities))} images "
                 f"(e.g. --tiers d8; {', '.join(TIERS['mvs'])})")
    tiers = select_tiers(args.tiers)
    for m in sorted(modalities):
        if not tiers[m]:
            print(f"WARNING: no {m.upper()} tier among {args.tiers} (published: {', '.join(TIERS[m])})",
                  file=sys.stderr)

    if args.objects:
        unknown = sorted(set(args.objects) - set(ALL_OBJECTS))
        if unknown:
            print(f"ERROR: unknown objects {unknown}; choose from {ALL_OBJECTS}", file=sys.stderr)
            return 2
        objects = set(args.objects)
    else:
        objects = set(ALL_OBJECTS)

    print(f"Dataset : {DATASET_DOI} ({args.version})")
    print(f"Objects : {len(objects)}  Modalities: {sorted(modalities)}  Tiers: {tiers}")
    print(f"GT: {want_gt}  Calibration: {want_calib}")

    try:
        files = list_dataset_files(DATASET_DOI, args.version)
    except (RuntimeError, urllib.error.URLError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    todo = sorted((e for e in files
                   if wanted(file_rel_path(e), objects, modalities, tiers, want_gt, want_calib)),
                  key=file_rel_path)
    if want_gt and not any(file_rel_path(e).split("/")[1:2] == ["gt"] for e in todo):
        print(f"WARNING: no gt/ archive in version {args.version} for the selected objects", file=sys.stderr)
    total = sum(e["dataFile"].get("filesize", 0) for e in todo)
    print(f"Selected {len(todo)} files, {total/1e9:.2f} GB")

    if args.dry_run:
        for e in todo[:50]:
            print("  ", file_rel_path(e))
        if len(todo) > 50:
            print(f"  ... (+{len(todo)-50} more)")
        return 0
    if not todo:
        return 0

    os.makedirs(args.out, exist_ok=True)
    free = shutil.disk_usage(args.out).free
    need = total + max(e["dataFile"]["filesize"] for e in todo)
    if free < need:
        print(f"WARNING: {free/1e9:.1f} GB free in {args.out}, up to {need/1e9:.1f} GB needed",
              file=sys.stderr)

    failed = []
    for i, e in enumerate(todo, 1):
        rel = file_rel_path(e)
        try:
            status = process(e, args.out, args.keep_archives, args.retries, args.verify)
        except (RuntimeError, OSError, tarfile.TarError) as err:
            failed.append(rel)
            status = f"FAILED: {err}"
        print(f"[{i}/{len(todo)}] {rel}: {status}", flush=True)
    if failed:
        print(f"{len(failed)} file(s) failed; rerun the same command to retry:", file=sys.stderr)
        for rel in failed:
            print("  ", rel, file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
