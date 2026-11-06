# Download script

`download_data.py` downloads the Martine dataset from Harvard Dataverse
([doi:10.7910/DVN/2BEPEH](https://doi.org/10.7910/DVN/2BEPEH)), selectively by modality, resolution tier and object.
It needs the Python standard library only.

```bash
python scripts/download_data.py --tiers d8 --dry-run                   # every modality at d8: list and size
python scripts/download_data.py --objects rock --mvs --groundtruth --tiers d8
python scripts/download_data.py --mvs --tiers d2 d4                    # MVS at d2 and d4
python scripts/download_data.py --objects knife vase --mvps --tiers d2 --out /data/martine
python scripts/download_data.py --groundtruth --calibration            # no image, no tier needed
```

`--tiers` (`d1`, `d2`, `d4`, `d8`; MVPS has no d1) is required whenever MVS or MVPS images are selected, which is the
case with no modality flag. The options are listed in the [README](../README.md#download) and by `--help`.

The files are TAR archives of at most 2 GB, listed through the Dataverse API by their folder in the dataset
(`<object>/mvs/<tier>/`, `<object>/mvps/<tier>/`, `<object>/gt/`, `calibration/`). Each one is downloaded into
`<out>/.archives/` (a partial download resumes), checked against its MD5, extracted into `<out>`, then deleted
(`--keep-archives` keeps it). Extraction refuses any member that is not a regular file or that lies outside the
archive's folder. Rerunning the same command skips the archives already extracted (files present with their
sizes); with `--verify` it also re-checks each file against the MD5 recorded at extraction and downloads again any
archive with a corrupted file.
Tests: `uv run pytest scripts/tests` (offline).
