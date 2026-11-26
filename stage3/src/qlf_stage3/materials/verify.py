"""Recompute the pilot in a fresh directory, then check the original stored evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ..files import dump, load, sha
from ..rgb.color import CHANNELS, decode, pixel_hash
from .pilot import run


def verify(data,study_run,root_run,original):
    original=Path(original).resolve()
    stat=load(original/"status.json")
    if stat.get("status")!="complete" or stat.get("frame_count")!=63:
        raise ValueError("An unexpected or incomplete pilot cannot be verified")
    for path,digest in stat["sha256"].items():
        file=(original/path).resolve()
        if not file.is_relative_to(original) or not file.is_file() or sha(file)!=digest:
            raise ValueError("Missing or changed stored evidence: "+path)
    old=pd.read_csv(original/"frame_audit.csv").set_index("rgb_image_id",verify_integrity=True)
    ok=old.status.isin(["reference","ok"])
    if (not old.loc[ok,"corrected_path"].notna().all() or old.loc[~ok,"corrected_path"].notna().any()
            or set(old.loc[ok,"corrected_path"])!={p.relative_to(original).as_posix() for p in (original/"corrected_rgb").glob("*.png")}):
        raise ValueError("Status and output PNG set differ")
    if not np.isfinite(old.loc[ok,[k+c for c in CHANNELS for k in ("a_","b_")]].to_numpy()).all():
        raise ValueError("An exported mapping has no finite coefficients")
    for row in old.loc[ok].itertuples():
        pixels,_=decode(original/row.corrected_path,corrected=True)
        if pixel_hash(pixels)!=row.corrected_pixel_sha256:
            raise ValueError("The stored PNG does not match its native pixel hash: "+row.Index)
    with TemporaryDirectory(prefix="material-pilot-replay-",dir=original.parent) as tmp:
        rep=Path(tmp)/"independent"
        run(data,study_run,root_run,rep,original/"protocol.json",original/"regions.json")
        repeat=load(rep/"status.json")["sha256"]
        recorded=stat["sha256"]
        if set(recorded)!=set(repeat):
            raise ValueError("Replay produced a different artifact set")
        original_env=load(original/"provenance.json")
        replay_env=load(rep/"provenance.json")
        strict_env=original_env["versions"]==replay_env["versions"]
        for path,digest in recorded.items():
            if repeat[path]==digest:
                continue
            # The native hashes and numeric tables must reproduce across installations.
            # PNG container bytes and rendered fonts can vary with the installed encoders.
            if path.endswith(".csv"):
                first=pd.read_csv(original/path)
                second=pd.read_csv(rep/path)
                keep=[c for c in first if c not in ("corrected_sha256",)]
                pd.testing.assert_frame_equal(first[keep],second[keep],rtol=1e-6,atol=1e-8)
                continue
            if path.startswith(("figures/","corrected_rgb/")):
                continue
            if path=="provenance.json" and all(original_env[k]==replay_env[k] for k in ("inputs","annotation_sha256","protocol_sha256")):
                continue
            raise ValueError("Replay mismatch: "+path)
        rerun=pd.read_csv(rep/"frame_audit.csv").set_index("rgb_image_id",verify_integrity=True)
        if not old.corrected_pixel_sha256.fillna("").equals(rerun.corrected_pixel_sha256.fillna("")):
            raise ValueError("Native calibrated pixel hashes differ")
        case=old.loc["I0654"]
        if int(case.old_crown_pixels)!=204958 or int(case.old_crown_new_clip_px)!=110122:
            raise ValueError("I0654 did not reproduce the historical native crown result")
        result=dict(status="verified",frames=len(old),replayed_png=int(ok.sum()),native_pixel_hashes_match=True,
                    artifact_count=len(recorded),identical_artifact_hashes=sum(v==repeat[p] for p,v in recorded.items()),
                    same_library_versions=strict_env,identical_source_files=original_env["source_files"]==replay_env["source_files"],
                    old_I0654_crown_pixels=204958,old_I0654_new_clip_px=110122)
    dump(original/"validation.json",result)
    return result
