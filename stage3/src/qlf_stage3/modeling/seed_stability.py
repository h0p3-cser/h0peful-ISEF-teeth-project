"""Same-source RGB and grayscale experiments with independent seeded fits and locked tooth folds."""
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import logging
import os
import platform
import random
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ..bundle import check_data
from ..files import PROTOCOL, config, dump, load, relative_path, sha
from ..rgb.color import RAW_MODE
from .cohort import MODELS, build_cohort, ensure, index_by, save_csv
from .evaluation import evaluate, read_result
from .training import fold_predictions, group_features
from .validation import verify_predictions
from .vision import GRAY_MODE, encoder_weights, imagenet18, white_vectors


BRANCHES = {"gray": GRAY_MODE, "rgb": RAW_MODE}
LOG = logging.getLogger(__name__)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def prepare(root, dataset, feature_csv, checkpoint, seeds):
    ensure(len(set(seeds)) == len(seeds) == 3 and all(0 <= s < 2**32 for s in seeds),
           "Choose three different unsigned 32-bit seeds")
    LOG.info("Verifying the dataset and the fixed 30 percent QLF export")
    checked = check_data(dataset)
    cfg = config(dataset, feature_csv)
    manifest, _, verification = build_cohort(cfg)
    frames = manifest.sort_values("sample_id").reset_index(drop=True)
    ensure(frames.included.all(), "A reviewed pair was lost")
    study = cfg["kind"] == "study"
    catalog = index_by(cfg["capture_catalog"]/"images.csv", "image_id")
    for row in frames.itertuples():
        source = (dataset / row.rgb_resolved).resolve()
        entry = catalog[row.rgb_image_id]
        expected = entry["sha256"] if study else entry.get("sha256", row.rgb_sha256)
        ensure(source.is_relative_to(dataset) and expected == row.rgb_sha256 == sha(source),
               "White-light input differs from the registered original: " + row.rgb_image_id)
    supplied = frames[(frames.rgb_catalog_tooth == "T005") & (frames.state_id == "S2")]
    if study:
        ensure(len(frames) == 873 and frames.group_id.nunique() == 127, "Locked study cohort differs")
        ensure(len(supplied) == 35 and all(catalog[i]["pixel_processing"] == "original_as_supplied"
               for i in supplied.rgb_image_id), "The 35 designated originals are not installed")
    packaged_weights = dataset / "weights/resnet18-f37072fd.pth"
    checkpoint, _ = encoder_weights(dict(cfg, checkpoint=checkpoint or (packaged_weights if packaged_weights.is_file() else None)))
    teeth = sorted(frames.tooth_id.unique())
    frames["test_fold"] = frames.tooth_id.map({tooth: n for n, tooth in enumerate(teeth, 1)})
    save_csv(root, "paired_inputs.csv", frames)
    cfg_fields = ["weights","weights_sha256","pca_components","classifier","states","feature_columns","bootstrap_seed","bootstrap_replicates"]
    plan = dict(created=now(), dataset_kind=cfg["kind"], branches=BRANCHES, training_seeds=list(seeds), dataset=relative_path(dataset,root),
                feature_csv=relative_path(feature_csv,root), checkpoint=relative_path(checkpoint,root),
                protocol={k:cfg[k] for k in cfg_fields}, dataset_check=checked, paired_frames=len(frames),
                groups=int(frames.group_id.nunique()), teeth=teeth, frames_by_stage=verification["exported_by_stage"],
                designated_t005_frames=len(supplied), training_color_correction=False, independent_reencoding=True,
                split="leave one tooth out; fixed acquisition groups; same paired data in all six runs",
                sha256=dict(paired_inputs=sha(root/"paired_inputs.csv"), features=sha(feature_csv),
                            checkpoint=sha(checkpoint), protocol=sha(PROTOCOL), dataset_index=sha(dataset/"files.sha256")),
                python=platform.python_version(), platform=platform.platform(),
                versions={p:importlib.metadata.version(p) for p in ("torch","torchvision","scikit-learn","numpy",
                                                                      "scipy","pandas","Pillow","joblib","threadpoolctl")})
    dump(root / "plan.json",plan)
    LOG.info("Locked %d local pairs, %d groups, nine teeth, train seeds %s, bootstrap seed %s",
             len(frames),plan["groups"],seeds,cfg["bootstrap_seed"])


def run_one(root, branch, seed):
    import torch

    started = perf_counter()
    plan = load(root / "plan.json")
    ensure(branch in plan["branches"] and seed in plan["training_seeds"], "Unregistered run")
    ensure(os.environ.get("PYTHONHASHSEED") == str(seed), "Start each worker with its registered seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mode = plan["branches"][branch]
    out = root / branch / f"seed_{seed}"
    ensure(not (out/"run_config.json").exists(), "Choose a fresh work directory")
    out.mkdir(parents=True,exist_ok=True)
    dataset,features,checkpoint = [(root / plan[k]).resolve() for k in ("dataset","feature_csv","checkpoint")]
    ensure(sha(root/"paired_inputs.csv") == plan["sha256"]["paired_inputs"] and
           sha(features) == plan["sha256"]["features"] and sha(PROTOCOL) == plan["sha256"]["protocol"],
           "A locked input changed")
    cfg = dict(plan["protocol"], training_seed=seed, white_light_mode=mode, qlf_workflow="endpoint3_"+branch+"_255_thr30",
               checkpoint=str(checkpoint), feature_csv=features)
    dump(out/"run_config.json",dict(cfg, dataset=relative_path(dataset,out),checkpoint=relative_path(checkpoint,out),
         feature_csv=relative_path(features,out), plan_sha256=sha(root/"plan.json"), branch=branch, process_id=os.getpid(),
         started=now(), torch_threads=4, blas_threads=1, white_preprocessing="full field, long edge 224, black padding"))
    frames = read_result(root,"paired_inputs.csv")
    paths = [(dataset/p).resolve() for p in frames.rgb_resolved]
    LOG.info("%s seed %d, PID %d. Encoding %d original images with a fresh frozen ResNet-18", branch,seed,os.getpid(),len(paths))
    net,_,_ = imagenet18(cfg)
    with threadpool_limits(limits=1,user_api="blas"):
        visual = white_vectors(paths,net,mode)
        for path,digest in zip(paths,frames.rgb_sha256):
            ensure(sha(path) == digest, "Input changed during extraction: " + str(path))
        frames["embedding_cache_key"] = [sha256("|".join([d,mode,cfg["weights_sha256"]]).encode()).hexdigest()
                                          for d in frames.rgb_sha256]
        save_csv(out,"sample_manifest.csv",frames)
        np.savez_compressed(out/"frozen_frame_features.npz",sample_ids=frames.sample_id.to_numpy(str),
                            cache_keys=frames.embedding_cache_key.to_numpy(str),white=visual)
        groups,white,qlf = group_features(frames,visual)
        save_csv(out,"groups.csv",groups)
        fold_predictions(cfg,out,groups,white,qlf,frames,visual)
        dump(out/"oof_sha256.json",{name:sha(out/name) for name in ("oof_frames.csv","oof_groups.csv")})
        results = evaluate(cfg,out)
        validation = verify_predictions(out,features)
    if seed == plan["training_seeds"][0]:
        from .report import figures
        from .vision import white_input
        visual_name = "灰度" if branch == "gray" else "原图 RGB"
        figures(out, results, dataset, names={"M1":"M1 "+visual_name,"M2":"M2 QLF","M3":"M3 "+visual_name+"＋QLF"},
                photo_input=lambda path:white_input(path,mode))
    for tooth in plan["teeth"]:
        bundle = joblib.load(out/"models"/(tooth+".joblib"))
        ensure(bundle["training_seed"] == bundle["pca"].random_state == seed and
               all(bundle["heads"][m].random_state == seed for m in MODELS), "Saved model seed mismatch")
    LOG.info("Replay verified %d heads, %d group and %d frame predictions; max probability difference %.3g",
             validation["converged_heads"],validation["oof_group_rows"],validation["oof_frame_rows"],
             validation["max_saved_model_probability_difference"])
    for r in results["summary"].itertuples():
        LOG.info("%s tooth-equal Macro-F1 %.9f; %d/%d groups correct",r.model,r.primary_macro_f1,r.correct,r.groups)
    dump(out/"validation.json",validation)
    dump(out/"complete.json",dict(finished=now(),seconds=round(perf_counter()-started,2),
         branch=branch,seed=seed,labels_per_group=True,group_oof_sha256=sha(out/"oof_groups.csv"),
         frame_oof_sha256=sha(out/"oof_frames.csv"),folds_sha256=sha(out/"folds.csv"),
         visual_sha256=sha256(np.ascontiguousarray(visual).tobytes()).hexdigest(),
         heads=validation["converged_heads"]))
