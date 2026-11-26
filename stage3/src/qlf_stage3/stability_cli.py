"""Installed entry point for paired grayscale and original-RGB stability experiments."""
import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import sys

from .files import dump, load, outside_inputs, sha
from .modeling.seed_stability import BRANCHES, now, prepare, run_one


def train(root, plan):
    for branch in BRANCHES:
        for seed in plan["training_seeds"]:
            dest = root / branch / f"seed_{seed}"
            if (dest/"complete.json").is_file():
                completed = load(dest/"complete.json")
                if (load(dest/"run_config.json")["plan_sha256"] != sha(root/"plan.json") or
                    sha(dest/"oof_groups.csv") != completed["group_oof_sha256"] or
                    sha(dest/"oof_frames.csv") != completed["frame_oof_sha256"]):
                    raise ValueError(f"Completed result differs from its recorded inputs or outputs: {dest}")
                logging.info("Resuming after completed %s seed %d",branch,seed)
                continue
            if dest.exists():
                archived = dest.with_name(dest.name+"_interrupted_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                if not dest.resolve().is_relative_to(root) or not archived.resolve().is_relative_to(root):
                    raise ValueError("Unexpected attempt directory")
                dest.rename(archived)
            dest.mkdir(parents=True)
            cmd = [sys.executable,"-B","-X","utf8","-m","qlf_stage3","stability",
                   "--out",str(root),"--worker",branch,str(seed)]
            env = dict(os.environ,PYTHONHASHSEED=str(seed),PYTHONIOENCODING="utf-8",OMP_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
            logging.info("Starting %s seed %d",branch,seed)
            with (dest/"run.log").open("w",encoding="utf8",buffering=1) as log, \
                 subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                  encoding="utf8",env=env) as child:
                for line in child.stdout:
                    log.write(line)
                    logging.info("[%s/%d] %s",branch,seed,line.rstrip())
                if child.wait():
                    raise RuntimeError(f"{branch} seed {seed} failed. See {dest/'run.log'}")
            logging.info("Completed %s seed %d",branch,seed)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qlf-stage3 stability",description=__doc__)
    parser.add_argument("--data",type=Path,help="Portable dataset directory")
    parser.add_argument("--features",type=Path,help="Verified four-feature CSV beside frames.csv and its verification JSON")
    parser.add_argument("--weights","--checkpoint",dest="weights",type=Path,help="Optional local ImageNet V1 checkpoint")
    parser.add_argument("--seeds",type=int,nargs=3,default=[42,123,2024])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--step",choices=["all","prepare","train","report"],default="all")
    parser.add_argument("--worker",nargs=2,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.out.resolve()
    format_string = "%(asctime)s %(levelname)s %(message)s"
    if args.worker:
        logging.basicConfig(level=logging.INFO,format=format_string,stream=sys.stdout)
        run_one(root,args.worker[0],int(args.worker[1]))
        return 0
    if args.step in ("all","prepare"):
        if not args.data or not args.features:
            parser.error("Supply --data and --features to prepare a new cohort")
        outside_inputs(root,args.data,args.features.parent)
        if (root/"plan.json").exists():
            parser.error("Use a new --out or resume with --step train / report")
    elif not (root/"plan.json").is_file():
        parser.error("Supply the --out directory of a prepared experiment")
    root.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format=format_string,handlers=[logging.StreamHandler(sys.stdout),
                             logging.FileHandler(root/"run.log",encoding="utf8")],force=True)
    logging.info("Output %s, step %s",root,args.step)
    try:
        if args.step in ("all","prepare"):
            prepare(root,args.data.resolve(),args.features.resolve(),args.weights,args.seeds)
        if args.step in ("all","train"):
            train(root,load(root/"plan.json"))
        if args.step in ("all","report"):
            from .modeling.seed_report import report
            report(root)
    except Exception as exc:
        logging.exception("Experiment interrupted")
        dump(root/"status.json",dict(status="failed",step=args.step,at=now(),error=str(exc)))
        raise
    status = {"prepare":"prepared","train":"trained","report":"complete","all":"complete"}[args.step]
    dump(root/"status.json",dict(status=status,at=now()))
    logging.info("%s",status)
    return 0
