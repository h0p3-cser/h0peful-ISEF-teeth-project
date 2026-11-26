"""Prepare, freeze, run and verify the physical-patch experiment."""
import argparse
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="step",required=True)
    start=subs.add_parser("prepare")
    for key in ("data","study-run","baseline","out"):
        start.add_argument("--"+key,type=Path,required=True)
    seal=subs.add_parser("freeze")
    seal.add_argument("--prepared",type=Path,required=True)
    seal.add_argument("--regions",type=Path,required=True)
    go=subs.add_parser("run")
    go.add_argument("--prepared",type=Path,required=True)
    go.add_argument("--out",type=Path,required=True)
    verify=subs.add_parser("verify")
    for key in ("prepared","first","second"):
        verify.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args(argv)
    if args.step=="prepare":
        from .prepare import prepare
        result=prepare(args.data,args.study_run,args.baseline,args.out)
    elif args.step=="freeze":
        from .freeze import freeze
        result=freeze(args.prepared,args.regions)
    elif args.step=="run":
        from .execution import run
        result=run(args.prepared,args.out)
    else:
        from .verify import verify_runs
        result=verify_runs(args.prepared,args.first,args.second)
    print(result)
