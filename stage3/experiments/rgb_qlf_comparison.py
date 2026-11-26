"""Three endpoint-3 models with measured QLF, plus a paired original-RGB control."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))

from qlf_stage3.bstar.fusion_inputs import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',type=Path,default=PACKAGE/'runs/bstar')
    parser.add_argument('--history',type=Path,default=PACKAGE/'runs/study/evaluation')
    parser.add_argument('--features',type=Path,default=PACKAGE/'runs/study/measurements/training-features-255-thr30.csv')
    parser.add_argument('--checkpoint',type=Path,default=PACKAGE/'data/study/weights/resnet18-f37072fd.pth')
    parser.add_argument('--rgb-root',type=Path,help='Relocated original Data_rgb_corrected directory')
    parser.add_argument('--out',type=Path,help='New comparison directory, or the encoded directory for --step compare')
    parser.add_argument('--step',choices=['all','encode','compare'],default='all')
    args=parser.parse_args()
    if args.step=='compare' and not args.out:
        parser.error('Provide --out for an encoded run')
    out=(args.out or PACKAGE/'runs'/('endpoint3_rgb_qlf255_thr30_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    print(out,flush=True)
    if args.step in ('all','encode'):
        encode(out,args.branch.resolve(),args.history.resolve(),args.features.resolve(),
               args.checkpoint.resolve(),args.rgb_root.resolve() if args.rgb_root else None)
    if args.step in ('all','compare'):
        from qlf_stage3.bstar.fusion import compare
        compare(out,args.checkpoint.resolve())


if __name__=='__main__':
    main()
