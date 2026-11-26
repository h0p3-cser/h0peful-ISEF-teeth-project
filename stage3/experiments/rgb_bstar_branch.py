"""Reproduce the paired full-frame b* RGB and same-source grayscale experiment."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))

from qlf_stage3.bstar.inputs import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['all', 'prepare', 'score', 'gray-qc'], default='all')
    parser.add_argument('--out', type=Path, help='New run directory, or existing one for score/gray-qc')
    parser.add_argument('--history', type=Path, default=PACKAGE / 'runs/study/evaluation')
    parser.add_argument('--rgb-root', type=Path, default=PACKAGE / 'data/study/workspace/Data_rgb_corrected')
    parser.add_argument('--config', type=Path, default=PACKAGE / 'src/qlf_stage3/config/protocol.json')
    parser.add_argument('--checkpoint', type=Path, default=PACKAGE / 'data/study/weights/resnet18-f37072fd.pth')
    parser.add_argument('--crown-run', type=Path, default=PACKAGE / 'runs/rgb_full')
    parser.add_argument('--study-data', type=Path, default=PACKAGE / 'data/study')
    parser.add_argument('--jobs', type=int, default=3, help='Parallel native image writers')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.step in ('score', 'gray-qc') and not args.out:
        parser.error('Provide the prepared --out directory')
    out = (args.out or PACKAGE / 'runs' / ('endpoint3_rgb_bstar_' +
           datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    print(out, flush=True)
    if args.step in ('all', 'prepare'):
        prepare(out, args.history.resolve(), args.rgb_root.resolve(), args.config.resolve(),
                args.checkpoint.resolve(), args.jobs)
    if args.step in ('all', 'score'):
        from qlf_stage3.bstar.models import score
        score(out, args.checkpoint.resolve())
    if args.step in ('all', 'gray-qc'):
        from qlf_stage3.bstar.gray_audit import audit
        audit(out, args.crown_run.resolve(), args.study_data.resolve())


if __name__ == '__main__':
    main()
