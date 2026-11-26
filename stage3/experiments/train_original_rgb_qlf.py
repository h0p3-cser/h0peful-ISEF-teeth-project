"""Train original RGB, QLF and original RGB + QLF with a camera-source audit."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))

from qlf_stage3.modeling.original_rgb import encode,fit,inspect_rgb_folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,default=PACKAGE/'runs/study/evaluation')
    parser.add_argument('--features',type=Path,default=PACKAGE/'runs/study/measurements/training-features-255-thr30.csv')
    parser.add_argument('--capture-root',type=Path,default=Path('data/acquisition_archive'),
                        help='Collection containing 00_originals and 02_metadata')
    parser.add_argument('--intake',type=Path,help='Acquisition manifest with input and camera digests')
    parser.add_argument('--extra-originals',type=Path,action='append',default=[],
                        help='Additional original JPEG, directory or ZIP, matched by source SHA-256')
    incoming=parser.add_mutually_exclusive_group()
    incoming.add_argument('--inspect-rgb',type=Path,help='Audit this RGB folder. Only verified originals enter training.')
    incoming.add_argument('--as-received-rgb',type=Path,help='For a full comparison, fill missing originals with same-ID, byte-verified delivered JPEGs from this folder, without color correction.')
    parser.add_argument('--config',type=Path,default=PACKAGE/'src/qlf_stage3/config/protocol.json')
    parser.add_argument('--checkpoint',type=Path,default=PACKAGE/'data/study/weights/resnet18-f37072fd.pth')
    parser.add_argument('--step',choices=['all','encode','fit'],default='all')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.step=='fit' and not args.out:
        parser.error('Use the output directory from the encoding step')
    prefix='endpoint3_received_rgb_qlf_' if args.as_received_rgb else 'endpoint3_original_rgb_qlf_'
    out=(args.out or PACKAGE/'runs'/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    print(out,flush=True)
    root=args.capture_root.resolve()
    intake=(args.intake or root/'02_metadata/collection_v18_S2_C04_20261124/image_manifest.csv').resolve()
    if args.step in ('all','encode'):
        offered=args.inspect_rgb or args.as_received_rgb
        sources=args.extra_originals+([offered] if offered else [])
        encode(out,args.history.resolve(),args.features.resolve(),root,intake,args.config.resolve(),
               args.checkpoint.resolve(),[p.resolve() for p in sources],args.as_received_rgb.resolve() if args.as_received_rgb else None)
    if args.inspect_rgb:
        inspect_rgb_folder(out,args.inspect_rgb.resolve(),intake)
    if args.step in ('all','fit'):
        fit(out)


if __name__=='__main__':
    main()
