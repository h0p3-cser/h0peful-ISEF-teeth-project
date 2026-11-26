"""Public research pipeline commands. All paths are supplied by the caller."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .files import FEATURE_CSV, PROTOCOL, config, dataset_info, dump, load, outside_inputs


COMMANDS = {
    'endpoint': 'Run the complete endpoint: QLF measurement, export, gray/RGB training, paired reports and inference models',
    'stability': 'Run or resume both white-light branches and their three-seed evaluation from verified features',
    'patch-huber-pilot': 'Prepare, freeze, run, or verify the version 1.0 three-view physical-patch experiment',
    'material-pilot': 'Reproduce the archived pre-protocol material study',
    'verify-material-pilot': 'Verify the archived pre-protocol material study',
    'calibrate-rgb': 'Calibrate paired white-light photographs and evaluate held-out root colour',
    'verify-rgb': 'Replay colour, geometry, model inputs and paired calibration QC',
    'prepare-data': 'Copy the reviewed study inputs into a portable private dataset',
    'demo-data': 'Generate the public synthetic dataset',
    'check-data': 'Verify every file recorded in the portable input manifest',
    'run': 'Measure QLF, export four features, train and verify the three models',
    'measure': 'Compute signed 255 loss maps and threshold curves',
    'export': 'Export the paired four-feature CSV at fixed 30%',
    'train': 'Run the nine leave-one-tooth-out folds and write the inference bundle',
    'predict': 'Predict S0/S1/S2 for unlabeled acquisitions',
    'verify': 'Verify the saved folds, models, probabilities and primary metrics',
    'compare': 'Compare predictions and partitions from two complete runs',
}


def child(*args):
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'qlf_stage3', *map(str,args)], check=True)


def evaluation_dir(path):
    path = path.resolve()
    return path / 'evaluation' if (path / 'pipeline.json').is_file() else path


def stability_dir(path):
    path = path.resolve()
    for directory in (path / 'stability', path):
        if (directory / 'plan.json').is_file():
            return directory
    return None


def compare_outputs(first, second, tolerance):
    from .modeling.training import compare_runs

    left, right = stability_dir(first), stability_dir(second)
    if left is None and right is None:
        return compare_runs(evaluation_dir(first), evaluation_dir(second), tolerance)
    if left is None or right is None:
        raise ValueError('Select two complete endpoints or two single-run evaluations')
    a, b = load(left / 'plan.json'), load(right / 'plan.json')
    if (a['branches'] != b['branches'] or set(a['training_seeds']) != set(b['training_seeds'])
            or a['teeth'] != b['teeth'] or a['paired_frames'] != b['paired_frames'] or a['groups'] != b['groups']):
        raise ValueError('The two endpoints use different cohorts, input modes or training seeds')
    checks = {}
    for branch in a['branches']:
        for seed in a['training_seeds']:
            name = f'{branch}/seed_{seed}'
            checks[name] = compare_runs(left / name, right / name, tolerance)
    return dict(equal_categories=True, compared_runs=len(checks), runs=checks, tolerance=tolerance)


def run_all(data, out, weights, *, endpoint=False, seeds=(42,123,2024)):
    out = outside_inputs(out, data)
    out.mkdir(parents=True, exist_ok=False)
    args = ['--data', data]
    measurements = out / 'measurements'
    current = 'measure'
    try:
        dump(out / 'pipeline.json', dict(status='running', step=current))
        child('measure', *args, '--out', measurements)
        current = 'export'
        dump(out / 'pipeline.json', dict(status='running', step=current))
        child('export', *args, '--measurements', measurements)
        if endpoint:
            current = 'stability'
            dump(out / 'pipeline.json', dict(status='running', workflow='endpoint', step=current))
            extra = ['--weights', weights] if weights else []
            child('stability', *args, '--features', measurements / FEATURE_CSV, '--out', out / 'stability',
                  '--seeds', *seeds, *extra)
            summary = load(out / 'stability' / 'verification.json')
            dump(out / 'pipeline.json', dict(status='complete', workflow='endpoint', features='measurements/' + FEATURE_CSV,
                 report='stability/report.md', models={b:f'stability/models/{b}.joblib' for b in ('gray','rgb')},
                 validation=summary))
            print(out / 'stability' / 'report.md')
            return
        current = 'train'
        dump(out / 'pipeline.json', dict(status='running', step=current))
        extra = ['--weights', weights] if weights else []
        child('train', *args, '--features', measurements / FEATURE_CSV, '--out', out / 'evaluation', *extra)
        summary = load(out / 'evaluation' / 'validation.json')
        dump(out / 'pipeline.json', dict(status='complete', features='measurements/' + FEATURE_CSV,
             report='evaluation/report.md', model='evaluation/models/final_bundle.joblib', validation=summary))
        print(out / 'evaluation' / 'report.md')
    except Exception as exc:
        dump(out / 'pipeline.json', dict(status='failed', step=current, error=str(exc)))
        raise


def command(name, args):
    parser = argparse.ArgumentParser(prog='qlf-stage3 ' + name, description=COMMANDS[name])
    if name == 'stability':
        from .stability_cli import main as stability
        return stability(args)
    if name=='patch-huber-pilot':
        from .patch_pilot.cli import main as patch_pilot
        patch_pilot(args)
        return 0
    if name in ('material-pilot','verify-material-pilot'):
        from .materials.pilot import PROTOCOL as MATERIAL_PROTOCOL, REGIONS, run as material_run
        parser.add_argument('--data',type=Path,required=True)
        parser.add_argument('--study-run',type=Path,required=True)
        parser.add_argument('--root-run',type=Path,required=True,help='Completed historical root RGB run')
        if name=='material-pilot':
            parser.add_argument('--out',type=Path,required=True,help='A new output directory')
            parser.add_argument('--protocol',type=Path,default=MATERIAL_PROTOCOL)
            parser.add_argument('--regions',type=Path,default=REGIONS)
            opt=parser.parse_args(args)
            print(material_run(opt.data,opt.study_run,opt.root_run,opt.out,opt.protocol,opt.regions))
        else:
            from .materials.verify import verify as material_verify
            parser.add_argument('--run',type=Path,required=True)
            opt=parser.parse_args(args)
            print(json.dumps(material_verify(opt.data,opt.study_run,opt.root_run,opt.run),ensure_ascii=False,indent=2))
        return 0
    if name in ('calibrate-rgb','verify-rgb'):
        from .rgb.calibration import DEFAULT_CONFIG, DEFAULT_REGIONS, run
        parser.add_argument('--data',type=Path,required=True)
        parser.add_argument('--study-run',type=Path,required=True)
        if name == 'calibrate-rgb':
            parser.add_argument('--scope',choices=['pilot','full'],required=True)
            parser.add_argument('--out',type=Path,required=True)
            parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
            parser.add_argument('--regions',type=Path,default=DEFAULT_REGIONS)
            parser.add_argument('--workers',type=int,default=1)
            opt=parser.parse_args(args)
            try:
                run(opt.data,opt.study_run,opt.out,opt.scope,opt.config,opt.regions,opt.workers)
            except Exception as exc:
                status=opt.out/"status.json"
                if status.is_file() and load(status).get("status")=="running" and not isinstance(exc, FileExistsError):
                    dump(status,dict(status="failed",error=str(exc)))
                raise
        else:
            from .rgb.verify import verify
            parser.add_argument('--run',type=Path,required=True)
            parser.add_argument('--compare',type=Path)
            parser.add_argument('--encoder-smoke',action='store_true',help='Read one pair through the frozen encoder')
            opt=parser.parse_args(args)
            print(json.dumps(verify(opt.run,opt.data,opt.study_run,opt.compare,opt.encoder_smoke),indent=2))
        return 0
    if name in ('check-data', 'measure', 'export', 'train', 'run', 'endpoint', 'verify', 'predict'):
        parser.add_argument('--data', type=Path, required=name not in ('verify','predict'), help='Portable dataset directory')
    if name in ('prepare-data', 'demo-data', 'measure', 'train', 'run', 'endpoint', 'predict'):
        parser.add_argument('--out', required=True, type=Path, help='A new output directory')
    if name in ('prepare-data', 'train', 'run', 'endpoint', 'predict'):
        parser.add_argument('--weights', type=Path, help='Local ImageNet V1 checkpoint for offline use')
    if name == 'endpoint':
        parser.add_argument('--seeds', type=int, nargs=3, default=[42,123,2024], help='Three distinct training seeds')
    if name == 'prepare-data':
        parser.add_argument('--workspace', type=Path, required=True, help='Reviewed workspace containing Data/')
        parser.add_argument('--catalog', type=Path, required=True, help='Capture catalog containing images.csv and pairs.csv')
    if name == 'export':
        parser.add_argument('--measurements', required=True, type=Path)
        parser.add_argument('--stage3-features', type=Path, help='Existing upstream per-image feature CSV')
        parser.add_argument('--single-image-features', type=Path, help='Optional additional upstream single-image table')
    if name in ('train', 'verify'):
        parser.add_argument('--features', type=Path, required=name == 'train', help='Four-feature CSV beside frames and export audit')
    if name == 'predict':
        parser.add_argument('--model', type=Path, required=True, help='Trusted local model bundle')
        parser.add_argument('--input', type=Path, required=True, help='Unlabeled per-frame CSV with capture_id')
        parser.add_argument('--arm', choices=['all','M1','M2','M3'], default='all')
    if name == 'compare':
        parser.add_argument('first', type=Path)
        parser.add_argument('second', type=Path)
        parser.add_argument('--atol', type=float, default=1e-6)
    if name == 'verify':
        parser.add_argument('--run', type=Path, required=True)
        parser.add_argument('--reference-study', action='store_true', help='Also compare to the original nine-tooth study')
    opt = parser.parse_args(args)
    if name == 'prepare-data':
        from .bundle import prepare_data
        print(prepare_data(opt.workspace, opt.catalog, opt.out, opt.weights))
    elif name == 'demo-data':
        from .demo import create_demo
        print(create_demo(opt.out))
    elif name == 'check-data':
        from .bundle import check_data
        print(json.dumps(check_data(opt.data), indent=2))
    elif name == 'compare':
        print(json.dumps(compare_outputs(opt.first, opt.second, opt.atol), indent=2))
    elif name == 'verify':
        directory = stability_dir(opt.run)
        if directory is not None:
            from .modeling.endpoint_validation import verify_stability
            print(json.dumps(verify_stability(directory, opt.data, opt.features, opt.reference_study), indent=2))
            return 0
        from .modeling.validation import verify_run
        print(json.dumps(verify_run(evaluation_dir(opt.run), opt.data, opt.features, opt.reference_study), indent=2))
    elif name == 'predict':
        from .modeling.training import predict_acquisition
        c = load(PROTOCOL)
        c['checkpoint'] = opt.weights
        if opt.data:
            c['dataset'], _ = dataset_info(opt.data)
            local = c['dataset'] / 'weights' / 'resnet18-f37072fd.pth'
            if not opt.weights and local.is_file():
                c['checkpoint'] = local
        print(json.dumps(predict_acquisition(c, opt.model, opt.input, opt.out, opt.arm), indent=2))
    else:
        root, description = dataset_info(opt.data)
        workspace = root / 'workspace'
        if name == 'endpoint':
            if len(set(opt.seeds)) != 3 or any(not 0 <= s < 2**32 for s in opt.seeds):
                parser.error('Choose three different unsigned 32-bit seeds')
            run_all(root, opt.out, opt.weights, endpoint=True, seeds=opt.seeds)
        elif name == 'run':
            run_all(root, opt.out, opt.weights)
        elif name == 'train':
            from .modeling.training import run_experiment
            c = config(root, opt.features)
            local = root / 'weights' / 'resnet18-f37072fd.pth'
            c['checkpoint'] = opt.weights or (local if local.is_file() else None)
            run_experiment(c, outside_inputs(opt.out, root, opt.features.parent))
            print(opt.out.resolve() / 'report.md')
        else:
            # The upstream measurement modules bind their data directory on import.
            os.environ['QLF_DATA_ROOT'] = str(workspace)
            os.environ.setdefault('MPLBACKEND','Agg')
            if name == 'measure':
                from .fullscale255 import measure
                measure(workspace, outside_inputs(opt.out, root), description['kind'])
            else:
                from .export import export
                directory = outside_inputs(opt.measurements, root)
                result = export(directory, workspace, opt.stage3_features, opt.single_image_features)
                print(json.dumps(dict(csv=str(directory / FEATURE_CSV), frames=result['exported'],
                                      by_stage=result['exported_by_stage']), indent=2))
    return 0
