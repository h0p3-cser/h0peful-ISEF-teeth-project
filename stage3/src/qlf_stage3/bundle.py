"""Assemble an independently copyable, private research dataset."""
import csv
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from .files import PROTOCOL, SOURCE_TABLE, dataset_info, dump, load, outside_inputs, sha


def write_checksums(root):
    with (root / 'files.sha256').open('w', encoding='utf-8', newline='\n') as checks:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path != root / 'files.sha256':
                checks.write(f'{sha(path)}  {path.relative_to(root).as_posix()}\n')


def check_data(data):
    root, descriptor = dataset_info(data)
    seen = set()
    with (root / 'files.sha256').open(encoding='utf-8') as checks:
        for line in checks:
            digest, name = line.rstrip('\n').split('  ', 1)
            path = (root / name).resolve()
            if path in seen or not path.is_relative_to(root) or sha(path) != digest:
                raise ValueError('Dataset checksum mismatch or duplicate: ' + name)
            seen.add(path)
    if root / 'dataset.json' not in seen:
        raise ValueError('Missing dataset descriptor in checksum manifest')
    return dict(kind=descriptor['kind'], verified_files=len(seen))


def prepare_data(workspace, catalog, out, weights=None):
    workspace, catalog = workspace.resolve(), catalog.resolve()
    out = outside_inputs(out, workspace, catalog)
    if out.exists():
        raise FileExistsError(out)
    for path in (workspace / SOURCE_TABLE, workspace / 'stage3-author-transforms.zip',
                 workspace / 'local-photo-bindings.csv', workspace / 'Data' / 'manifest.csv',
                 catalog / 'images.csv', catalog / 'pairs.csv'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if weights and sha(weights) != load(PROTOCOL)['weights_sha256']:
        raise ValueError('The supplied offline ResNet-18 checkpoint has a different SHA-256')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.stage3-input-', dir=out.parent) as temp:
        data = Path(temp) / 'dataset'
        work = data / 'workspace'
        work.mkdir(parents=True)
        for name in ('Data', 'Data_rgb_corrected'):
            shutil.copytree(workspace / name, work / name)
        (work / SOURCE_TABLE).parent.mkdir()
        shutil.copyfile(workspace / SOURCE_TABLE, work / SOURCE_TABLE)
        shutil.copyfile(workspace / 'stage3-author-transforms.zip', work / 'stage3-author-transforms.zip')
        bindings = 'local-photo-bindings.csv'
        fields = ['workspace_path','modality','image_id','sha256']
        with (workspace / bindings).open(encoding='utf-8-sig', newline='') as source, \
             (work / bindings).open('w', encoding='utf-8-sig', newline='') as dest:
            writer = csv.DictWriter(dest, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in csv.DictReader(source):
                local = (work / row['workspace_path']).resolve()
                if not local.is_relative_to(work) or sha(local) != row['sha256']:
                    raise ValueError(f"Photo binding does not match the copied bytes: {row['image_id']}")
                writer.writerow(row)
        (data / 'catalog').mkdir()
        for name in ('images.csv','pairs.csv'):
            shutil.copyfile(catalog / name, data / 'catalog' / name)
        if weights:
            (data / 'weights').mkdir()
            shutil.copyfile(weights, data / 'weights' / 'resnet18-f37072fd.pth')
        dump(data / 'dataset.json', {'schema':1,'kind':'study','unit':'paired capture within tooth and view'})
        write_checksums(data)
        data.rename(out)
    return out
