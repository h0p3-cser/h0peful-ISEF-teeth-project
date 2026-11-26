"""Train the three endpoint-3 models on capture-original JPEGs and fixed QLF measurements."""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, is_zipfile
import json

import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from ..files import PACKAGE, dump, load, relative_path, sha
from ..rgb.color import RAW_MODE
from .cohort import FEATURES, IDS, MODELS, STATES, ensure, save_csv
from .evaluation import evaluate, read_result
from .report import figures
from .training import fold_predictions, group_features
from .validation import verify_predictions
from .vision import imagenet18, white_input, white_vectors

HISTORY_FIELDS = ['rgb_image_id','qlf_image_id','rgb_catalog_tooth','rgb_sha256','rgb_event','qlf_event',
                  'group_id','group_frames','test_fold','session_id','geometry_version','fit_source']
FOLD_FIELDS = ['fold','held_tooth','group_id','tooth_id','state_id','role']


def find_supplements(roots, missing, out):
    """Identify candidate originals by the acquisition digest, including shared repeat slots."""
    found = {}
    for root in roots:
        if not missing.difference(found):
            break
        if root.is_file() and is_zipfile(root):
            with ZipFile(root) as bundle:
                for item in bundle.infolist():
                    if Path(item.filename).suffix.lower() not in ('.jpg','.jpeg'):
                        continue
                    data = bundle.read(item)
                    digest = sha256(data).hexdigest()
                    if digest in missing and digest not in found:
                        target = out/'recovered_originals'/(digest+'.jpg')
                        target.parent.mkdir(exist_ok=True)
                        target.write_bytes(data)
                        found[digest] = (target, str(root)+'!'+item.filename)
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        items = root.rglob('*') if root.is_dir() else [root]
        for path in items:
            if path.suffix.lower() not in ('.jpg','.jpeg') or not path.is_file():
                continue
            digest = sha(path)
            if digest in missing:
                found.setdefault(digest,(path,str(path)))
    return found


def inspect_rgb_folder(out, folder, intake_path):
    """Compare each identifiable RGB file with both the intake copy and its camera digest."""
    intake = pd.read_csv(intake_path,keep_default_na=False,dtype=str).set_index('image_id',verify_integrity=True)
    records = []
    for path in sorted(folder.rglob('*')):
        if path.suffix.lower() not in ('.jpg','.jpeg') or path.name.startswith('._'):
            continue
        image_id = path.stem.rsplit('_',1)[-1]
        if image_id not in intake.index or intake.at[image_id,'modality']!='RGB':
            continue
        row = intake.loc[image_id]
        actual = sha(path)
        expected = row.source_camera_sha256
        if not expected and row.source_storage in ('filesystem','copy_of_archived_image') and row.pixel_processing=='as_in_previous_collection':
            expected = row.sha256
        records.append(dict(rgb_image_id=image_id,tooth_id=row.tooth_id,state_id=row.state_id,
            session_id=row.session_id,angle_id=row.angle_id,incoming_path=path.relative_to(folder).as_posix(),
            incoming_sha256=actual,original_sha256=expected,delivered_sha256=row.sha256,
            matches_original=actual==expected if expected else None,matches_delivered=actual==row.sha256,
            upstream_processing=row.pixel_processing,source_camera_filename=row.source_filename))
    ensure(records, 'No RGB image IDs from the acquisition manifest found in '+str(folder))
    audit = pd.DataFrame(records)
    ensure(audit.rgb_image_id.is_unique, 'Multiple supplied RGB files claim the same image ID')
    save_csv(out,'incoming_rgb_comparison.csv',audit)
    dump(out/'incoming_rgb_folder.json',dict(folder=str(folder),compared_files=len(audit),
         matched_original_files=int(audit.matches_original.eq(True).sum()),
         matched_delivered_files=int(audit.matches_delivered.sum()),
         comparison_sha256=sha(out/'incoming_rgb_comparison.csv')))
    return audit


def bind(out, history, feature_csv, archive, intake_path, supplementary, received_folder=None):
    history_frame = read_result(history,'sample_manifest.csv')
    history_frame = history_frame.loc[history_frame.included]
    formal = pd.read_csv(feature_csv,float_precision='round_trip')
    ensure(formal.sample_id.is_unique and history_frame.rgb_image_id.is_unique,'Nonunique frame identity')
    frames = history_frame[IDS+HISTORY_FIELDS].merge(formal[IDS+FEATURES],on=IDS,how='left',
                    validate='one_to_one',indicator=True).sort_values('sample_id').reset_index(drop=True)
    ensure(frames['_merge'].eq('both').all() and np.isfinite(frames[FEATURES].values).all()
           and frames.loss_area_gt30_pct.between(0,100).all(),'Missing or invalid paired QLF measurement')
    frames = frames.drop(columns='_merge').rename(columns={'rgb_sha256':'historical_rgb_sha256',
                                                         'rgb_path':'historical_pair_rgb_path'})
    catalog = pd.read_csv(intake_path,keep_default_na=False,dtype=str).set_index('image_id',verify_integrity=True)
    old_groups = read_result(history,'groups.csv').set_index('group_id',verify_integrity=True)
    ensure(set(frames.group_id)==set(old_groups.index),'Unexpected source cohort')
    for key,block in frames.groupby('group_id'):
        ensure(sorted(block.sample_id)==sorted(json.loads(old_groups.at[key,'member_ids'])),
               'Historical member list differs: '+key)
    evidence, originals = [], {}
    for r in frames.itertuples():
        item = catalog.loc[r.rgb_image_id]
        ensure(item.modality=='RGB' and item.state_id==r.state_id and item.angle_id==r.angle_id and
               item.session_id==r.session_id and item.tooth_id==r.rgb_catalog_tooth,
               'Photograph does not match the historical capture: '+r.rgb_image_id)
        ensure(item.sha256==r.historical_rgb_sha256, 'Different catalog release for '+r.rgb_image_id)
        camera = item.source_camera_sha256
        if camera:
            expected, basis = camera, 'recorded_camera_sha256'
        else:
            ensure(item.source_storage in ('filesystem','copy_of_archived_image') and
                   item.pixel_processing=='as_in_previous_collection',
                   'No capture-original digest for '+r.rgb_image_id)
            expected, basis = item.sha256, 'original_intake_backup_sha256'
        source = archive/item.original_backup_relative_path
        observed = sha(source) if source.is_file() else ''
        if observed == expected:
            originals.setdefault(expected,(source,str(source)))
        evidence.append(dict(rgb_image_id=r.rgb_image_id,expected_original_sha256=expected,digest_basis=basis,
            intake_storage=item.source_storage,upstream_processing=item.pixel_processing,
            registered_backup=item.original_backup_relative_path,registered_backup_sha256=observed,
            duplicate_of=item.duplicate_of,source_camera_filename=item.source_filename))
    needed = {a['expected_original_sha256'] for a in evidence}.difference(originals)
    recovered = find_supplements(supplementary,needed,out)
    delivered = {}
    if received_folder:
        supplied = inspect_rgb_folder(out,received_folder,intake_path)
        delivered = {r.rgb_image_id:(received_folder/r.incoming_path, r.incoming_sha256)
                     for r in supplied.itertuples() if r.matches_delivered}
    for entry in evidence:
        digest = entry['expected_original_sha256']
        source = originals.get(digest) or recovered.get(digest)
        basis = 'verified_backup' if digest in originals else 'verified_supplement'
        if source is None and entry['rgb_image_id'] in delivered:
            path,digest = delivered[entry['rgb_image_id']]
            source,basis = (path,str(path)),'supplied_delivery_copy'
        usable = source is not None
        entry.update(included=usable,original_found_in=basis if usable else '',
                     matches_original=usable and digest==entry['expected_original_sha256'],rgb_path='',rgb_resolved='',rgb_sha256='',
                     source_container='',exclusion_reason='' if usable else 'verified_original_unavailable')
        if usable:
            path,container = source
            with Image.open(path) as im:
                ensure(im.format=='JPEG' and im.mode=='RGB','Expected a camera RGB JPEG: '+str(path))
            relative = relative_path(path,out)
            entry.update(rgb_path=relative,rgb_resolved=relative,rgb_sha256=digest,source_container=container)
    frames = frames.merge(pd.DataFrame(evidence),on='rgb_image_id',validate='one_to_one')
    audit_fields = ['sample_id','rgb_image_id','tooth_id','state_id','angle_id','repeat_rank','group_id',
        'expected_original_sha256','historical_rgb_sha256','digest_basis','upstream_processing','intake_storage',
        'registered_backup','registered_backup_sha256','source_camera_filename','duplicate_of','original_found_in',
        'rgb_resolved','rgb_sha256','source_container','matches_original','included','exclusion_reason']
    save_csv(out,'original_source_audit.csv',frames[audit_fields])
    save_csv(out,'missing_originals.csv',frames.loc[~frames.matches_original,audit_fields])
    return frames


def encode(out, history, feature_csv, archive, intake_path, config_path, weights, supplementary, received_folder=None):
    out.mkdir(parents=True,exist_ok=True)
    ensure(not any(out.iterdir()),'Choose a new run directory')
    config = load(config_path)
    config.setdefault('csv_sha256', load(PACKAGE/'config/endpoint-reference.json')['features_sha256'])
    ensure(config['csv_sha256']==sha(feature_csv) and config['states']==STATES and
           config['feature_columns']==FEATURES,'The fixed QLF source or protocol differs')
    frames = bind(out,history,feature_csv,archive,intake_path,supplementary,received_folder)
    if received_folder:
        ensure(frames.included.all(), 'To run the full delivered cohort, every input must match its acquisition or same-ID delivery digest')
    active = frames[frames.included].copy()
    ensure(active.tooth_id.nunique()==9 and set(active.state_id)==set(STATES),'Too few original inputs for nine folds')
    ensure(active.groupby('rgb_sha256').tooth_id.nunique().max()==1,'Same source photograph occurs across teeth')
    test_fold={t:i for i,t in enumerate(sorted(active.tooth_id.unique()),1)}
    ensure(frames.test_fold.eq(frames.tooth_id.map(test_fold)).all(),'Historical held-tooth assignment differs')
    keep = read_result(history,'folds.csv')
    keep = keep[keep.group_id.isin(active.group_id)]
    for fold,part in keep.groupby('fold'):
        held=part.held_tooth.unique()
        ensure(len(held)==1 and test_fold[held[0]]==fold and part.group_id.is_unique and
               set(part.group_id)==set(active.group_id) and (part.role.eq('test')==part.tooth_id.eq(held[0])).all(),
               'Invalid historical test fold')
    save_csv(out,'expected_folds.csv',keep)
    net,checkpoint,_ = imagenet18(dict(config,checkpoint=str(weights)))
    paths = [out/p for p in active.rgb_resolved]
    print(f'{len(active)}/{len(frames)} JPEGs, {active.matches_original.sum()} verified originals, '
          f'{active.group_id.nunique()} groups. Encoding RGB.',flush=True)
    vectors = white_vectors(paths,net,RAW_MODE)
    keys = [sha256('|'.join([d,RAW_MODE,config['weights_sha256']]).encode()).hexdigest() for d in active.rgb_sha256]
    frames['embedding_cache_key']=''
    frames.loc[active.index,'embedding_cache_key']=keys
    active_counts = active.groupby('group_id').size()
    frames['historical_group_frames']=frames.group_frames
    frames['group_frames']=frames.group_id.map(active_counts).fillna(0).astype(int)
    save_csv(out,'sample_manifest.csv',frames)
    np.savez_compressed(out/'frozen_frame_features.npz',sample_ids=active.sample_id.to_numpy(str),
                        cache_keys=np.asarray(keys),white=vectors)
    for path,digest in zip(paths,active.rgb_sha256):
        ensure(sha(path)==digest,'Original changed during encoding: '+str(path))
    policy = 'verified_or_supplied_delivery' if received_folder else 'verified_camera_original'
    dump(out/'run_config.json',dict(config,white_light_mode=RAW_MODE,white_input_policy=policy,
         qlf_workflow='endpoint3_original_or_delivered_rgb_qlf255_thr30',feature_csv=relative_path(feature_csv,out),
         history=relative_path(history,out),original_archive=relative_path(archive,out),
         intake=relative_path(intake_path,out),supplements=[relative_path(s,out) for s in supplementary],
         received_folder=relative_path(received_folder,out) if received_folder else None,
         checkpoint=relative_path(checkpoint,out),intake_sha256=sha(intake_path),
         history_manifest_sha256=sha(history/'sample_manifest.csv'),fold_source_sha256=sha(history/'folds.csv')))
    dump(out/'encoded.json',dict(target_frames=len(frames),included_frames=len(active),
         groups=active.group_id.nunique(),teeth=active.tooth_id.nunique(),training_color_corrections=0,
         verified_originals=int(active.matches_original.sum()),delivery_version_inputs=int((~active.matches_original).sum()),
         sha256={n:sha(out/n) for n in ['original_source_audit.csv','missing_originals.csv','sample_manifest.csv',
                       'frozen_frame_features.npz','expected_folds.csv','run_config.json']}))


def fit(out):
    ensure(not (out/'run_summary.json').exists(),'This run has already been reported')
    provenance=load(out/'encoded.json')
    for name,digest in provenance['sha256'].items():
        ensure(sha(out/name)==digest,'Changed original-RGB input: '+name)
    cfg=load(out/'run_config.json')
    cfg['feature_csv']=(out/cfg['feature_csv']).resolve()
    ensure(sha(cfg['feature_csv'])==cfg['csv_sha256'],'Changed QLF measurements')
    audit=read_result(out,'sample_manifest.csv')
    active=audit.loc[audit.included].reset_index(drop=True)
    with np.load(out/'frozen_frame_features.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sample_ids'],active.sample_id)
        np.testing.assert_array_equal(data['cache_keys'],active.embedding_cache_key)
        white=data['white']
    groups,visual,qlf=group_features(active,white)
    save_csv(out,'groups.csv',groups)
    print('Fitting fresh visual, QLF and fused heads in nine held-tooth folds.',flush=True)
    with threadpool_limits(limits=1,user_api='blas'):
        fold_predictions(cfg,out,groups,visual,qlf,active,white)
    order=['fold','group_id']
    actual,expected=[read_result(out,name)[FOLD_FIELDS].sort_values(order).reset_index(drop=True)
                     for name in ('folds.csv','expected_folds.csv')]
    pd.testing.assert_frame_equal(actual,expected)
    dump(out/'oof_sha256.json',{n:sha(out/n) for n in ['oof_groups.csv','oof_frames.csv']})
    results=evaluate(cfg,out)
    verified=verify_predictions(out,cfg['feature_csv'])
    dump(out/'validation.json',verified)
    rgb_label='交付 RGB' if active.original_found_in.eq('supplied_delivery_copy').any() else '原始 RGB'
    figures(out,results,out,names={'M1':'M1 '+rgb_label,'M2':'M2 QLF','M3':'M3 '+rgb_label+'＋QLF'},
            photo_input=lambda p:white_input(p,RAW_MODE))
    from .original_rgb_report import report
    report(out,audit,results)
    dump(out/'run_summary.json',dict(finished_at=datetime.now().astimezone().isoformat(timespec='seconds'),
         training_completed=True,full_requested_cohort=bool(audit.included.all()),target_frames=len(audit),
         training_frames=len(active),excluded_frames=int((~audit.included).sum()),groups=len(groups),
         held_teeth=active.tooth_id.nunique(),three_stage_teeth=int(results['summary'].evaluable_teeth.iloc[0]),
         training_color_corrections=0,verified_originals=int(active.rgb_sha256.eq(active.expected_original_sha256).sum()),
         delivery_version_inputs=int(active.original_found_in.eq('supplied_delivery_copy').sum()),
         source_review='original_source_audit.csv',replay=verified,
         outputs={n:sha(out/n) for n in ['model_scores.csv','oof_groups.csv','paired_group_predictions.csv','report.md']}))
    print(read_result(out,'model_scores.csv').to_string(index=False),flush=True)
