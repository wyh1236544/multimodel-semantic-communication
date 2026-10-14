"""No-Torch schedule/source identity contracts; legacy gates remain unchanged."""
import math
from pathlib import Path
from docci_reconstruction_v10.core import (atomic_json,read,digest,safe_write,jsonlines,
    load as old_load,choose_indices,geometry,joined_ids,state_hash,gate)

FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]


def load(path=None):
    template=read(FOLDER/'config.json'); c=read(path or FOLDER/'config.json')
    if set(c)!=set(template) or c['schema']!=template['schema']: raise ValueError('New phase schema differs')
    for k,lo,hi in (('target_epochs',11,40),('probe_every',500,4000),('probe_samples',2,16),('smoke_updates',2,2)):
        if type(c[k]) is not int or not lo<=c[k]<=hi: raise ValueError('Invalid '+k)
    for k in ('phase_learning_rate','phase_min_learning_rate'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or not 0<c[k]<=.0001: raise ValueError('Unsafe '+k)
    if c['phase_min_learning_rate']>c['phase_learning_rate']: raise ValueError('LR floor exceeds start')
    return c


def schedule(c,parent_updates=10000,count=2048,batch=2):
    target=math.ceil(c['target_epochs']*count/batch)
    if not 0<parent_updates<target or target-parent_updates>40000: raise ValueError('Invalid continuation budget')
    return dict(parent_updates=parent_updates,target_updates=target,additional_updates=target-parent_updates,
        train_samples=count,batch_size=batch,additional_presentations=(target-parent_updates)*batch,
        final_cumulative_presentations=target*batch,final_epochs=target*batch/count,
        optimizer_policy='NEW AdamW; weights-only warm start, NOT exact resume',
        LR_policy='new phase linear start-to-floor, no retrospective change to old LR')


def phase_lr(c,local_update,updates):
    if not 0<=local_update<updates: raise ValueError('Update outside stage')
    return c['phase_learning_rate']+(c['phase_min_learning_rate']-c['phase_learning_rate'])*local_update/max(1,updates-1)


def check_sources():
    m=read(FOLDER/'MANIFEST.json')
    for n,sha in m['files'].items():
        if digest(FOLDER/n)!=sha: raise ValueError('Published new source changed: '+n)
    for n,sha in m['required_sources'].items():
        if digest(PROJECT/n)!=sha: raise ValueError('Pinned historical source changed: '+n)
    return m


def parent(path,weight=False):
    path=safe_write(path).resolve(strict=True); contract=read(FOLDER/'parent_contract.json')
    for n,sha in contract['files'].items():
        f=safe_write(path/n)
        if not f.is_file() or digest(f)!=sha: raise ValueError('Approved parent record differs: '+n)
    identity=read(path/'run_identity.json'); data=read(path/'data_identity.json'); c=old_load(path/'config.json')
    if identity['schema']!='docci-reconstruction-v10' or identity['arm']!='T1_chunks' or identity['mode']!='screen' or identity['status']!='QUALITY_STOP':
        raise ValueError('Wrong approved parent')
    if identity['schedule']!=dict(train_samples=2048,val_samples=100,gate_train_samples=32,updates=10000,smoke=False) or c['text_batch_size']!=2:
        raise ValueError('Parent is not approved2048/batch2/10000')
    ckpt=path/'train_clean/last.pth'; done=read(path/'train_clean/TRAIN_COMPLETE.json')
    if done['checkpoint_sha256']!=contract['checkpoint_sha256']: raise ValueError('Parent checkpoint receipt differs')
    if weight and (not ckpt.is_file() or digest(ckpt)!=contract['checkpoint_sha256']): raise ValueError('Actual parent checkpoint missing or changed')
    return dict(root=path,identity=identity,data=data,config=c,checkpoint=ckpt,checkpoint_sha256=contract['checkpoint_sha256'])


def probes(total,parent_updates,interval):
    return [parent_updates]+list(range(parent_updates+interval,total,interval))+[total]
