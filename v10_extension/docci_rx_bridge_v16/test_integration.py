"""End-to-end STORAGE/verification/collection fixtures. NO neural performance."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import PROJECT, FOLDER, load, atomic, sha, read, seal, check_source, PARENTS, DOCCI_SHA256, batches, geometry, parameter_groups
from docci_rx_bridge_v16.run import snapshot
from docci_rx_bridge_v16.report import verify
from docci_rx_bridge_v16.collect import collect
from docci_rx_bridge_v16.cache import write_npz, Dataset
from docci_rx_bridge_v16.test_report import row as synthetic_row


def jsonlines(path, values):
    with path.open('x', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def fixture(base):
    """Fabricated tensors, source and receipts, explicitly TEST_ONLY."""
    c = load(smoke=True)
    cache = base/'cache'
    cache.mkdir()
    (cache/'text').mkdir()
    (cache/'image').mkdir()
    (cache/'images').mkdir()
    snapshot(cache, check_source())
    atomic(cache/'identity.json', dict(action='cache', smoke=True, TEST_ONLY=True,
           source_manifest_sha256=sha(FOLDER/'MANIFEST.json')))
    atomic(cache/'config.json', c)
    atomic(cache/'source_identity.json', dict(annotation_sha256=DOCCI_SHA256,
           official_counts=dict(train=9647, test=5000, qual_dev=100, qual_test=100), TEST_ONLY=True))
    atomic(cache/'parents.json', {k: dict(sha256=v['sha256']) for k, v in PARENTS.items()})
    for task in ('text', 'image'):
        atomic(cache/(task+'_frozen_audit.json'), dict(before='SYNTHETIC', after='SYNTHETIC', unchanged=True))
    entries, baseline = [], []
    ids = np.asarray([[0, 4, 5, 6, 2]+[1]*59]+[[0, 2]+[1]*62 for _ in range(8)], np.int64)
    for i in range(10):
        eid = ('train_' if i < 8 else 'qual_dev_')+str(i)
        r = synthetic_row(eid)
        r.update(official_split='train' if i < 8 else 'qual_dev',
                 source_text_sha256=hashlib.sha256(r['source_text'].encode()).hexdigest())
        text_file, image_file = 'text/'+eid+'.npz', 'image/'+eid+'.npz'
        text_sha = write_npz(cache/text_file, dict(ids=ids, memory=np.zeros((9, 32, 768), np.float32),
                    gate=np.asarray([.99]+[.001]*8, np.float32), wire=np.ones(18432, np.float32)))
        image_sha = write_npz(cache/image_file, dict(wire=np.ones(8192, np.float32),
                    rx_i0=np.zeros((196, 96), np.float32), rx_clip=np.zeros((196, 768), np.float32),
                    oracle_clip=np.zeros((196, 768), np.float32)))
        entries.append(dict(example_id=eid, official_split=r['official_split'], text=r['source_text'],
                text_sha256=r['source_text_sha256'], source_tokens=5,
                text_file=text_file, text_file_sha256=text_sha, image_file=image_file,
                image_file_sha256=image_sha, source_image_sha256='SYNTHETIC'))
        baseline.append(r)
    atomic(cache/'index.json', dict(records=entries))
    jsonlines(cache/'baseline.jsonl', baseline)
    seal(cache)
    atomic(cache/'VERIFIED.json', verify(cache))
    root = base/'oracle'
    root.mkdir()
    snapshot(root, check_source())
    atomic(root/'config.json', c)
    atomic(root/'identity.json', dict(action='smoke_train', smoke=True, cache=str(cache),
           cache_index_sha256=sha(cache/'index.json'), arms=['self', 'oracle_clip', 'rx_clip', 'rx_i0'], TEST_ONLY=True,
           source_manifest_sha256=sha(FOLDER/'MANIFEST.json')))
    atomic(root/'original_pipeline_audit.json', dict(before='SYNTHETIC', after='SYNTHETIC', frozen_codec_and_BART_unchanged=True))
    training = entries[:8]
    for arm in ('self', 'oracle_clip', 'rx_clip', 'rx_i0'):
        folder = root/arm
        folder.mkdir()
        atomic(folder/'trainability.json', dict(trainable_parameters=sum(parameter_groups(c).values()), alignment_updates=2,
               ce_updates=3, training_ids=[r['example_id'] for r in training]))
        history = []
        for k, (epoch, ix) in enumerate(list(batches(8, 1, 1, c['seed']))[:3]):
            lr = c['ce_lr']*(.1+.9*(1-k/3))
            history.append(dict(update=k+1, total_updates=3, epoch=epoch+1,
                    batch_ids=[training[i]['example_id'] for i in ix], valid_target_tokens=21,
                    lr=lr, loss=1., gradient_norm=1.))
        jsonlines(folder/'history.jsonl', history)
        jsonlines(folder/'alignment.jsonl', [dict(update=k+1,
                   batch_ids=[training[i]['example_id'] for i in ix], loss=1.)
                   for k, (_, ix) in enumerate(batches(8, 4, 1, c['seed']))])
        atomic(folder/'learning_audit.json', dict(passed=True, groups={g: dict(parameters=n,
               changed=True, nonzero_gradient=True) for g, n in parameter_groups(c).items()}))
        atomic(folder/'frozen_audit.json', dict(before='SYNTHETIC', after='SYNTHETIC', unchanged=True))
        with (folder/'last.pth').open('xb') as f:
            f.write(b'TEST_ONLY_NOT_A_NEURAL_CHECKPOINT')
        atomic(folder/'TRAIN_COMPLETE.json', dict(checkpoint_sha256=sha(folder/'last.pth'),
               updates=3, alignment_updates=2, actual_disk_roundtrip=True, fixed_final=True, TEST_ONLY=True))
        for split in ('train', 'qual_dev'):
            subset = [r for r in entries if r['official_split'] == split]
            ev = folder/split
            ev.mkdir()
            atomic(ev/'matrix.json', dict(ids=[r['example_id'] for r in subset],
                   controls=['matched', 'shuffled', 'zero', 'off'], arm=arm, geometry=geometry(arm)))
            rows = []
            for i, entry in enumerate(subset):
                for control in ('matched', 'shuffled', 'zero', 'off'):
                    r = synthetic_row(entry['example_id'], control)
                    donor = subset[(i+1) % len(subset)] if control == 'shuffled' else entry
                    r.update(arm=arm, official_split=split, source_text_sha256=entry['text_sha256'],
                             oracle=arm == 'oracle_clip', own_text_cache_sha256=entry['text_file_sha256'],
                             auxiliary_cache_sha256=donor[('text' if arm == 'self' else 'image')+'_file_sha256'],
                             auxiliary_donor_id=donor['example_id'] if control == 'shuffled' else None)
                    rows.append(r)
            jsonlines(ev/'rows.jsonl', rows)
    seal(root)
    return root, cache


class StorageIntegration(unittest.TestCase):
    def test_full_CPU_verifier_and_real_archive_restore(self):
        with tempfile.TemporaryDirectory(prefix='v16_fixture_', dir=str(PROJECT/'releases')) as d:
            root, cache = fixture(Path(d))
            before = {p: sha(p) for p in root.rglob('*') if p.is_file()}
            result = verify(root)
            self.assertFalse(result['full_semantic_success'])
            self.assertFalse(result['comparisons']['oracle_clip']['numerical_screen_pass'])
            out = Path(d)/'records.tar.gz'
            collect(root, out)
            self.assertTrue(all(sha(p) == digest for p, digest in before.items()))
            restored = Path(d)/'restored'
            with tarfile.open(out) as archive:
                for member in archive.getmembers():
                    self.assertTrue(member.isfile())
                    self.assertFalse(Path(member.name).is_absolute())
                    self.assertNotIn('..', Path(member.name).parts)
                    self.assertNotIn(Path(member.name).suffix, ('.pth', '.npz', '.bin'))
                    target = restored/member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as f:
                        f.write(archive.extractfile(member).read())
            light = restored/'v16_records'
            for name, digest in read(light/'ARCHIVE_MANIFEST.json').items():
                self.assertEqual(sha(light/name), digest)
            second = verify(light/'records', arrays=False)
            self.assertFalse(second['arrays_verified'])
            self.assertEqual(result['scores'], second['scores'])
            with self.assertRaises(FileExistsError):
                collect(root, out)

    def test_tampered_cache_and_bounded_LRU(self):
        with tempfile.TemporaryDirectory(prefix='v16_fixture_', dir=str(PROJECT/'releases')) as d:
            root, cache = fixture(Path(d))
            dataset = Dataset(cache, 'train', limit=2)
            for i in range(8):
                dataset[i]
            self.assertEqual(len(dataset.cached), 2)
            target = cache/dataset.entries[0]['text_file']
            with target.open('ab') as f:
                f.write(b'SYNTHETIC_CORRUPTION')
            with self.assertRaises(ValueError):
                Dataset(cache, 'train')[0]
            with self.assertRaises(ValueError):
                verify(root)

    def test_resealed_wrong_update_and_self_gate_rejected(self):
        with tempfile.TemporaryDirectory(prefix='v16_fixture_', dir=str(PROJECT/'releases')) as d:
            root, _ = fixture(Path(d))
            p = root/'self/history.jsonl'
            history = [json.loads(x) for x in p.read_text().splitlines()]
            history[0]['valid_target_tokens'] = 1
            p.unlink()
            jsonlines(p, history)
            seal(root)
            with self.assertRaises(ValueError):
                verify(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
