"""Synthetic stage/replay/render/archive negative contracts; never model-effect claims."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.common import PROJECT,load,atomic,seal,unseal,sha,lines
from docci_digital_compare_v19.experiment import digital
from docci_digital_compare_v19.report import stage,aggregate

class Storage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='_v19_TEST_ONLY_',dir=str(PROJECT/'experiment_results'))
        self.root=Path(self.temp.name);self.prepared=self.root/'prepared';self.prepared.mkdir()
        self.c=load();self.sources=[dict(example_id='qual_dev_{:05d}'.format(i),official_split='qual_dev',text='TEST ONLY: object {} is red beside a blue door.'.format(i)) for i in range(2)]
        for r in self.sources:r['text_sha256']=hashlib.sha256(r['text'].encode()).hexdigest()
        atomic(self.prepared/'sources.json',self.sources);(self.prepared/'old_rows.jsonl').write_text('')
    def tearDown(self):self.temp.cleanup()
    def make_stage(self):
        root=self.root/'digital';digital(self.prepared,root,self.c,lambda:None,True);return root
    def test_real_smoke_packet_stage_and_full_replay(self):
        r=stage(self.make_stage(),self.prepared,self.c,lambda:None,True)
        self.assertEqual(r['rows'],12);self.assertTrue(r['packet_receiver_replayed_here'])
    def test_tampered_seal_is_rejected(self):
        r=self.make_stage();p=r/'rows.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Sealed artifact'):stage(r,self.prepared,self.c,lambda:None)
    def test_coherently_resealed_false_BER_is_caught_by_receiver_replay(self):
        r=self.make_stage();p=r/'rows.jsonl';rows=lines(p);rows[-1]['pre_fec_errors']+=1
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        sealed=json.loads((r/'SEALED.json').read_text());sealed['rows.jsonl']=sha(p);(r/'SEALED.json').write_text(json.dumps(sealed))
        with self.assertRaisesRegex(ValueError,'packet replay'):stage(r,self.prepared,self.c,lambda:None,True)
    def test_derived_no_overwrite_and_manual_facts_preserved(self):
        from docci_digital_compare_v19.display import render,write_derived
        p=self.root/'new.html';write_derived(p,'same');write_derived(p,'same')
        with self.assertRaises(FileExistsError):write_derived(p,'different')
        r=self.make_stage();atomic(self.root/'config.json',self.c)
        result=dict(status='TEST_ONLY',comparison=aggregate(lines(r/'rows.jsonl'),self.c),stages={'digital':{}})
        render(self.root,result);facts=self.root/'fact_review.csv';facts.write_text('HUMAN ANNOTATIONS')
        render(self.root,result);self.assertEqual(facts.read_text(),'HUMAN ANNOTATIONS')
    def test_archive_checksum_and_duplicate_refusal(self):
        from docci_digital_compare_v19.collect import collect
        run=self.root/'fixture';run.mkdir();atomic(run/'PIPELINE.json',dict(status='TEST_ONLY'))
        destination=self.root/'fixture_records.tar.gz'
        receipt=dict(status='TEST_ONLY',comparison=None,stages={})
        with patch('docci_digital_compare_v19.collect.verify',return_value=receipt):
            collect(run,destination,1)
            with self.assertRaises(FileExistsError):collect(run,destination,1)
        with tarfile.open(destination) as tf:
            m=json.load(tf.extractfile('v19_records/ARCHIVE_MANIFEST.json'))
            for n,h in m['files'].items():self.assertEqual(hashlib.sha256(tf.extractfile('v19_records/'+n).read()).hexdigest(),h)
    def test_weights_never_enter_light_records(self):
        from docci_digital_compare_v19.collect import collect
        run=self.root/'fixture';run.mkdir();(run/'last.pth').write_bytes(b'TEST_ONLY')
        with patch('docci_digital_compare_v19.collect.verify',return_value=dict(status='TEST_ONLY',comparison=None,stages={})):
            with self.assertRaisesRegex(ValueError,'stay on server'):collect(run,self.root/'bad.tar.gz',1)
    def test_missing_completed_rows_rejected_even_with_new_seal(self):
        r=self.make_stage();p=r/'rows.jsonl';rows=lines(p)[:-1];p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        sealed=json.loads((r/'SEALED.json').read_text());sealed['rows.jsonl']=sha(p);(r/'SEALED.json').write_text(json.dumps(sealed))
        with self.assertRaisesRegex(ValueError,'Missing/duplicate'):stage(r,self.prepared,self.c,lambda:None)

if __name__=='__main__':unittest.main(verbosity=2)
