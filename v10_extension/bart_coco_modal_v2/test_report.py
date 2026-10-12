"""Synthetic metadata tests ONLY: partial text failure cannot suppress image records."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from bart_coco_modal_v2.core import load,geometry,metrics,quality_gate
from bart_coco_modal_v2.report import matrix_check,text_check,controls,MARKERS,inspect,report,seal,verify
from bart_coco_modal_v2 import VERSION


def fixture():
    c=load(); c.update(train_samples=8,val_samples=2)
    views={}
    for mode in ('image_only','text_only'):
        views[mode]={part:dict(mode=mode,base_image_split_sha256='0'*64,original_split_not_rewritten=True,text_encoder_executed=mode!='image_only',image_file_opened=mode!='text_only',
            image_ids=ids,source_sha256={str(i):'fixture' for i in ids},annotation_ids={str(i):i*10 for i in ids}) for part,ids in [('train',list(range(8))),('validation',[101,102])]}
    d=dict(config=c,caption_views=views,channel_config=dict(conditions=[]),resources={})
    states=dict(image_only=dict(evaluated=True,status='evaluated',rows=2),text_only=dict(evaluated=False,status='overfit_stop'))
    rows=[dict(mode='image_only',condition='clean',image_id=i,annotation_id=i*10,source_sha256='fixture',source_split='validation',geometry=geometry('image_only'),
        text_used_for_source=False,image_used_for_source=True,source_utf8_bytes=None,received_caption=None,psnr=20.,ssim=.7,
        stats=dict(receiver_available=True,pre_fec_ber=None,post_fec_ber=None,clean_control_no_physical_transmission=True,scheduled_seconds=None)) for i in (101,102)]
    return rows,d,states


class Contracts(unittest.TestCase):
    def test_negative_text_retains_independent_image_matrix(self):
        rows,d,s=fixture(); matrix_check(rows,d,s)
    def test_missing_image_not_silently_complete(self):
        rows,d,s=fixture()
        with self.assertRaises(ValueError): matrix_check(rows[:1],d,s)
    def test_duplicate_not_complete(self):
        rows,d,s=fixture()
        with self.assertRaises(ValueError): matrix_check(rows+[rows[0]],d,s)
    def test_opposite_modality_metric_rejected(self):
        rows,d,s=fixture(); rows[0]['received_caption']='not transmitted'
        with self.assertRaises(ValueError): matrix_check(rows,d,s)
    def test_source_leak_rejected(self):
        rows,d,s=fixture(); rows[0]['text_used_for_source']=True
        with self.assertRaises(ValueError): matrix_check(rows,d,s)
    def test_true_text_metrics_and_unbounded_wer(self):
        r=dict(source_text='one',covered_text='one',received_caption='two three four',**metrics('one','one','two three four'))
        text_check(r); self.assertGreater(r['covered_wer'],1.); r['covered_wer']=0.
        with self.assertRaises(ValueError): text_check(r)
    def test_actual_controls_define_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'controls.jsonl'; rows=[]
            for k in ('matched','zero','shuffled'):
                for i in (1,2):
                    received='one' if k=='matched' else 'different'
                    rows.append(dict(control=k,image_id=i,source_text='one',covered_text='one',received_caption=received,input_truncated=False,stopped_by_capacity=False,**metrics('one','one',received)))
            path.write_text('\n'.join(json.dumps(r) for r in rows)); self.assertTrue(quality_gate(controls(path),load())['passed'])
            path.write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
            with self.assertRaises(ValueError): controls(path)
    def test_markers_are_distinct(self):
        self.assertEqual(len(set(MARKERS.values())),3); self.assertNotEqual(MARKERS['complete'],MARKERS['partial_complete'])
    def full_fixture(self,root):
        def write(name,value):
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))
        def jsonl(name,rows):
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('\n'.join(json.dumps(v) for v in rows)+'\n')
        rows,design,states=fixture(); cfg=design['config']; cfg.update(diagnostic_samples=2,diagnostic_updates=8,feature_warmup_updates=2,clean_epochs=1,save_images=0)
        design.update(version=VERSION,modes=['image_only','text_only'],smoke=False,clean_training_only=True,no_test_source_used=True,limitations='synthetic fixture, NOT neural results')
        write('run_identity.json',dict(version=VERSION,status='partial_complete',wall_seconds=1.,split_sha256='0'*64))
        write('design.json',design); write('mode_status.json',states)
        write('frozen_verification.json',dict(passed=True,BART_reference_VAE_CLIP_unchanged=True,no_test_source_used=True))
        jsonl('samples.jsonl',rows); (root/'events.jsonl').write_text(''); (root/'common_controls.jsonl').write_text('')
        controls_rows=[]; native=[]
        for k in ('matched','zero','shuffled'):
            for i in (0,1):
                row=dict(control=k,image_id=i,source_split='train',source_text='one',covered_text='one',received_caption='wrong',input_truncated=False,stopped_by_capacity=False,**metrics('one','one','wrong'))
                controls_rows.append(row)
                if k=='matched': native.append(dict(row,diagnostic_uncompressed=True))
        jsonl('text_only/pretrained_uncompressed.jsonl',native)
        for label in ('private_T32','mode_overfit'):
            jsonl('text_only/'+label+'_controls.jsonl',controls_rows)
            write('text_only/'+label+'_gate.json',quality_gate(controls(root/'text_only'/(label+'_controls.jsonl')),cfg))
        for name in ('image_only/clean','text_only/private_T32_diagnostic','text_only/mode_diagnostic'):
            write(name+'/learning_audit.json',dict(passed=True,frozen_unchanged=True,groups=dict(fixture=dict(parameters=1,nonzero_gradient=True,changed=True))))
            write(name+'/forward_consistency.json',dict(passed=True,errors=dict(source=0.)))
            write(name+'/TRAIN_COMPLETE.json',dict(steps=8,optimizer_updates=4))
            jsonl(name+'/history.jsonl',[dict(step=i) for i in range(1,9)])
            (root/name/'last_bridge.pth').write_bytes(b'Synthetic integrity fixture; NOT a neural checkpoint')
        return design
    def test_full_partial_report_archive_and_hash_tamper(self):
        import tarfile
        from bart_coco_modal_v2.collect import collect
        checkdir=EXT.parent/'experiment_results/bart_coco_modal_v2_checks'; checkdir.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(checkdir)) as d:
            root=Path(d)/'records'; root.mkdir(); self.full_fixture(root)
            summary=report(root); self.assertEqual(summary['rows'],2); seal(root); self.assertEqual(verify(root),'partial_complete')
            dest=Path(d)/'records.tar.gz'; collect(root,dest)
            with tarfile.open(dest) as tar: self.assertFalse(any(n.endswith('.pth') for n in tar.getnames()))
            (root/'summary.json').write_text('{}')
            with self.assertRaises(ValueError): verify(root)
    def test_failed_text_cannot_claim_full_completion(self):
        checkdir=EXT.parent/'experiment_results/bart_coco_modal_v2_checks'; checkdir.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(checkdir)) as d:
            root=Path(d); self.full_fixture(root); path=root/'run_identity.json'; value=json.loads(path.read_text()); value['status']='complete'; path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): inspect(root)


if __name__=='__main__': unittest.main(verbosity=2)
