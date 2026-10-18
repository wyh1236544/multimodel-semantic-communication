"""CPU-only repair/negative/recovery contracts; fixtures are TEST_ONLY."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from docci_text_guided_image_v20.core import PROJECT,load,atomic,sha,read
from docci_text_guided_image_v20_fix1 import compat
from docci_text_guided_image_v20_fix1.launch import rewrite,routed
from docci_text_guided_image_v20_fix1.entry import remove_option

class Array:
    def __init__(self,x): self.x=x;self.shape=x.shape
    def reshape(self,shape): return Array(self.x.reshape(shape))
    def mean(self,dim): return self.x.mean(axis=dim)

class RepairContracts(unittest.TestCase):
    def test_bins_match_independent_adaptive_definition(self):
        rng=np.random.RandomState(42)
        for h,w in ((16,16),(32,32),(64,64),(16,32)):
            x=rng.normal(size=(2,3,h,w))
            expected=np.empty((2,3,4,8))
            for i in range(4):
                for j in range(8):
                    expected[:,:,i,j]=x[:,:,i*h//4:(i+1)*h//4,j*w//8:(j+1)*w//8].mean((2,3))
            np.testing.assert_allclose(compat.fixed_pool(Array(x)),expected,rtol=1e-14,atol=1e-14)

    def test_unsupported_geometry_never_silently_changes_bins(self):
        for shape in ((1,3,17,32),(1,3,16,31),(0,3,16,32),(3,16,32)):
            with self.assertRaises(ValueError): compat.layout(shape)
        with self.assertRaises(ValueError): compat.layout((1,3,16,32),(8,4))

    def test_module_scope_and_exception_restore(self):
        old=SimpleNamespace(adaptive_avg_pool2d=object())
        module=SimpleNamespace(F=old)
        with self.assertRaisesRegex(RuntimeError,'TEST_ONLY'):
            with compat.pooling_patch(module):
                self.assertIs(module.F.adaptive_avg_pool2d,compat.fixed_pool)
                self.assertIsNot(module.F,old)
                raise RuntimeError('TEST_ONLY')
        self.assertIs(module.F,old)

    def test_rewrite_preserves_all_legacy_arguments_and_other_commands(self):
        old=compat.ORIGINAL
        cmd=['python','-B',str(old/'run.py'),'train','NEW','--arm','A_self']
        self.assertEqual(rewrite(cmd)[4:],cmd[3:])
        self.assertEqual(rewrite(['python','elsewhere.py']),['python','elsewhere.py'])
        cache=['python',str(old/'run.py'),'cache','NEW','--smoke']
        self.assertEqual(rewrite(cache,reuse='OLD')[-2:],['--reuse-smoke-cache','OLD'])
        self.assertNotIn('--reuse-smoke-cache',rewrite(cache[:-1],reuse='OLD'))
        for name in ('report','collect'):
            self.assertEqual(rewrite(['python',str(old/(name+'.py')),'ROOT'])[2],name)
        test=rewrite(['python',str(old/'test_model.py')],gate='GATE')
        self.assertEqual(test[-2:],['--receipt','GATE'])

    def test_router_restores_owned_process_constructor(self):
        import subprocess
        from docci_text_guided_image_v20 import launch
        original_popen,original_snapshot=subprocess.Popen,launch.snapshot
        with self.assertRaisesRegex(RuntimeError,'TEST_ONLY'):
            with routed(None,None):
                self.assertIsNot(subprocess.Popen,original_popen)
                raise RuntimeError('TEST_ONLY')
        self.assertIs(subprocess.Popen,original_popen)
        self.assertIs(launch.snapshot,original_snapshot)

    def test_option_removal_is_exact_and_does_not_touch_other_args(self):
        argv,value=remove_option(['pipeline','NEW','--reuse-smoke-cache','OLD','--max-hours','48'],'--reuse-smoke-cache')
        self.assertEqual(value,'OLD');self.assertEqual(argv,['pipeline','NEW','--max-hours','48'])
        for bad in (['--reuse-smoke-cache'],['--reuse-smoke-cache','a','--reuse-smoke-cache','b']):
            with self.assertRaises(ValueError): remove_option(bad,'--reuse-smoke-cache')

    def test_forged_cuda_gate_rejected(self):
        gate=dict(schema=compat.SCHEMA,repair_manifest_sha256='TEST_ONLY_FROZEN_FIX',
                  CUDA_gate_passed=True,strict_determinism=True,GPU_tests_run=3,GPU_tests_skipped=0,physical_GPU='1')
        with patch.object(compat,'sha',return_value='TEST_ONLY_FROZEN_FIX'):
            compat.validate_gate(gate)
            for key,value in (('strict_determinism',False),('GPU_tests_skipped',1),('physical_GPU','0'),('repair_manifest_sha256','wrong')):
                with self.assertRaises(ValueError): compat.validate_gate(dict(gate,**{key:value}))

class ReuseContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v20fix1_TEST_ONLY_',dir=str(PROJECT/'experiment_results'))
        self.base=Path(self.temp.name)
        self.source=self.base/'old/smoke_cache';self.source.mkdir(parents=True)
        self.target=self.base/'new/smoke_cache';self.target.parent.mkdir()
        for root in (self.source.parent,self.target.parent): atomic(root/'config.json',load())
        entries=[]
        (self.source/'arrays').mkdir()
        for group,count in (('train',8),('dev',2),('qual_dev',2)):
            for i in range(count):
                eid='TEST_ONLY_'+group+str(i);n='arrays/'+eid+'.npz'
                (self.source/n).write_bytes(b'TEST_ONLY NOT NEURAL ARRAYS')
                entries.append(dict(example_id=eid,cache_file=n))
        for n in ('CACHE_COMPLETE.json','backbones.json','frozen_audit.json','parents.json','protected_split.json','source_identity.json'):
            atomic(self.source/n,dict(TEST_ONLY=True))
        atomic(self.source/'index.json',dict(smoke=True,records=entries,image_codec_outputs_cached=False,V18_text_wire_frozen=True))
        self.before={p.relative_to(self.source).as_posix():sha(p) for p in self.source.rglob('*') if p.is_file()}
        self.verifier=patch('docci_text_guided_image_v20.report.check_cache',return_value={'TEST_ONLY':True})
        self.mock=self.verifier.start()

    def tearDown(self): self.verifier.stop();self.temp.cleanup()

    def test_copy_reuses_without_old_source_mutation_or_overwrite(self):
        compat.reuse_cache(self.source,self.target)
        self.assertEqual(self.mock.call_count,2)
        for n,h in self.before.items():
            self.assertEqual(sha(self.source/n),h);self.assertEqual(sha(self.target/n),h)
            self.assertNotEqual((self.source/n).stat().st_ino,(self.target/n).stat().st_ino)
        self.assertFalse(read(self.target.parent/'SMOKE_CACHE_REUSE.json')['cache_generation_repeated'])
        with self.assertRaises(FileExistsError): compat.reuse_cache(self.source,self.target)

    def test_invalid_cache_rejected_before_copy(self):
        self.mock.side_effect=ValueError('TEST_ONLY corrupt array')
        with self.assertRaises(ValueError): compat.reuse_cache(self.source,self.target)
        self.assertFalse(self.target.exists())

    def test_different_config_or_full_cache_rejected(self):
        c=load();c['epochs']=9;atomic(self.target.parent/'config.json',c)
        with self.assertRaises(ValueError): compat.reuse_cache(self.source,self.target)
        atomic(self.target.parent/'config.json',load())
        index=read(self.source/'index.json');index['smoke']=False;atomic(self.source/'index.json',index)
        with self.assertRaises(ValueError): compat.reuse_cache(self.source,self.target)

    def test_traversal_symlink_and_unexpected_files_rejected(self):
        index=read(self.source/'index.json');index['records'][0]['cache_file']='../escape.npz'
        atomic(self.source/'index.json',index)
        with self.assertRaises(ValueError): compat.reuse_cache(self.source,self.target)
        index['records'][0]['cache_file']='arrays/'+index['records'][0]['example_id']+'.npz'
        atomic(self.source/'index.json',index)
        link=self.source/'unexpected';link.symlink_to(self.source/'index.json')
        with self.assertRaises(ValueError): compat.reuse_cache(self.source,self.target)
        link.unlink()
        link.write_bytes(b'TEST_ONLY unexpected regular file')
        with self.assertRaises(ValueError): compat.reuse_cache(self.source,self.target)

class ProvenanceContracts(unittest.TestCase):
    def setUp(self):
        from docci_text_guided_image_v20.test_storage import fixture
        self.temp=tempfile.TemporaryDirectory(prefix='v20fix1_PROVENANCE_TEST_ONLY_',dir=str(PROJECT/'experiment_results'))
        self.base=Path(self.temp.name);self.root=self.base/'run';self.root.mkdir()
        fixture(self.root)
        self.fix=self.base/'fix';self.fix.mkdir()
        (self.fix/'dummy.py').write_text('# TEST_ONLY\n')
        atomic(self.fix/'MANIFEST.json',dict(TEST_ONLY=True))
        self.manifest=dict(files={'dummy.py':sha(self.fix/'dummy.py')})
        self.folder_patch=patch.object(compat,'FOLDER',self.fix)
        self.check_patch=patch.object(compat,'source_check',return_value=self.manifest)
        self.folder_patch.start();self.check_patch.start()
        gate=dict(schema=compat.SCHEMA,repair_manifest_sha256=sha(self.fix/'MANIFEST.json'),
                  CUDA_gate_passed=True,strict_determinism=True,GPU_tests_run=3,GPU_tests_skipped=0,
                  physical_GPU='1',TEST_ONLY_NOT_ACTUAL_CUDA=True)
        atomic(self.base/'gate.json',gate);compat.snapshot(self.root,self.base/'gate.json')
        for stage in read(self.root/'PIPELINE.json')['stages']:
            if 'train_' in stage or 'eval_' in stage:
                atomic(self.root/stage/'FIX1_RUNTIME.json',dict(repair_manifest_sha256=sha(self.fix/'MANIFEST.json'),
                    strict_determinism=True,module_local_pooling_replaced=True,TEST_ONLY=True))

    def tearDown(self):
        self.check_patch.stop();self.folder_patch.stop();self.temp.cleanup()

    def test_wrapped_verifier_preserves_all_original_numeric_fields(self):
        from docci_text_guided_image_v20 import report
        original=report.verify(self.root)
        with compat.installed() as verify:
            result=verify(self.root)
        self.assertEqual({k:v for k,v in result.items() if k!='compatibility_fix1'},original)

    def test_neural_completion_requires_actual_fix_runtime_receipt(self):
        p=self.root/'smoke_train_A_self/FIX1_RUNTIME.json'
        atomic(p,dict(repair_manifest_sha256=sha(self.fix/'MANIFEST.json'),strict_determinism=False,
                      module_local_pooling_replaced=True))
        with compat.installed() as verify:
            with self.assertRaisesRegex(ValueError,'Verified neural stage'):
                verify(self.root)

    def test_tampered_fix_snapshot_rejected(self):
        p=self.root/'compatibility_snapshot/docci_text_guided_image_v20_fix1/dummy.py'
        p.write_text('# TEST_ONLY changed\n')
        with compat.installed() as verify:
            with self.assertRaisesRegex(ValueError,'Fix snapshot changed'):
                verify(self.root)

    def test_private_collection_keeps_fix_gate_snapshot_and_runtime(self):
        import tarfile
        from docci_text_guided_image_v20 import report
        from docci_text_guided_image_v20.collect import collect
        result=report.verify(self.root)
        # TEST_ONLY lightweight fixture has no arrays; no full neural claim.
        with patch.object(report,'verify',return_value=result),compat.installed():
            archive=self.base/'records.tar.gz';collect(self.root,archive,1)
        with tarfile.open(archive) as tf:
            names={m.name for m in tf.getmembers()}
        for n in ('V20_FIX1.json','FIX1_GPU_GATE.json','compatibility_snapshot/docci_text_guided_image_v20_fix1/dummy.py',
                  'smoke_train_A_self/FIX1_RUNTIME.json'):
            self.assertIn('v20_records/'+n,names)

if __name__=='__main__': unittest.main(verbosity=2)
