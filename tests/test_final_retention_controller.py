"""CPU controller/fault tests; synthetic summaries are not final3B evidence."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from common.data_contracts import resolve, atomic_json, sha256_file
from scripts.run_final_retention_audit import (GIB, capacity_check, require_result, verify_current,
    make_contract, owned_workers, reject_previous, validate_output)
from train.final_resume_probe import file_stats


class FinalRetentionControllerTest(unittest.TestCase):
    def test_full_reservations_and_existing_bytes_not_double_counted(self):
        c = dict(durable_budget_gib=1)
        p = dict(nfs_budget_bytes=437*GIB,local_budget_bytes=635*GIB,temporary_budget_bytes=1024*GIB,
                 reserve_bytes=200*GIB,other_nfs_future_bytes=0)
        v = dict(nfs_free=936*GIB,local_free=1888*GIB,main_used=0,history_used=0,temp_used=0,
                 own_used=0,native_used=0,resume_used=0,headroom_gib=1700,combined_memory_gib=912)
        result = capacity_check(c,p,**v)
        self.assertEqual(result['nfs_after_budget_reserve_bytes'],289*GIB)
        self.assertEqual(result['local_after_budget_reserve_bytes'],29*GIB)
        self.assertEqual(result['audit_additional_local_bytes'],0)
        changed = dict(v,nfs_free=v['nfs_free']-100*GIB,main_used=100*GIB,
                       local_free=v['local_free']-40*GIB,temp_used=40*GIB)
        self.assertEqual(result,capacity_check(c,p,**changed)|{
            'nfs_remaining_bytes':result['nfs_remaining_bytes'],
            'local_remaining_bytes':result['local_remaining_bytes']})
        for key,value in [('nfs_free',600*GIB),('local_free',1800*GIB),('headroom_gib',911),('own_used',2*GIB)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                capacity_check(c,p,**dict(v,**{key:value}))

    def summary(self,main='runs/test_main'):
        c = dict(main_run=main,source_contract_sha256='a'*64,code={'scripts/audit_checkpoint_retention.py':'b'*64},inputs={})
        cp = dict(model_tensors=255,state_tensors=1020,rng_ranks=8,payload_files=8,full_payload_sha_refreshed=True)
        nfs = {f'ckpt_{s:07d}':dict(cp) for s in range(52500,57001,500)}
        local = {f'ckpt_{s:07d}':dict(cp) for s in range(45000,52001,500)}
        r = dict(status='passed_final_retention_content',production=True,final=True,full_payload_sha_refreshed=True,
                 source_run=main,source_contract_sha256=c['source_contract_sha256'],audit_code_sha256='b'*64,
                 issues=[],residues=[],overlap=[],nfs=nfs,local=local,nfs_resume_candidates=sorted(nfs),expected_nfs=sorted(nfs),
                 expected_local=sorted(local),distinct_checkpoints=25,local_history_optional_loss=False,local_namespace_missing=False,
                 certificate_receipts={f'ckpt_{s:07d}.json':'x' for s in range(500,57001,500)},
                 training_audit=dict(validation_source=False,through_step=57000,
                     counts=dict(committed_tokens=119537664000,successful_target_ranks=list(range(8))),evidence={}))
        return c,r

    def test_marker_summary_fixture_partial_hash_and_incomplete_world_fail(self):
        c,r = self.summary();require_result(r,c)
        for k,v in [('production',False),('final',False),('full_payload_sha_refreshed',False),
                    ('status','passed_fixture_retention_content'),('issues',['bad']),('residues',[{}]),('distinct_checkpoints',24)]:
            with self.subTest(field=k),self.assertRaises(ValueError):require_result(dict(r,**{k:v}),c)
        for target in ['counts','checkpoint']:
            bad=copy.deepcopy(r)
            if target=='counts':bad['training_audit']['counts']['successful_target_ranks'].pop()
            else:bad['nfs']['ckpt_0057000']['rng_ranks']=7
            with self.assertRaises(ValueError):require_result(bad,c)

    def test_local_loss_is_explicit_and_never_becomes_an_nfs_resume_candidate(self):
        c,r=self.summary();r.update(local={},distinct_checkpoints=10)
        with self.assertRaisesRegex(ValueError,'Unexplained'):require_result(r,c)
        r.update(local_history_optional_loss=True,local_namespace_missing=True);require_result(r,c)
        self.assertEqual(r['nfs_resume_candidates'][-1],'ckpt_0057000')
        r['nfs_resume_candidates'].append('ckpt_0045000')
        with self.assertRaises(ValueError):require_result(r,c)

    def post_worker_fixture(self,root):
        main=root/'source';local=root/'local';main.mkdir();local.mkdir()
        for name in ['training','checkpoint_certificates','retention_trash']:(main/name).mkdir()
        for name in ['checkpoints','certificates','staging','retention_trash']:(local/name).mkdir()
        atomic_json(local/'.retention_owner.json',{'test_only':True})
        atomic_json(main/'contract.json',dict(tiered_retention=dict(local_root=str(local))))
        (main/'training/training_contract.json').write_text('{}');(main/'training/log.jsonl').write_text('')
        c,r=self.summary(str(main));c['source_contract_sha256']=sha256_file(main/'contract.json')
        r['source_contract_sha256']=c['source_contract_sha256']
        c['code']={'scripts/audit_checkpoint_retention.py':sha256_file(resolve('scripts/audit_checkpoint_retention.py'))}
        r['audit_code_sha256']=c['code']['scripts/audit_checkpoint_retention.py']
        r['resident_file_stats']={}
        for tier,base in [('nfs',main/'training'),('local',local/'checkpoints')]:
            for name in r[tier]:
                path=base/name;path.mkdir();(path/'test_payload').write_bytes(b'1234')
                r['resident_file_stats'][str(path)]=file_stats(path)
        for name in r['certificate_receipts']:
            path=main/'checkpoint_certificates'/name;path.write_text('{}');r['certificate_receipts'][name]=sha256_file(path)
        for name in r['local']:(local/'certificates'/(name+'.json')).write_text('{}')
        r['metadata_evidence']={str(p):sha256_file(p) for p in [main/'contract.json',local/'.retention_owner.json',*list((local/'certificates').iterdir())]}
        return c,r,main,local

    def test_post_worker_same_size_mutation_and_new_residue_invalidate_sha_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,r,main,local=self.post_worker_fixture(Path(tmp));verify_current(r,c)
            payload=main/'training/ckpt_0057000/test_payload';payload.write_bytes(b'4321')
            with self.assertRaisesRegex(ValueError,'changed after fresh SHA'):verify_current(r,c)
        with tempfile.TemporaryDirectory() as tmp:
            c,r,main,local=self.post_worker_fixture(Path(tmp));verify_current(r,c)
            (local/'staging/unverified').write_text('partial')
            with self.assertRaisesRegex(ValueError,'residue namespace'):verify_current(r,c)
            self.assertTrue((local/'staging/unverified').exists())

    def test_actual_orphan_is_exact_and_failed_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
                'run_final_retention_audit.py','--worker','--run-dir',str(root)])
            try:
                self.assertEqual(owned_workers(root),[proc.pid])
                self.assertEqual(owned_workers(root/'other'),[])
                with self.assertRaisesRegex(RuntimeError,'orphan'):reject_previous(root)
            finally:proc.terminate();proc.wait(timeout=5)
            (root/'worker_handle.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Preserve prior'):reject_previous(root)
            self.assertTrue((root/'worker_handle.json').exists())

    def test_actual_parent_paths_protected_and_current_bindings_preparable(self):
        active=json.loads(resolve('runs/main_training_readiness_v1/progress.json').read_text())
        native,resume=active['final_native_controller'],active['final_resume_controller']
        c=make_contract('runs/3b_full_v1',native,resume)
        for path in ['runs','runs/3b_full_v1/diagnostic',native+'/diagnostic']:
            with self.subTest(path=path),self.assertRaises(ValueError):validate_output(resolve(path),c)
        validate_output(resolve('runs/test_independent_retention_output'),c)
        self.assertEqual(c['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(c['max_pss_gib'],16)

    def test_sigterm_stops_owned_worker_only_and_preserves_incomplete_evidence(self):
        # Test-only observation/command substitution exercises the real CLI
        # signal handler and process supervisor, never claims actual final3B.
        with tempfile.TemporaryDirectory(prefix='retention-control-test-',dir=resolve('runs')) as tmp:
            root=Path(tmp)
            active=json.loads(resolve('runs/main_training_readiness_v1/progress.json').read_text())
            c=make_contract('runs/3b_full_v1',active['final_native_controller'],active['final_resume_controller'])
            atomic_json(root/'contract.json',c);atomic_json(root/'prepared.json',dict(contract_sha256=sha256_file(root/'contract.json')))
            code='''import argparse,sys
from pathlib import Path
import scripts.run_final_retention_audit as m
m.main_observation=lambda source: dict(complete=True)
original=m.supervise_command
def sleeping(command,**kw):
    return original([sys.executable,'-c','import time;time.sleep(30)'],**kw)
m.supervise_command=sleeping
m.main(argparse.Namespace(run_dir=sys.argv[1],worker=False,prepare=False))
'''
            other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            with (root/'test_controller.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-c',code,str(root)],stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+10
                while not (root/'worker_handle.json').exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.02)
                self.assertIsNone(proc.poll(),(root/'test_controller.log').read_text())
                handle=json.loads((root/'worker_handle.json').read_text());proc.send_signal(signal.SIGTERM);proc.wait(timeout=10)
                self.assertNotEqual(proc.returncode,0)
                self.assertFalse(Path(f"/proc/{handle['pid']}").exists())
                self.assertIsNone(other.poll());self.assertFalse((root/'.done').exists())
                self.assertTrue((root/'execution_started.json').exists())
                self.assertEqual(json.loads((root/'state.json').read_text())['phase'],'error')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
                other.terminate();other.wait(timeout=5)


if __name__=='__main__':unittest.main()
