"""Bounded trace/storage fixtures; not evidence of actual1B rehearsal completion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_rehearsal import make_config,trace_review,certify_checkpoint,recover_checkpoint_prefix


def fixtures():
    categories=dict(wiki='ko',fw='ko',hplt='ko',english='en',code='code',math='math')
    groups=[dict(source=s,category=c,count=i+1) for i,(s,c) in enumerate(categories.items())]
    probe=dict(seq_len=4096,tokenizer_sha256='fixture',groups=groups,holdout=['fixture.hold.bin'])
    mix=dict(seq_len=4096,tokenizer_sha256='fixture',sources={s:dict(weight=1/6) for s in categories})
    config=make_config(dict({'1b':dict(micro_bsz=8,global_tokens=2097152,lr=.0006),'3b':dict(micro_bsz=4,global_tokens=2097152,lr=.0003)}),mix,probe,'output')
    return config,probe,mix


def trace(config,probe):
    from train.train import lr_at
    rows=[]
    for step in [1,2,3,4,*range(10,5001,10)]:
        rows.append(dict(step=step,loss=10-step/1000,lr=lr_at(step-1,config),gnorm=2.,tok_s=300000.,tokens_b=step*config['global_tokens']/1e9))
    for step in [2,4,*range(250,5001,250)]:
        by_source={};by_category={};nll=0.;tokens=0
        for i,group in enumerate(probe['groups']):
            count=group['count']*probe['seq_len'];loss=10+i-step/1000
            by_source[group['source']]=dict(valid_tokens=count,nll_sum=count*loss,holdout_loss=loss)
            c=by_category.setdefault(group['category'],dict(valid_tokens=0,nll_sum=0.));c['valid_tokens']+=count;c['nll_sum']+=count*loss
            nll+=count*loss;tokens+=count
        for c in by_category.values():c['holdout_loss']=c['nll_sum']/c['valid_tokens']
        r=dict(step=step,holdout_loss=nll/tokens,valid_tokens=tokens,by_source=by_source,by_category=by_category)
        if step in [4,2500,5000]:r['fixed_generations']=[dict(id=str(i),category='ko' if i<3 else 'en') for i in range(6)]
        rows.append(r)
    return rows


class RehearsalControllerTest(unittest.TestCase):
    def test_planned_token_geometry_preserves_all_checkpoint_outputs(self):
        config,probe,mix=fixtures()
        self.assertEqual(config['max_steps']*config['global_tokens'],10485760000)
        self.assertEqual(config['keep_ckpts'],config['max_steps']+1);self.assertFalse(config['compile'])
        self.assertEqual(config['model'],'1b');self.assertEqual(config['holdout'],probe['holdout'])
        with self.assertRaisesRegex(ValueError,'geometry'):make_config({'1b':dict(micro_bsz=8,global_tokens=1048576)},mix,probe,'other')
        with self.assertRaisesRegex(ValueError,'tokenizer'):make_config({'1b':dict(micro_bsz=8,global_tokens=2097152)},dict(mix,tokenizer_sha256='changed'),probe,'other')

    def test_complete_weighted_curves_and_replays_cannot_mask_missing_results(self):
        config,probe,_=fixtures();rows=trace(config,probe)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.jsonl'
            def save(value):path.write_text(''.join(json.dumps(r)+'\n' for r in value))
            save(rows);r=trace_review(path,config,probe)
            self.assertEqual((r['train_points'],r['eval_points']),(504,22));self.assertTrue(r['loss_decreased']);self.assertEqual(r['trained_tokens'],10485760000)
            save(rows+[copy.deepcopy(rows[5])]);self.assertEqual(trace_review(path,config,probe),r)
            bad=copy.deepcopy(rows);bad[-1]['by_category']['ko']['holdout_loss']+=.1;save(bad)
            with self.assertRaisesRegex(ValueError,'token weighted'):trace_review(path,config,probe)
            bad=copy.deepcopy(rows);bad[-1].pop('fixed_generations');save(bad)
            with self.assertRaisesRegex(ValueError,'generation'):trace_review(path,config,probe)
            save(rows[:-1])
            with self.assertRaisesRegex(ValueError,'Incomplete'):trace_review(path,config,probe)
            changed=copy.deepcopy(rows[5]);changed['loss']+=.1;save(rows+[changed])
            with self.assertRaisesRegex(ValueError,'trajectory'):trace_review(path,config,probe)
            bad=copy.deepcopy(rows);bad[0]['lr']=.0006;save(bad)
            with self.assertRaisesRegex(ValueError,'warmup/WSD'):trace_review(path,config,probe)

    def test_checkpoint_content_proof_rejects_changed_payload_and_missing_rank_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'training').mkdir();(root/'checkpoint_certificates').mkdir()
            checkpoint=root/'training/ckpt_0000004';checkpoint.mkdir()
            (checkpoint/'.complete').touch();(checkpoint/'.metadata').write_bytes(b'small storage fixture')
            for rank in range(8):(checkpoint/f'rng_rank{rank}.pt').write_bytes(bytes([rank]))
            atomic=dict(step=4,world_size=8,tokens_per_step=2097152,config=dict(model='1b'),training_contract=dict(id='fixture'))
            (checkpoint/'meta.json').write_text(json.dumps(atomic));output=root/'checkpoint_certificates/ckpt_0000004.json'
            first=certify_checkpoint(checkpoint,output,'fixture');self.assertEqual(first,certify_checkpoint(checkpoint,output,'fixture'))
            unverified=root/'training/ckpt_0000005';unverified.mkdir();(unverified/'.complete').touch();(unverified/'payload').write_bytes(b'preserve')
            verified={};moved=recover_checkpoint_prefix(root,verified);self.assertEqual(len(moved),1);self.assertFalse(unverified.exists())
            self.assertEqual(verified['ckpt_0000004'],first)
            self.assertEqual(Path(moved[0]['preserved']).joinpath('payload').read_bytes(),b'preserve');self.assertTrue(checkpoint.exists())
            (checkpoint/'rng_rank3.pt').write_bytes(b'Z')
            with self.assertRaisesRegex(ValueError,'proof changed'):certify_checkpoint(checkpoint,output,'fixture')
            (checkpoint/'rng_rank3.pt').unlink()
            with self.assertRaisesRegex(ValueError,'RNG missing'):certify_checkpoint(checkpoint,root/'missing.json','fixture')


if __name__=='__main__':unittest.main()
