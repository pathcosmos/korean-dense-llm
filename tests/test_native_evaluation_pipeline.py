"""Small metadata/provenance rejection checks, never actual-model certification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from common.data_contracts import atomic_json,sha256_file
from scripts.run_native_evaluation import certified,checkpoint_inputs,verify_evaluation


class NativePipelineTest(unittest.TestCase):
    def test_bound_actual_checkpoint_metadata_and_changed_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings={m:dict(micro_bsz=4,global_tokens=2097152,lr=.0003) for m in ['1b','3b']}
            results=[]
            for model in settings:
                checkpoint=root/(model+'_validated_checkpoint');checkpoint.mkdir()
                atomic_json(checkpoint/'meta.json',dict(step=4,config=dict(model=model,**settings[model]),training_contract=dict(mix=dict(tokenizer_sha256='fixture'))))
                (checkpoint/'.complete').touch();atomic_json(checkpoint/'publication.json',dict(files=[]))
                results.append(dict(model=model,published_checkpoint=dict(path=str(checkpoint),publication_sha256=sha256_file(checkpoint/'publication.json'))))
            def publish():
                atomic_json(root/'report.json',dict(status='passed_bounded_actual_training',results=results))
                atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
            atomic_json(root/'contract.json',dict(world=8,sequence=4096,steps=4,compile=False,code={},batch_geometry=settings));publish()
            self.assertEqual(set(checkpoint_inputs(root,'fixture')),{'1b','3b'})
            with self.assertRaisesRegex(ValueError,'tokenizer'):checkpoint_inputs(root,'changed')
            (root/'1b_validated_checkpoint/publication.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'publication'):checkpoint_inputs(root,'fixture')
            results.pop();publish()
            with self.assertRaises(ValueError):checkpoint_inputs(root,'fixture')
            (root/'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'proof'):certified(root,'passed_bounded_actual_training')

    def test_reuse_fresh_checkpoint_hashes_and_require_both_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'weights').write_bytes(b'fixture')
            atomic_json(root/'publication.json',dict(files=[dict(path='weights',sha256=sha256_file(root/'weights'))]))
            report=dict(status='passed',checkpoint_step=4,languages={k:dict(valid_tokens=1) for k in ['ko','en']},
                        generations=[dict(id=str(i),category='ko' if i<3 else 'en') for i in range(6)],
                        contract=dict(tokenizer_sha256='fixture',input_provenance=dict(level='verified-eval-split'),
                                      checkpoint={p.name:sha256_file(p) for p in root.iterdir()}))
            verify_evaluation(report,root,'fixture')
            bad=copy.deepcopy(report);bad['contract']['checkpoint']['weights']='changed'
            with self.assertRaisesRegex(ValueError,'publication'):verify_evaluation(bad,root,'fixture')
            bad=copy.deepcopy(report);bad['languages']['ko']['valid_tokens']=0
            with self.assertRaisesRegex(ValueError,'targets'):verify_evaluation(bad,root,'fixture')
            bad=copy.deepcopy(report);bad['contract']['input_provenance']['level']='manual-input'
            with self.assertRaisesRegex(ValueError,'provenance'):verify_evaluation(bad,root,'fixture')
            bad=copy.deepcopy(report);bad['generations'][5]['id']='0'
            with self.assertRaisesRegex(ValueError,'generation'):verify_evaluation(bad,root,'fixture')


if __name__=='__main__':unittest.main()
