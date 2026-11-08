"""Real CPU/Gloo collective protocol; no claim of actual1B/FSDP execution."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train.rehearsal import distributed_generate


def process(rank,root):
    torch.set_num_threads(1);root=Path(root)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=(root/'group').as_uri())
    class Model(torch.nn.Module):
        def forward(self,x):
            logits=torch.zeros((1,x.shape[1],8))
            selected=(3 if int(x[0,-1])!=3 else 1) if rank==0 else 7
            logits[0,-1,selected]=10
            return logits
    model=Model()
    try:
        generated=distributed_generate(model,model,[2],eos_id=1,context_length=8,max_new_tokens=5,device='cpu')
        def bad(x):return torch.full((1,x.shape[1],8),float('nan')) if rank else model(x)
        stopped=False
        try:distributed_generate(model,bad,[2],eos_id=1,context_length=8,max_new_tokens=5,device='cpu')
        except FloatingPointError:stopped=True
        (root/f'rank{rank}.json').write_text(json.dumps(dict(generated=generated,nonfinite_global_stop=stopped,training_mode_restored=model.training)))
    finally:dist.destroy_process_group()


class DistributedGenerationTest(unittest.TestCase):
    def test_different_local_argmax_uses_same_eos_and_nonfinite_stops_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(process,args=(str(Path(tmp).resolve()),),nprocs=2,join=True)
            records=[json.loads((Path(tmp)/f'rank{rank}.json').read_text()) for rank in range(2)]
            self.assertEqual(records[0],records[1]);self.assertEqual(records[0]['generated']['generated_ids'],[3,1])
            self.assertTrue(records[0]['nonfinite_global_stop']);self.assertTrue(records[0]['training_mode_restored'])


if __name__=='__main__':unittest.main()
