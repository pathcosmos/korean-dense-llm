"""Anonymous bounded real-data consumer check; never runs the 3B training job."""
import hashlib,itertools,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT


def check_sources(commits):
    root=Path(PROJECT_ROOT);examples=root/'runs/dataset_upload_v1/index_assets/examples';sys.path.insert(0,str(examples))
    from stream_documents import stream
    from huggingface_hub import HfApi
    public=HfApi(token=False);results=[]
    for repo,revision in commits.items():
        assert public.repo_info(repo,repo_type='dataset',revision=revision).private is False
        observed={}
        for split in ['train','validation']:
            rows=list(itertools.islice(stream(repo,revision,split),16));assert len(rows)==16
            observed[split]=[r['doc_id'] for r in rows]
            assert all(r['split']==('eval' if split=='validation' else 'train') for r in rows)
        assert not set(observed['train'])&set(observed['validation'])
        results.append({'repository':repo,'revision':revision,'sample_documents':32,'canonical_splits_disjoint':True})
    return results


def tiny_cpu_check(repo,revision,manifest):
    import io
    import numpy as np
    import requests
    import torch
    torch.set_num_threads(2);torch.manual_seed(1234);torch.use_deterministic_algorithms(True)
    tokens={};session=requests.Session();session.trust_env=False
    if manifest is None:
        response=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/manifest.json',
                             headers={'Accept-Encoding':'identity'},timeout=60);response.raise_for_status();m=response.json()
    else:m=manifest
    for split in ['train','validation']:
        chosen=next(r for r in m['payload_files'] if f'/tokens/{split}/' in '/'+r['path'])
        with session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{chosen["path"]}',
                         headers={'Range':'bytes=0-255'},stream=True,timeout=60) as response:
            response.raise_for_status();content=next(response.iter_content(256))[:256]
        assert len(content)==256
        tokens[split]=torch.from_numpy(np.frombuffer(content,dtype='<u2').astype(np.int64))
    def model():return torch.nn.Sequential(torch.nn.Embedding(65536,8),torch.nn.Linear(8,65536))
    a=model();opt=torch.optim.AdamW(a.parameters(),lr=.001)
    batch=tokens['train'][:18].reshape(2,9);x,y=batch[:,:-1],batch[:,1:]
    def step(m,o):
        o.zero_grad();loss=torch.nn.functional.cross_entropy(m(x).reshape(-1,65536),y.reshape(-1));assert torch.isfinite(loss)
        loss.backward();o.step();return float(loss.detach())
    loss=step(a,opt);buffer=io.BytesIO();torch.save({'model':a.state_dict(),'optimizer':opt.state_dict()},buffer)
    step(a,opt);reference={k:v.detach().clone() for k,v in a.state_dict().items()}
    b=model();other=torch.optim.AdamW(b.parameters(),lr=.001);buffer.seek(0);saved=torch.load(buffer,weights_only=True)
    b.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);step(b,other)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in reference.items())
    validation=tokens['validation'][:18].reshape(2,9)
    with torch.no_grad():vl=torch.nn.functional.cross_entropy(b(validation[:,:-1]).reshape(-1,65536),validation[:,1:].reshape(-1))
    assert torch.isfinite(vl)
    return {'status':'passed','model':'CPU_toy_embedding8_linear65536_not_published_3B_model',
            'real_public_token_input':True,'initial_train_loss':loss,'validation_loss':float(vl),
            'save_resume_second_update_exact':True,'cuda_used':False}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--commits',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    commits=json.loads(Path(a.commits).read_text());results=check_sources(commits)
    repo=next(iter(commits));tiny=tiny_cpu_check(repo,commits[repo],None)
    Path(a.output).write_text(json.dumps({'status':'passed','anonymous_source_checks':results,'bounded_cpu_training':tiny,
        'full_clean_room_3B_training_reproduction':False},indent=2)+'\n')
    print(json.dumps({'status':'passed','sources':len(results),'cpu_save_resume_exact':True}))
