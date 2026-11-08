"""MIT: anonymously stream an explicitly chosen split without whole-shard downloads.

Requires pyarrow, huggingface_hub, and the sibling public_corpus_reader.py.
"""
import argparse,json
import requests
from huggingface_hub import HfApi,HfFileSystem
from public_corpus_reader import iter_split


def stream(repo,revision,split,source=None):
    fs=HfFileSystem(token=False)
    prefix=f'datasets/{repo}@{revision}'
    # Small Git JSON may be HTTP-compressed; use an explicit identity request.
    # Parquet payloads below use ranged binary reads, never a full JSON file cache.
    session=requests.Session();session.trust_env=False
    manifest_path=source+'/manifest.json' if source else 'manifest.json'
    response=session.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{manifest_path}',
                         headers={'Accept-Encoding':'identity'},timeout=60)
    response.raise_for_status();manifest=response.json()
    for item in manifest['source_documents']:
        with fs.open(prefix+'/'+item['text_path'],'rb',block_size=4*1024**2) as text, \
             fs.open(prefix+'/'+item['assignment_path'],'rb',block_size=4*1024**2) as assignments:
            yield from iter_split(text,assignments,split=split,source=manifest['source'],
                original_file=item['original_final_file'],inventory_id=manifest['content_inventory_id'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--revision');
    p.add_argument('--source',required=True);p.add_argument('--split',required=True,choices=['train','validation']);p.add_argument('--limit',type=int,default=10)
    a=p.parse_args();revision=a.revision or HfApi(token=False).repo_info(a.repo,repo_type='dataset').sha
    print(json.dumps({'repository':a.repo,'revision':revision,'split':a.split}))
    for i,row in enumerate(stream(a.repo,revision,a.split,source=a.source)):
        print(json.dumps(row,ensure_ascii=False))
        if i+1>=a.limit:break
