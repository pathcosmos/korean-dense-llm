"""MIT: read an explicitly selected canonical split from original text shards.

No training on the all-documents view by accident: split is mandatory.
"""
from itertools import zip_longest
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def document_id(inventory_id,source,original_file,row):
    encoded=json.dumps(['release-doc-id-v1',inventory_id,source,original_file,row],
                       ensure_ascii=False,separators=(',',':')).encode('utf8')
    return hashlib.sha256(encoded).hexdigest()


def iter_split(text_path,assignment_path,*,split,source,original_file,inventory_id,batch_size=8192):
    if split=='validation':split='eval'
    if split not in ('train','eval'):raise ValueError('Explicit train or validation split required')
    text=pq.ParquetFile(text_path);assignment=pq.ParquetFile(assignment_path)
    if text.metadata.num_rows!=assignment.metadata.num_rows:raise ValueError('Text/assignment coverage mismatch')
    metadata=assignment.schema_arrow.metadata or {}
    if metadata.get(b'original_final_file')!=original_file.encode():raise ValueError('Assignment belongs to another shard')
    row_number=0
    for tb,ab in zip_longest(text.iter_batches(batch_size=batch_size,use_threads=False),
                            assignment.iter_batches(batch_size=batch_size,use_threads=False)):
        if tb is None or ab is None or tb.num_rows!=ab.num_rows:raise ValueError('Batch coverage mismatch')
        for row,assigned in zip(tb.to_pylist(),ab.to_pylist()):
            if assigned['row']!=row_number:raise ValueError('Assignment row order mismatch')
            if hashlib.sha256(row['text'].encode('utf8')).digest()!=assigned['content_sha256']:
                raise ValueError('Document body differs from canonical split')
            if assigned['split']==split:
                yield {**row,'doc_id':document_id(inventory_id,source,original_file,row_number),
                       'source':source,'original_row':row_number,'split':split,'group_id':assigned['group_id']}
            elif assigned['split'] not in ('train','eval'):raise ValueError('Unknown canonical split')
            row_number+=1


def download_and_iter(repo_id,*,revision,split,max_documents=None,source=None):
    """Authenticated downloads are unnecessary for the public release."""
    from huggingface_hub import hf_hub_download
    def fetch(path):return hf_hub_download(repo_id,path,repo_type='dataset',revision=revision,token=False)
    if source is not None and (not source or '/' in source or '\\' in source or source in ('.','..')):
        raise ValueError('Source must be a single source directory name')
    manifest_path=f'{source}/manifest.json' if source is not None else 'manifest.json'
    manifest=json.loads(Path(fetch(manifest_path)).read_text());done=0
    if not manifest.get('source_documents'):raise ValueError('No text shards listed')
    for item in manifest['source_documents']:
        tp=fetch(item['text_path']);ap=fetch(item['assignment_path'])
        # Bind cached files to release bytes before decoding.
        for p,wanted in [(tp,item['text_sha256']),(ap,item['assignment_sha256'])]:
            h=hashlib.sha256()
            with open(p,'rb') as f:
                while block:=f.read(8*1024**2):h.update(block)
            if h.hexdigest()!=wanted:raise ValueError('Downloaded file SHA mismatch')
        for row in iter_split(tp,ap,split=split,source=manifest['source'],original_file=item['original_final_file'],
                              inventory_id=manifest['content_inventory_id']):
            yield row;done+=1
            if max_documents is not None and done>=max_documents:return
