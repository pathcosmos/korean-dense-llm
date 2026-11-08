"""Native base-model multiple-choice evaluation on explicitly versioned local tasks.
Task JSON: schema_version=1, name, revision, template (with {question}, optional
{choices}), target=choice_text|label, scoring=sum|mean. JSONL records:
{id, question, choices:[text,...], answer:zero-based integer}. No implicit benchmark defaults.
Full concatenated encoding must preserve the prompt token prefix; unstable boundaries
are rejected instead of scoring separately encoded text or silently changing context.
"""
import argparse
import json
import math
from pathlib import Path
import string
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from common.data_contracts import resolve,identity,sha256_file,atomic_json
from eval.run import load_checkpoint


def validate_task(task):
    if task.get('schema_version')!=1 or any(not isinstance(task.get(k),str) or not task[k].strip() for k in ['name','revision','template']):
        raise ValueError('Task requires schema_version=1 and explicit name/revision/template')
    if task.get('target') not in ('choice_text','label') or task.get('scoring') not in ('sum','mean'):
        raise ValueError('Task requires explicit target and scoring rule')
    fields=[]
    for _,field,spec,conversion in string.Formatter().parse(task['template']):
        if field is not None:
            if field not in ('question','choices') or spec or conversion:raise ValueError('Unsupported template field')
            fields.append(field)
    if fields.count('question')!=1 or fields.count('choices')>1:raise ValueError('Template must contain one question')
    if task['target']=='label' and 'choices' not in fields:raise ValueError('Label task must display choices')


def adapt(row,task):
    validate_task(task)
    if not isinstance(row.get('id'),str) or not row['id'].strip():raise ValueError('Require example ID')
    if not isinstance(row.get('question'),str) or not row['question'].strip():raise ValueError('Require question')
    choices=row.get('choices')
    if not isinstance(choices,list) or not 2<=len(choices)<=26 or any(not isinstance(c,str) or not c.strip() for c in choices):
        raise ValueError('Require 2..26 nonempty choices')
    if len(set(choices))!=len(choices):raise ValueError('Duplicate choices')
    answer=row.get('answer')
    if type(answer) is not int or not 0<=answer<len(choices):raise ValueError('Answer must be a zero-based index')
    labels=list(string.ascii_uppercase[:len(choices)])
    prompt=task['template'].format(question=row['question'],choices='\n'.join(f'{label}. {text}' for label,text in zip(labels,choices)))
    return dict(id=row['id'],prompt=prompt,choices=labels if task['target']=='label' else choices,answer=answer)


@torch.no_grad()
def score_choices(model,tok,prompt,choices,*,scoring,context_length,device='cpu'):
    if scoring not in ('sum','mean') or type(context_length) is not int or context_length<1:raise ValueError('Invalid scoring/context')
    prefix=tok.encode(prompt,add_special_tokens=False).ids
    if not prefix:raise ValueError('Prompt must encode to at least one token')
    if len(choices)<2 or any(not isinstance(c,str) or not c for c in choices):raise ValueError('Require nonempty choices')
    result=[];was_training=model.training;model.eval()
    try:
        for choice in choices:
            full=tok.encode(prompt+choice,add_special_tokens=False).ids
            if full[:len(prefix)]!=prefix:raise ValueError('Unstable prompt/choice token boundary; revise task template')
            count=len(full)-len(prefix)
            if count<1:raise ValueError('Choice must encode to at least one target')
            if len(full)-1>context_length:raise ValueError('Choice exceeds context; no silent truncation')
            ids=torch.tensor([full],dtype=torch.long,device=device)
            logits=model(ids[:,:-1]).float()
            scored=logits[:,len(prefix)-1:,:]
            targets=ids[:,len(prefix):]
            logprobs=scored.log_softmax(-1).gather(-1,targets[...,None]).squeeze(-1)
            total=float(logprobs.sum());score=total if scoring=='sum' else total/count
            if not math.isfinite(score):raise ValueError('Nonfinite choice score')
            result.append(dict(log_probability=total,target_tokens=count,score=score))
    finally:model.train(was_training)
    best=max(r['score'] for r in result);tied=[i for i,r in enumerate(result) if r['score']==best]
    return dict(choices=result,prediction=tied[0],tied_indices=tied,tie_policy='first-index')


def run(checkpoint,tokenizer_path,task_path,examples_path,output,*,device='cpu'):
    checkpoint,tokenizer_path,task_path,examples_path,output=map(Path,(checkpoint,tokenizer_path,task_path,examples_path,output))
    task=json.loads(task_path.read_text());validate_task(task)
    rows=[adapt(json.loads(line),task) for line in examples_path.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Require nonempty task with unique example IDs')
    if not (checkpoint/'.complete').is_file():raise ValueError('Checkpoint is incomplete')
    token_sha=sha256_file(tokenizer_path)
    contract=dict(version=1,task=task,task_sha256=sha256_file(task_path),examples_sha256=sha256_file(examples_path),
        tokenizer_sha256=token_sha,checkpoint={str(p.relative_to(checkpoint)):sha256_file(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()},
        code={p:sha256_file(resolve(p)) for p in ['eval/multiple_choice.py','eval/run.py','model/model.py']},
        device=device,torch_version=str(torch.__version__),append_eos=False,token_boundary='require-stable-prefix',tie_policy='first-index')
    cid=identity(contract);marker=Path(str(output)+'.done')
    if output.exists():
        old=json.loads(output.read_text())
        if old.get('contract_id')!=cid:raise ValueError('Evaluation identity changed; use a new report path')
        if not marker.exists() or json.loads(marker.read_text())!=dict(contract_id=cid,sha256=sha256_file(output)):
            raise ValueError('Uncertified evaluation report')
        return old
    tok=Tokenizer.from_file(str(tokenizer_path));model,meta=load_checkpoint(checkpoint,token_sha,device)
    if not tok.get_vocab() or max(tok.get_vocab().values())>=model.c.vocab_size:raise ValueError('Tokenizer vocabulary incompatible')
    results=[]
    for row in rows:
        result=score_choices(model,tok,row['prompt'],row['choices'],scoring=task['scoring'],context_length=model.c.max_seq_len,device=device)
        results.append(dict(id=row['id'],answer=row['answer'],correct=result['prediction']==row['answer'],**result))
    correct=sum(r['correct'] for r in results)
    report=dict(status='passed',contract_id=cid,contract=contract,checkpoint_step=meta['step'],
        examples=len(rows),correct=correct,accuracy=correct/len(rows),results=results,
        scope='Score on supplied versioned local task only; no claim of official benchmark equivalence')
    atomic_json(output,report);atomic_json(marker,dict(contract_id=cid,sha256=sha256_file(output)))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ['checkpoint','tokenizer','task','examples','output']:ap.add_argument('--'+field,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=ap.parse_args()
    r=run(*[resolve(getattr(a,f)) for f in ['checkpoint','tokenizer','task','examples','output']],device=a.device)
    print(json.dumps({k:r[k] for k in ['status','examples','correct','accuracy']}))
