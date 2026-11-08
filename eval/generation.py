"""Fixed-seed generation through the current native-model API, no HF export needed."""
import math
import torch


@torch.no_grad()
def generate(model, prompt_ids, *, eos_id, context_length, max_new_tokens, seed=0, temperature=0.0, device='cpu'):
    if not prompt_ids or type(context_length) is not int or context_length < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('Require nonempty prompt and positive context/token limits')
    if type(seed) is not int or seed < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Invalid generation seed/temperature')
    if any(type(x) is not int or x < 0 for x in prompt_ids) or type(eos_id) is not int or eos_id < 0:
        raise ValueError('Invalid prompt/EOS IDs')
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = list(prompt_ids);new = [];stopped = False
    training = model.training;model.eval()
    try:
        for _ in range(max_new_tokens):
            context = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
            logits = model(context)[0, -1].float()
            if eos_id >= len(logits) or not torch.isfinite(logits).all():
                raise ValueError('Invalid EOS or nonfinite logits')
            if temperature == 0:
                token = int(logits.argmax().item())
            else:
                token = int(torch.multinomial(torch.softmax(logits/temperature, dim=-1), 1, generator=generator).item())
            ids.append(token);new.append(token)
            if token == eos_id:
                stopped = True;break
    finally:
        model.train(training)
    return dict(prompt_ids=list(prompt_ids), generated_ids=new, stopped_at_eos=stopped,
                seed=seed, temperature=temperature, context_length=context_length, max_new_tokens=max_new_tokens)
