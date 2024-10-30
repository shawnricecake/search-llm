import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange


@torch.no_grad()
def evaluate(model, testenc, device, full_eval=False):
    loss_fct = nn.CrossEntropyLoss()

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    # for i in range(nsamples):
    for i in tqdm(range(nsamples), desc='Evaluating'):
        if not full_eval and i > 4:
            break
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)

        outputs = model(batch)
        lm_logits = outputs.logits

        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        shift_labels = shift_labels.view(-1)
        shift_logits = lm_logits[:, :-1, :].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    if full_eval:
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    else:
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * model.seqlen))

    torch.cuda.empty_cache()
    return ppl

