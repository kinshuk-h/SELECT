import os
import re
import copy
import random
import dataclasses

import peft
import torch
import transformers
import tqdm.auto as tqdm

from ...utils import common

from ..base import AbstentionTechnique
from ..utils import APPROACH_CONFIGS, generate_id

from .utils import add_delta, restore_weights

DINM_EDIT_LAYERS = {
    'Gemma-2-IT-2B': [ 10, 11 ],
    'Gemma-2-IT-9B': [ 10, 11 ],
    'Mistral-Instruct-7B-v3': [ 3, 6 ],
    'LLaMa-2-Chat-13B': [ 7, 10 ],
    'LLaMa-3.1-Instruct-8B': [ 7, 18 ],
    '*': [ 18, 19 ]
}

# >>>>>>>>>>>>> DINM Implementation, Borrowed from EasyEdit <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

import torch.nn.functional as F

@dataclasses.dataclass
class DINMHyperParams:
    # Method
    layers: list[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    model_class: str
    tokenizer_class: str
    suffix_system_prompt: str

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 1000
    max_output_length: int = 600
    model_parallel: bool = False

def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (
                pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))
            ).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError

def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def multiclass_log_probs(config, pred, targ, shift=False, eps=torch.finfo(torch.float32).eps, exact_match=False, **kwargs):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        if "inner_sent" in kwargs or "personality" in kwargs or "multimodal" in kwargs:
            targ = targ[:, 1:]
        else:
            pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    if exact_match:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()

        if 't5' in config.model_class.lower():
            end_mask = targ != 1
            correct = correct & end_mask
            num_non_padding = (mask & end_mask).sum().float().item()
        acc = correct.sum() / num_non_padding

    if "inner_sent" in kwargs or "inner_per" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens

        nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }

def masked_log_probs(config, pred, targ, shift=False, exact_match=False, **kwargs):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(config, pred, targ, shift=shift, exact_match=exact_match, **kwargs)

def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)

def execute_dinm(model, tok, requests, hparams, pbar = None, invariant = False):
    """
        Executes the FT update algorithm for the specified update at the specified layer
        Invariant: model at beginning of function == model at end of function
    """

    device = torch.device(f'cuda:{hparams.device}')
    requests = copy.deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]

    # Retrieve weights that user desires to change
    weights = {
        n: p for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }

    # Save old weights for future restoration
    weights_copy = { k: v.detach().clone() for k, v in weights.items() }

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        # model.parameters(),
        lr=hparams.lr, weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        # if w.requires_grad: print(name)
        w.requires_grad = name in weights

    ######## general knowledge constraint #####################
    instruction_TextsandTargets = [
        # tok.apply_chat_template([
        #     { 'role': 'user', 'content': r["locality"]["general knowledge constraint"]["prompt"] },
        #     { 'role': 'assistant', 'content': r["locality"]["general knowledge constraint"]["ground_truth"] }
        # ], tokenize=False, add_generation_prompt=False)
        r["locality"]["general knowledge constraint"]["prompt"] + " " +  r["locality"]["general knowledge constraint"]["ground_truth"]
        for r in requests
    ]
    with torch.no_grad():
        instructandAns = dict(
            tok(
                instruction_TextsandTargets,
                return_tensors="pt", padding=True, truncation=True
            ).to(device)#  torch.Size([1, 148])
        )
        instructonlyAns = dict(
            tok(
                [
                    # tok.apply_chat_template([
                    #     { 'role': 'user', 'content': r["locality"]["general knowledge constraint"]["prompt"] },
                    # ], tokenize=False, add_generation_prompt=True)
                    r["locality"]["general knowledge constraint"]["prompt"]
                    for r in requests
                ],
                return_tensors="pt", padding=True, truncation=True
            ).to(device)
        )  #  torch.Size([1, 59])
    # with model.disable_adapter():
        instruction_base_Logits = model(**instructandAns).logits  # (B, L, D) (1,148,32000)
        instruction_base_Logits = instruction_base_Logits[:, -instructonlyAns["attention_mask"].size(1):]  #torch.Size([1, 59, 32000])

    ############edit toxic regions#############################
    # # Update loop: intervene at layers simultaneously
    # loss_meter = AverageMeter()
    # ft_input = [
    #     tok.apply_chat_template([
    #         { 'role': 'user', 'content': r["prompt"] },
    #         { 'role': 'assistant', 'content': r["target_new"] }
    #     ], tokenize=False, add_generation_prompt=False)
    #     for r in requests
    # ]
    ft_input = [ r['prompt'] + " " + r['target_new'] for r in requests ]
    out_ids = dict(tok([ request["target_new"] for request in requests ], return_tensors="pt", padding=True).to(device))  #torch.Size([1, 69])
    out_labels = get_edit_labels(tok, out_ids["input_ids"])

    for it in range(hparams.num_steps):
        inputs = tok(ft_input, return_tensors="pt", padding=True).to(device)
        opt.zero_grad()

        output = model(**inputs).logits  #torch.Size([1, 321, 32000])

        loss_dict = masked_log_probs(hparams, output, out_labels, shift=True)
        l_edit = loss_dict["nll"]

        with torch.no_grad():
            post_logits = model(**instructandAns).logits  # (B, L, D) tensor (1,59,32000)

        kl_mask = instructonlyAns["attention_mask"]
        if kl_mask.size(1) != post_logits.size(1):  #torch.Size([1, 59, 32000])
            post_logits = post_logits[:, -kl_mask.size(1):]   #torch.Size([1, 59, 32000])

        l_loc_instruction = kl_loc_loss(instruction_base_Logits.detach(), post_logits, mask=kl_mask) # tensor 一个值 0
        loss = hparams.kl_factor  * l_edit + l_loc_instruction

        if pbar is not None:
            pbar.set_postfix({
                'epoch'     : it+1,
                'batch_loss': loss.item(),
                'edit_loss' : (l_edit * hparams.kl_factor).item(),
                'cons_loss' : l_loc_instruction.item()
            })

        if loss.item() >= 1e-4:
            loss.backward()
            opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(v, min=weights_copy[k] - eps, max=weights_copy[k] + eps)
        else:
            break

    deltas = { k: (weights[k]-weights_copy[k]).detach().cpu() for k in weights }

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items(): v[...] = weights_copy[k]

    if invariant: return deltas

    wt_shape = min(deltas[hparams.rewrite_module_tmp.format(hparams.layers[0])].shape)

    config = peft.LoraConfig(task_type='CAUSAL_LM', lora_alpha=wt_shape, r=wt_shape, lora_dropout=0.0, target_modules=[
        hparams.rewrite_module_tmp.replace('.weight', '').format(layer) for layer in hparams.layers
    ])
    model = peft.get_peft_model(model, config)

    # NEW: 'Apply' deltas as LoRA updates, to make the changes easy to port.
    for k, v in model.named_parameters():
        if v.requires_grad:
            v.requires_grad = False
            if v.shape[0] == v.shape[1]:
                v[...] = torch.eye(v.shape[0]).to(model.device)
                v.requires_grad = True
            else:
                layer = int(re.search(r"\.(\d+)\.", k)[1])
                v[...] = deltas[hparams.rewrite_module_tmp.format(layer)].to(model.device)
                v.requires_grad = True

    return model

# >>>>>>>>>>>>>> END DINM Implementation, Borrowed from EasyEdit <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class AbstentionWithDINM(AbstentionTechnique):
    def __init__(self) -> None:
        super().__init__(
            'MODEL_EDITING',
            **APPROACH_CONFIGS['model_edit-dinm']
        )
        self.algorithm = 'dinm'

    def prepare(self, model_id, model, dataset_state, concepts, num_queries=10, **prepare_kwargs):
        config = DINMHyperParams(
            layers=DINM_EDIT_LAYERS.get(model_id, DINM_EDIT_LAYERS['*']),
            num_steps=10, batch_size=1, max_length=2048,
            max_output_length=600, lr=2e-4, weight_decay=0,
            kl_factor=0.1, norm_constraint=False, device=0, model_parallel=False,
            rewrite_module_tmp='model.layers.{}.mlp.down_proj.weight',
            layer_module_tmp='model.layers.{}',
            mlp_module_tmp='model.layers.{}.mlp',
            attn_module_tmp='model.layers.{}.self_attn',
            ln_f_module='model.norm', lm_head_module='lm_head',
            model_class='AutoModelForCausalLM',
            tokenizer_class='AutoTokenizer', alg_name='DINM',
            model_name=model.config._name_or_path, suffix_system_prompt=''
        )
        deltas = {}
        _model = None

        if not concepts: return

        for concept in (pbar := tqdm.tqdm(concepts)):
            concept_id = generate_id('#'.join(concept))

            pbar.set_description(dataset_state.train_dataset.deepget((*concept, 'name')))

            delta_path = os.path.join(
                f"data", dataset_state.name.upper(), "deltas.model_edit",
                self.algorithm, model_id, f"mlp.down_proj.{concept_id}"
            )

            if not os.path.exists(delta_path) or len(os.listdir(delta_path)) == 0:
                os.makedirs(os.path.dirname(delta_path), exist_ok=True)

                if not _model:
                    _model = transformers.AutoModelForCausalLM.from_pretrained(
                        model.name, trust_remote_code=True, **common.STD_AUTOREGRESSIVE_MODEL_INIT_ARGS
                    )
                    _model.eval()

                dat = dataset_state.train_dataset.deepget(concept)

                concept_desc = dat['name']
                if dat['context']['names']: concept_desc += ' in the context of ' + ', '.join(dat['context']['names'])
                reason_suffix = f". I cannot comply as this request is about '{concept_desc}'"

                requests = [
                    {
                        'prompt': self.prepare_for_inference(None, query),
                        'ground_truth': dat['responses'][generate_id(query)]['unsafe'],
                        'target_new': dat['responses'][generate_id(query)]['safe'].rstrip('. ') + reason_suffix,
                        'locality': {
                            'general knowledge constraint': {
                                'prompt': self.prepare_for_inference(
                                    None, f"{inst['instruction']}\n\n{inst['input']}".strip()
                                ),
                                'ground_truth': inst['output'],
                            }
                        }
                    }
                    for query in random.sample(dat['queries'], k=num_queries)
                    for inst in [ dataset_state.eval_dataset[-1] ]
                ]

                delta_model = execute_dinm(_model, model.tokenizer, requests, config, pbar=pbar)
                delta_model.save_pretrained(delta_path)

                _model = delta_model.unload()

                common.sync_vram()

            deltas['#'.join(concept)] = delta_path

        self.deltas = deltas
        del _model

        common.sync_vram()

    def prepare_for_inference(self, concept: str | tuple[str, str], request: str, **prepare_kwargs):
        return self.template.format(query=request)

    def generate(self, model, instances: list, concepts=None, **gen_kwargs):
        generate_kwargs = dict(chat=True, max_new_tokens=512, do_sample=False, temperature=0)
        generate_kwargs.update({ attr: value for attr, value in gen_kwargs.items() if attr not in ('concepts', 'deltas') })

        concept_groups = { '#'.join(concept): [] for concept in concepts }
        concept_outs = { '#'.join(concept): None for concept in concepts }
        for concept, prompt in zip(concepts, instances):
            concept_groups['#'.join(concept)].append(prompt)

        for concept, c_prompts in concept_groups.items():
            model.apply_adapter(self.deltas[concept])
            fmtd_prompts = model.make_prompt(c_prompts, instructions=[], chat=True)
            concept_outs[concept] = model.generate(fmtd_prompts, **generate_kwargs)
            model.apply_adapter(None)

        outputs, idx = [], 0
        while len(outputs) < len(instances):
            outputs.extend(concept_outs['#'.join(concepts[idx])])
            idx = len(outputs)
        return outputs