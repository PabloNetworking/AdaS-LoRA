"""Synchronized task-head federation for AdaS and AdaS-LayerAware.

AdaS aggregators federate LoRA/component updates as DELTAS that the server ADDS. The
trainable classification head (classifier + pooler) is federated here, like HetLoRA/FedAvg,
and SEPARATELY from the LoRA delta path:

  * head params are ABSOLUTE (not deltas): sample-size FedAvg over the clients' uploaded
    head tensors, then the server REPLACES the model head (never adds).
  * strict key-consumption: every head key the clients uploaded must map to a model param.
  * round-start tensor-equality: after the download, each client's head must equal the
    broadcast global head (verified by the two-round federation test).

Head keys = trainable, non-LoRA params (classifier + pooler). We identify them by exclusion
(not lora_A/lora_B) among the client's UPLOADED trainable params (the frozen backbone is
never uploaded), matching hetlora_aggregator._average_non_lora_trainable.
"""
import torch

from federatedscope.core.auxiliaries.utils import param2tensor


def is_head_key(k):
    """A trainable non-LoRA (task-head) parameter key."""
    return isinstance(k, str) and ('lora_A' not in k) and ('lora_B' not in k)


def extract_head_params(state):
    """The head (non-LoRA) subset of a client's uploaded param dict."""
    return {k: v for k, v in state.items() if is_head_key(k)}


def average_head_params(models, device='cpu'):
    """Sample-size-weighted ABSOLUTE average of head params.

    models: list of (sample_size, params_dict). params_dict may hold the full upload; only
    its non-LoRA (head) keys are averaged. Returns {head_key: absolute averaged tensor}.
    """
    if not models:
        return {}
    total = float(sum(float(ss) for ss, _ in models))
    if total <= 0:
        total = float(len(models))
    # union of head keys across clients (robust to a client missing one)
    keys = []
    seen = set()
    for _, mp in models:
        for k in mp:
            if is_head_key(k) and k not in seen:
                seen.add(k)
                keys.append(k)
    out = {}
    for k in keys:
        acc = None
        for ss, mp in models:
            v = mp.get(k)
            if v is None:
                continue
            t = param2tensor(v)
            if not isinstance(t, torch.Tensor):
                continue
            t = t.to(device).float() * (float(ss) / total)
            acc = t if acc is None else acc + t
        if acc is not None:
            out[k] = acc
    return out


def replace_model_head(model, head_avg, strict=True):
    """REPLACE (not add) the model's head params with the absolute averaged head.

    Runs in the server postprocess BEFORE the LoRA-only delta merge, so the merge never
    touches these keys. With strict=True, every averaged head key must exist in the model
    (strict key-consumption): an unmapped head key raises instead of leaving the head
    unfederated.
    """
    if not head_avg:
        return 0
    sd = model.state_dict()
    applied = 0
    with torch.no_grad():
        for k, v in head_avg.items():
            if k not in sd:
                if strict:
                    raise KeyError(
                        f"[head-fed] averaged head key '{k}' absent from model state_dict "
                        f"(strict key-consumption failed)")
                continue
            tgt = sd[k]
            src = param2tensor(v).to(tgt.device, tgt.dtype)
            if tuple(src.shape) != tuple(tgt.shape):
                if strict:
                    raise ValueError(
                        f"[head-fed] head key '{k}' shape {tuple(src.shape)} != model "
                        f"{tuple(tgt.shape)}")
                continue
            tgt.copy_(src)
            applied += 1
    return applied


def head_keys_from_model(model):
    """STATE_DICT keys of the trainable non-LoRA (head) params -- what to upload / broadcast /
    replace. Must return state_dict() keys (not named_parameters() names): a wrapper module
    (e.g. GLUEAdapterModel: self.model = peft_model) prefixes named_parameters with 'model.'
    while its state_dict() strips it, so the two key spaces differ. We map each trainable
    non-LoRA named param to its state_dict key (identity, or with a leading 'model.' removed).
    Using named_parameters() names here would match no state_dict key and drop the head."""
    sd = model.state_dict()
    out = []
    for n, p in model.named_parameters():
        if not (getattr(p, 'requires_grad', False) and is_head_key(n)):
            continue
        for cand in (n, n[len('model.'):] if n.startswith('model.') else None):
            if cand and cand in sd:
                out.append(cand)
                break
    return out
