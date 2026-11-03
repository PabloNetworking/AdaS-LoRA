"""Tests for the synchronized task-head federation used by AdaS.

Covers four properties of the head federation:
  * ABSOLUTE sample-size average (not a delta),
  * server REPLACE (never add),
  * STRICT key-consumption (unmapped head key raises),
  * two-round federation: after aggregation all clients' heads are EQUAL at round start
    (round-start tensor-equality) and re-converge each round.

Run: python -m pytest \
     federatedscope/contrib/common/test_head_federation.py -q
"""
import copy
import torch
import torch.nn as nn

from federatedscope.contrib.common.head_federation import (
    is_head_key, extract_head_params, average_head_params,
    replace_model_head, head_keys_from_model,
)


class TinyHeadModel(nn.Module):
    """A frozen 'backbone', a trainable LoRA param (excluded from the head), and a trainable
    task head (classifier + pooler)."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 2)
        self.pooler = nn.Linear(4, 4)
        self.enc_lora_A = nn.Parameter(torch.randn(8, 4))   # name has 'lora_A' -> not head
        self.backbone = nn.Linear(4, 4)
        for p in self.backbone.parameters():
            p.requires_grad = False


def _upload_head(model):
    sd = model.state_dict()
    return {k: sd[k].detach().clone() for k in head_keys_from_model(model) if k in sd}


def _download_head(model, head):
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in head.items():
            sd[k].copy_(v)


def test_head_keys_exclude_lora_and_frozen():
    m = TinyHeadModel()
    keys = set(head_keys_from_model(m))
    assert 'classifier.weight' in keys and 'pooler.weight' in keys
    assert not any('lora_A' in k for k in keys)      # LoRA excluded
    assert not any(k.startswith('backbone') for k in keys)  # frozen excluded


def test_average_is_absolute_sample_size_weighted():
    # heads = constant tensors 1.0, 2.0, 3.0 with sample sizes 1, 1, 2 -> weighted mean 2.25
    models = []
    for val, ss in [(1.0, 1), (2.0, 1), (3.0, 2)]:
        h = {'classifier.weight': torch.full((2, 4), val),
             'pooler.weight': torch.full((4, 4), val)}
        models.append((ss, h))
    avg = average_head_params(models)
    assert torch.allclose(avg['classifier.weight'], torch.full((2, 4), 2.25))
    assert torch.allclose(avg['pooler.weight'], torch.full((4, 4), 2.25))


def test_replace_not_add():
    m = TinyHeadModel()
    with torch.no_grad():
        m.classifier.weight.fill_(5.0)      # arbitrary prior value
    avg = {'classifier.weight': torch.zeros(2, 4),
           'pooler.weight': torch.zeros(4, 4)}
    replace_model_head(m, avg, strict=True)
    # REPLACE: head becomes exactly the average (0), NOT prior(5)+avg(0)
    assert torch.count_nonzero(m.classifier.weight) == 0
    assert torch.count_nonzero(m.pooler.weight) == 0


def test_strict_key_consumption_raises_on_unmapped():
    m = TinyHeadModel()
    bad = {'classifier.weight': torch.zeros(2, 4), 'nonexistent.head': torch.zeros(3)}
    try:
        replace_model_head(m, bad, strict=True)
    except KeyError:
        return
    raise AssertionError("strict replace must raise on an unmapped head key")


def test_two_round_federation_convergence():
    torch.manual_seed(0)
    server = TinyHeadModel()
    clients = [copy.deepcopy(server) for _ in range(3)]
    ss = [1, 1, 2]

    def local_head_drift(m, k):          # simulate divergent local head training
        with torch.no_grad():
            for p in [m.classifier.weight, m.pooler.weight]:
                p.add_(torch.randn_like(p) * 0.1 * (k + 1))

    prev_heads = None
    for rnd in range(2):
        # round-start tensor-equality: from round 2 on, every client head must equal the
        # previous round's federated average (they were all broadcast the same head).
        if rnd >= 1:
            for c in clients:
                for k, v in _upload_head(c).items():
                    assert torch.allclose(v, prev_heads[k], atol=1e-6), \
                        f"round-start head mismatch @round {rnd} key {k}"
        # local training diverges the heads
        for k, c in enumerate(clients):
            local_head_drift(c, k)
        # aggregate (absolute sample-size) + server replace + broadcast
        drifted = [(ss[i], _upload_head(clients[i])) for i in range(3)]
        avg = average_head_params(drifted)
        # cross-check against an explicit weighted mean of the drifted heads
        tot = float(sum(ss))
        for k in avg:
            man = sum(s * d[k] for s, d in drifted) / tot
            assert torch.allclose(avg[k], man, atol=1e-6)
        replace_model_head(server, avg, strict=True)
        for c in clients:
            _download_head(c, {k: server.state_dict()[k] for k in avg})
        prev_heads = {k: server.state_dict()[k].detach().clone() for k in avg}

    # after 2 rounds all clients share the server head exactly
    for c in clients:
        for k, v in _upload_head(c).items():
            assert torch.allclose(v, server.state_dict()[k], atol=1e-6)
