"""Tests for the federated LoRA download path.

The server sends the aggregated global adapter with rank-annotated keys
(``base_model...lora_A.default.weight.<rank>``), while the client model's state_dict keys
carry no rank suffix, so ``load_state_dict(strict=False)`` alone would ignore them.
``heterolora_utils.apply_distributed_lora_download`` maps each rank-annotated key onto the
model's state_dict key (inferring each key's rank from the key itself, so the per-module
HetLoRA and per-layer AdaS formats both work), pads/truncates the rank dimension, passes
non-LoRA keys through unchanged, and raises KeyError under ``strict=True`` if any LoRA key
cannot be matched.

This module contains:
  (A) Fast, CPU, deterministic unit tests of ``apply_distributed_lora_download``.
  (B) Two-round end-to-end checks that the aggregated global adapter reaches (and changes)
      the next client update:
        - ``test_two_round_federation_download`` (always runs, light, mock model) shows that a
          plain strict=False load ignores the rank-annotated keys and that the re-keyed
          payload is loaded.
        - ``test_two_round_real_hetlora_fl_slow`` (gated behind ADAS_RUN_SLOW_TESTS=1, name
          contains "slow") runs a minimal 2-round HetLoRA simulation with
          federate.assert_download_tensor_equality=True and asserts exit 0.

Run (fast unit + light e2e only):
  python -m pytest \
      federatedscope/contrib/common/test_federation_download.py -q -k "not slow" -p no:cacheprovider
Or:
  python federatedscope/contrib/common/test_federation_download.py
"""
import os
import sys

import torch

from federatedscope.contrib.common.heterolora_utils import (
    apply_distributed_lora_download,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class MockModel:
    """Minimal stand-in exposing only ``.state_dict()`` (all that the canonicalizer needs)."""

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self._state.items()}


class MockLoRAModel:
    """Mock model reproducing torch ``load_state_dict(strict=False)`` semantics.

    ``load_state_dict`` copies (in place) only values whose key already exists in the model;
    keys present in the incoming dict but absent from the model (e.g. the distributed rank-suffixed
    keys the server sends) are ignored, which is why the payload is re-keyed first.
    """

    def __init__(self, state):
        self._state = {k: v.clone() for k, v in state.items()}

    def state_dict(self):
        return {k: v.clone() for k, v in self._state.items()}

    def load_state_dict(self, incoming, strict=False):
        unexpected = []
        for k, v in incoming.items():
            if k in self._state:
                self._state[k].copy_(torch.as_tensor(v))
            else:
                unexpected.append(k)
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")
        return unexpected


_PFX = "base_model.model.deberta.encoder.layer.0"
KA = f"{_PFX}.attention.self.in_proj.lora_A.default.weight"
KB = f"{_PFX}.attention.self.in_proj.lora_B.default.weight"
KA2 = f"{_PFX}.attention.output.dense.lora_A.default.weight"
KB2 = f"{_PFX}.attention.output.dense.lora_B.default.weight"
KBASE = f"{_PFX}.attention.self.in_proj.base_layer.weight"  # non-LoRA passthrough witness


def _model_state(rank=8, din=32, dout=32):
    """A tiny state_dict with two LoRA modules at ``rank`` plus one non-LoRA base weight."""
    return {
        KA: torch.zeros(rank, din),
        KB: torch.zeros(dout, rank),
        KA2: torch.zeros(rank, din),
        KB2: torch.zeros(dout, rank),
        KBASE: torch.randn(dout, din),
    }


def _distributed_agg(rank=8, din=32, dout=32, seed=0):
    """A distributed-format aggregated global adapter (keys carry a ``.<rank>`` suffix)."""
    g = torch.Generator().manual_seed(seed)
    return {
        f"{KA}.{rank}": torch.randn(rank, din, generator=g),
        f"{KB}.{rank}": torch.randn(dout, rank, generator=g),
        f"{KA2}.{rank}": torch.randn(rank, din, generator=g),
        f"{KB2}.{rank}": torch.randn(dout, rank, generator=g),
    }


# --------------------------------------------------------------------------- #
# (A) UNIT TESTS -- fast, CPU, deterministic
# --------------------------------------------------------------------------- #
def test_canonicalization_to_model_keys():
    """Distributed rank-suffixed keys are re-keyed to the model's suffix-free state_dict keys,
    tensors preserved, and consume counts reported."""
    model = MockModel(_model_state(rank=8))
    agg = _distributed_agg(rank=8, seed=1)

    out, n_consumed, n_model = apply_distributed_lora_download(agg, model, strict=True)

    assert n_consumed == 4 and n_model == 4
    # Output is re-keyed to the MODEL's keys (no rank suffix); the distributed keys are gone.
    assert set(out.keys()) == {KA, KB, KA2, KB2}
    for dk in agg:
        assert dk not in out
    # Values are carried through untouched (shapes already match at equal rank).
    assert torch.equal(out[KA], agg[f"{KA}.8"])
    assert torch.equal(out[KB], agg[f"{KB}.8"])


def test_canonicalization_aligns_model_prefix():
    """A model whose state_dict keys carry a leading 'model.' (AdapterModel wrapper) still
    matches the server's un-prefixed distributed keys, and output uses the model's own keys."""
    wrapped = {"model." + KA: torch.zeros(8, 32), "model." + KB: torch.zeros(32, 8)}
    model = MockModel(wrapped)
    agg = {f"{KA}.8": torch.randn(8, 32), f"{KB}.8": torch.randn(32, 8)}

    out, n_consumed, n_model = apply_distributed_lora_download(agg, model, strict=True)

    assert n_consumed == 2 and n_model == 2
    assert set(out.keys()) == {"model." + KA, "model." + KB}
    assert torch.equal(out["model." + KA], agg[f"{KA}.8"])


def test_strict_raises_on_unmapped_distributed_key():
    """strict=True raises KeyError when a distributed LoRA key has no matching model param
    (otherwise the key would be ignored by a strict=False load)."""
    model = MockModel(_model_state(rank=8))
    agg = _distributed_agg(rank=8, seed=2)
    # A distributed LoRA key for a layer the model does not have.
    ghost = "base_model.model.deberta.encoder.layer.99.attention.self.in_proj.lora_A.default.weight.8"
    agg[ghost] = torch.randn(8, 32)

    raised = False
    try:
        apply_distributed_lora_download(agg, model, strict=True)
    except KeyError:
        raised = True
    assert raised, "strict=True must raise KeyError on an unmapped distributed LoRA key"


def test_non_strict_skips_unmapped_key():
    """strict=False does not raise; the unmapped distributed key is dropped from the output
    while the mappable keys are still consumed."""
    model = MockModel(_model_state(rank=8))
    agg = _distributed_agg(rank=8, seed=3)
    ghost = "base_model.model.deberta.encoder.layer.99.attention.self.in_proj.lora_A.default.weight.8"
    agg[ghost] = torch.randn(8, 32)

    out, n_consumed, n_model = apply_distributed_lora_download(agg, model, strict=False)

    assert n_consumed == 4 and n_model == 4
    assert ghost not in out
    assert set(out.keys()) == {KA, KB, KA2, KB2}


def test_non_distributed_payload_passthrough_identity():
    """A payload with NO rank-suffixed keys is not a distributed download: it is returned
    unchanged (same object) with zero consumption."""
    model = MockModel(_model_state(rank=8))
    content = {
        KA: torch.randn(8, 32),           # LoRA key but NO rank suffix -> not distributed
        "classifier.weight": torch.randn(2, 32),
        "classifier.bias": torch.randn(2),
    }

    out, n_consumed, n_model = apply_distributed_lora_download(content, model, strict=True)

    assert out is content
    assert n_consumed == 0 and n_model == 0


def test_non_lora_keys_passthrough_within_distributed_payload():
    """Inside a distributed payload, non-LoRA keys (e.g. a classifier head) pass through
    unchanged while the LoRA keys are re-keyed."""
    model = MockModel(_model_state(rank=8))
    agg = _distributed_agg(rank=8, seed=4)
    head = torch.randn(2, 32)
    agg["classifier.weight"] = head

    out, n_consumed, n_model = apply_distributed_lora_download(agg, model, strict=True)

    assert n_consumed == 4
    assert "classifier.weight" in out
    assert out["classifier.weight"] is head  # untouched, same object


def test_rank_pad_lora_A():
    """lora_A rank is dim 0: download rank (4) < model rank (8) -> pad rows, zero the tail."""
    model = MockModel(_model_state(rank=8, din=32))
    dl = torch.randn(4, 32)
    agg = {f"{KA}.4": dl}

    out, _, _ = apply_distributed_lora_download(agg, model, strict=True)

    res = out[KA]
    assert tuple(res.shape) == (8, 32)
    assert torch.equal(res[:4], dl)
    assert torch.count_nonzero(res[4:]) == 0


def test_rank_truncate_lora_A():
    """lora_A rank is dim 0: download rank (16) > model rank (8) -> keep leading 8 rows."""
    model = MockModel(_model_state(rank=8, din=32))
    dl = torch.randn(16, 32)
    agg = {f"{KA}.16": dl}

    out, _, _ = apply_distributed_lora_download(agg, model, strict=True)

    res = out[KA]
    assert tuple(res.shape) == (8, 32)
    assert torch.equal(res, dl[:8])


def test_rank_pad_lora_B():
    """lora_B rank is dim 1: download rank (4) < model rank (8) -> pad cols, zero the tail."""
    model = MockModel(_model_state(rank=8, dout=32))
    dl = torch.randn(32, 4)
    agg = {f"{KB}.4": dl}

    out, _, _ = apply_distributed_lora_download(agg, model, strict=True)

    res = out[KB]
    assert tuple(res.shape) == (32, 8)
    assert torch.equal(res[:, :4], dl)
    assert torch.count_nonzero(res[:, 4:]) == 0


def test_rank_truncate_lora_B():
    """lora_B rank is dim 1: download rank (16) > model rank (8) -> keep leading 8 cols."""
    model = MockModel(_model_state(rank=8, dout=32))
    dl = torch.randn(32, 16)
    agg = {f"{KB}.16": dl}

    out, _, _ = apply_distributed_lora_download(agg, model, strict=True)

    res = out[KB]
    assert tuple(res.shape) == (32, 8)
    assert torch.equal(res, dl[:, :8])


# --------------------------------------------------------------------------- #
# (B) TWO-ROUND END-TO-END FEDERATION ASSERTION
# --------------------------------------------------------------------------- #
def _lora_snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()
            if 'lora_A' in k or 'lora_B' in k}


def _round_start_tensor_equality(model, applied_download):
    """Mirror of Client._maybe_assert_download_applied: after loading the aggregated global
    adapter, every applied LoRA key must exist in the model and match tensor-for-tensor."""
    sd = model.state_dict()
    for k, v in applied_download.items():
        assert k in sd, f"applied download key {k} absent from model after load"
        assert torch.allclose(sd[k].float(), torch.as_tensor(v).float(), atol=1e-4), (
            f"model LoRA {k} does not match the applied global adapter after round-start load")


def test_two_round_federation_download():
    """LIGHT end-to-end proof (mock model, always runs): the aggregated global adapter reaches
    and CHANGES the next client update every round; a plain load without re-keying does not.

    Round r:
      1. server produces a distinct distributed-format aggregate G_r,
      2. plain path -- loading G_r's rank-suffixed keys directly leaves the model untouched,
      3. re-keyed path -- apply_distributed_lora_download re-keys G_r; strict=False load then makes
         the model's LoRA EQUAL G_r (the round-start tensor-equality assertion holds),
      4. local "training" perturbs the adapter before the next round.
    Because G_1 != G_2, the round-2 starting point differs from round-1 -> the aggregate
    demonstrably affects the next update.
    """
    torch.manual_seed(0)
    init = _model_state(rank=8)
    round_start_lora = []

    for rnd, seed in enumerate((11, 22), start=1):
        agg = _distributed_agg(rank=8, seed=seed)

        # (2) Plain path: rank-annotated keys never match the model's suffix-free keys.
        plain = MockLoRAModel(init)
        before = _lora_snapshot(plain)
        plain.load_state_dict(agg, strict=False)          # no re-keying
        after = _lora_snapshot(plain)
        for k in before:
            assert torch.equal(before[k], after[k]), (
                f"[round {rnd}] the plain load unexpectedly modified {k}")

        # (3) Re-keyed path: re-key, then a strict=False load applies the aggregate.
        rekeyed = MockLoRAModel(init)
        applied, n_consumed, n_model = apply_distributed_lora_download(
            agg, rekeyed, strict=True)
        assert n_consumed == n_model == 4
        rekeyed.load_state_dict(applied, strict=False)
        applied_lora = {k: v for k, v in applied.items() if 'lora_A' in k or 'lora_B' in k}
        _round_start_tensor_equality(rekeyed, applied_lora)  # aggregate really landed

        round_start_lora.append(_lora_snapshot(rekeyed))

        # (4) Local training perturbs the adapter -> next round aggregates something new.
        init = {k: (v + 0.05 * torch.randn_like(v) if ('lora_A' in k or 'lora_B' in k) else v)
                for k, v in rekeyed.state_dict().items()}

    # The two rounds started the client from DIFFERENT global adapters: federation is live.
    changed = any(not torch.equal(round_start_lora[0][k], round_start_lora[1][k])
                  for k in round_start_lora[0])
    assert changed, "round-2 client start identical to round-1 -> aggregate did not affect update"


def test_two_round_real_hetlora_fl_slow():
    """REAL 2-round HetLoRA standalone FL with federate.assert_download_tensor_equality=True.

    Gated behind ADAS_RUN_SLOW_TESTS=1 (loads deberta-large; needs a free GPU). The
    round-start tensor-equality assertion runs inside the client every round, so a clean exit 0
    shows the aggregated global adapter is applied each round; otherwise the assertion raises and
    the process exits non-zero.
    """
    if os.environ.get("ADAS_RUN_SLOW_TESTS") != "1":
        _skip("set ADAS_RUN_SLOW_TESTS=1 to run the real 2-round HetLoRA FL (deberta-large + GPU)")

    import subprocess
    import tempfile

    repo = os.environ.get(
        "ADAS_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    py = sys.executable
    gpu = os.environ.get("ADAS_TEST_GPU", "0")
    outdir = tempfile.mkdtemp(prefix="fed_dl_test_", dir=os.environ.get("TMPDIR", "/tmp"))

    cmd = [
        py, "federatedscope/main.py",
        "--cfg", "configs/hetlora.yaml",
        "device", gpu,
        "federate.client_num", "2",
        "federate.total_round_num", "2",
        "federate.assert_download_consumed", "True",
        "federate.assert_download_tensor_equality", "True",
        "train.local_update_steps", "1",
        "glue.max_length", "64",
        "glue.adapter.max_rank", "16",
        "glue.adapter.hetlora.rank_max", "16",
        "glue.adapter.hetlora.init_rank", "8",
        "glue.adapter.hetero_strategy", "homo",
        "data.type", "sst2@glue",
        "eval.freq", "2",
        "outdir", outdir,
    ]
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
        raise AssertionError(
            f"real 2-round HetLoRA FL exited {proc.returncode} "
            f"(tensor-equality assertion or run failed). Tail:\n{tail}")


# --------------------------------------------------------------------------- #
# __main__ runner
# --------------------------------------------------------------------------- #
def _skip(msg):
    """Skip under pytest; raise a sentinel the plain runner recognizes otherwise."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise _Skipped(msg)


class _Skipped(Exception):
    pass


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except BaseException as e:  # noqa: BLE001 -- test runner catch-all incl. skip sentinel
            if 'Skip' in type(e).__name__:
                print(f"SKIP {fn.__name__}: {e}")
            else:
                failed += 1
                print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed (skips not counted as failures)")
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
