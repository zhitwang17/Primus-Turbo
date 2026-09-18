###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The quant configs must stay registrable as torch opaque types.

torch >= 2.11 rejects register_opaque_type on a class whose metaclass is not
OpaqueBaseMeta. That call runs at import time, so a regression here does not
fail one operator -- it makes `import primus_turbo.pytorch` raise, on every
arch. These assertions are arch-independent on purpose.
"""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    ScalingRecipe,
    rht_mask_from_seed,
)

_OPAQUE_CONFIGS = (Float8QuantConfig, Float4QuantConfig, ScalingRecipe)


@pytest.mark.parametrize("cls", _OPAQUE_CONFIGS, ids=lambda c: c.__name__)
def test_carries_the_opaque_metaclass(cls):
    opaque_base = pytest.importorskip(
        "torch._opaque_base", reason="torch without the OpaqueBaseMeta requirement"
    )
    assert isinstance(cls, opaque_base.OpaqueBaseMeta)


@pytest.mark.parametrize("cls", _OPAQUE_CONFIGS, ids=lambda c: c.__name__)
def test_stays_hashable_and_comparable(cls):
    # register_opaque_type(typ="value") guards the baked-in constant on __eq__.
    assert cls() == cls()
    assert hash(cls()) == hash(cls())


@pytest.mark.parametrize("cls", _OPAQUE_CONFIGS, ids=lambda c: c.__name__)
def test_fx_repr_round_trips(cls):
    # torch.compile regenerates the config from this string, so it has to eval
    # back to an equal object under the globals the method hands out.
    config = cls()
    source, globals_ = config.__fx_repr__()
    assert eval(source, dict(globals_)) == config


def test_scaling_recipe_is_still_a_named_tuple():
    # It is spelled as a subclass of a NamedTuple to carry the metaclass, which
    # would silently cost tuple behaviour if that split were ever undone wrong.
    recipe = ScalingRecipe(use_2d_block=True, use_rht=True)
    assert isinstance(recipe, tuple)
    assert recipe[0] is True
    assert recipe._asdict()["use_rht"] is True
    assert recipe._replace(use_sr=True).use_sr is True
    first, *_ = recipe
    assert first is True


def test_randomized_rht_seed_contract_and_fx_round_trip():
    assert rht_mask_from_seed(0) == 0
    assert rht_mask_from_seed(1) == 0x514E28B7
    assert rht_mask_from_seed(42) == 0x087FCD5C
    assert rht_mask_from_seed(0xFFFFFFFF) == 0x81F16F39

    recipe = ScalingRecipe(use_rht=True, rht_seed=42)
    assert recipe.rht_mask == 0x087FCD5C
    assert ScalingRecipe(use_rht=False, rht_seed=42).rht_mask == 0

    config = Float4QuantConfig(rht_seed=42)
    assert config.rht_mask == recipe.rht_mask
    source, globals_ = config.__fx_repr__()
    assert eval(source, dict(globals_)) == config

    for bad_seed in (-1, 1 << 32):
        with pytest.raises(ValueError, match="rht_seed must be"):
            rht_mask_from_seed(bad_seed)
        with pytest.raises(ValueError, match="rht_seed must be"):
            ScalingRecipe(use_rht=False, rht_seed=bad_seed)
        with pytest.raises(ValueError, match="rht_seed must be"):
            Float4QuantConfig(rht_seed=bad_seed)


def _rht32_reference(x, mask):
    signs = torch.tensor([-1.0 if (mask >> i) & 1 else 1.0 for i in range(32)], dtype=x.dtype)
    value = x * signs
    halves = []
    for half in value.reshape(2, 16):
        transformed = half.clone()
        width = 1
        while width < 16:
            transformed = transformed.reshape(-1, 2, width)
            lo, hi = transformed[:, 0].clone(), transformed[:, 1].clone()
            transformed[:, 0] = lo + hi
            transformed[:, 1] = lo - hi
            transformed = transformed.reshape(16)
            width *= 2
        halves.append(transformed * 0.25)
    return torch.cat(halves)


def test_randomized_rht_pairing_contract():
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(32, generator=generator, dtype=torch.float64)
    right = torch.randn(32, generator=generator, dtype=torch.float64)
    mask = rht_mask_from_seed(42)

    paired_dot = torch.dot(_rht32_reference(left, mask), _rht32_reference(right, mask))
    torch.testing.assert_close(paired_dot, torch.dot(left, right), rtol=1e-12, atol=1e-12)

    mismatched_dot = torch.dot(_rht32_reference(left, mask), _rht32_reference(right, rht_mask_from_seed(1)))
    assert not torch.isclose(mismatched_dot, torch.dot(left, right), rtol=1e-4, atol=1e-4)


def test_torch_version_that_needs_the_metaclass_is_the_one_we_have():
    # A sanity line rather than a constraint: if torch drops the requirement,
    # the metaclass stays harmless and the test above skips.
    assert torch.__version__
