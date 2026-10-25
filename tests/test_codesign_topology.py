"""Topology-space decoding and conditional hardware distribution contracts."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from draft.codesign.policy import TopologyHardwarePolicy
from draft.codesign.space import DesignSpace
from draft.codesign.topology import (JOINT_GROUP_NAMES, JOINT_NAMES,
                                     expand_symmetric_topology, split_design)


def test_topology_schema_round_trip_bounds_and_canonical_active_angles():
    space = DesignSpace.quadruped(joint_topology=True,
                                   lock_angle_bounds={"front_knee": (-2.8, -.1)})
    assert space.dim == space.conditioning_dim == 19
    assert space.geometry_space() == DesignSpace.quadruped()
    assert len(space.names) == 19
    assert DesignSpace.from_dict(space.to_dict()) == space
    physical = np.array(space.center)
    physical[7:13] = [0, 1, 0, 1, 0, 1]
    physical[13:19] = [.2, 0, -1.4, 0, -.7, 0]
    normalized = space.normalize(physical)
    np.testing.assert_array_equal(normalized[7:13], physical[7:13])
    np.testing.assert_allclose(space.denormalize(normalized), physical)
    overrides = space.overrides(physical)
    assert set(overrides["joint_topology"]) == {"active", "lock_angles"}
    assert overrides["joint_topology"]["active"] == [0, 1, 0, 1, 0, 1]
    assert set(overrides).isdisjoint(set(space.names[7:]))
    active_reference = np.array(space.center)
    assert space.normalize(active_reference)[15] == 0  # active knee, outside inactive bounds
    assert space.denormalize(space.normalize(active_reference))[15] == 0


def test_topology_schema_rejects_fractional_masks_and_modified_symmetry():
    space = DesignSpace.quadruped(joint_topology=True)
    physical = np.array(space.center)
    physical[7] = .5
    with pytest.raises(ValueError, match="binary"):
        space.normalize(physical)
    schema = space.to_dict()
    schema["joint_topology"]["symmetry"] = "independent"
    with pytest.raises(ValueError, match="schema"):
        DesignSpace.from_dict(schema)
    for bounds in ({"front_hip_roll": (-.9, .7)},
                   {"rear_hip_roll": (-.7, .9)},
                   {"front_knee": (-2.9, -.1)}):
        with pytest.raises(ValueError, match="physical joint limits"):
            DesignSpace.quadruped(joint_topology=True, lock_angle_bounds=bounds)
    schema = space.to_dict()
    schema["version"] = True
    with pytest.raises(ValueError, match="version"):
        DesignSpace.from_dict(schema)
    schema = space.to_dict()
    schema["joint_topology"]["lock_angle_bounds"] = [None] * 6
    with pytest.raises(ValueError, match="bounds"):
        DesignSpace.from_dict(schema)


def test_symmetric_expansion_has_expected_joint_order_and_roll_sign():
    assert len(JOINT_GROUP_NAMES) == 6
    assert JOINT_NAMES == tuple(f"{leg}_{role}" for leg in ("fl", "fr", "rl", "rr")
                                for role in ("hip_roll", "hip_pitch", "knee"))
    active6 = np.array([0, 0, 0, 0, 0, 0])
    angles6 = np.array([.3, -.4, -1.2, -.5, .7, -2.])
    active12, angles12 = expand_symmetric_topology(active6, angles6)
    np.testing.assert_array_equal(active12, [0] * 12)
    np.testing.assert_allclose(angles12, [.3, -.4, -1.2, -.3, -.4, -1.2,
                                          -.5, .7, -2., .5, .7, -2.])
    _, masks, angles = split_design(np.r_[np.ones(7), active6, angles6])
    np.testing.assert_array_equal(masks, active6)
    np.testing.assert_array_equal(angles, angles6)


def test_sampling_reproducibility_decoding_and_conditional_likelihood():
    policy = TopologyHardwarePolicy(DesignSpace.quadruped(joint_topology=True),
                                    initial_active_probability=.75)
    first = policy.sample(128, torch.Generator().manual_seed(3))
    second = policy.sample(128, torch.Generator().manual_seed(3))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    raw, logprob, decoded, baseline = first
    assert raw.shape == (128, 14)
    assert decoded.shape == (128, 19)
    torch.testing.assert_close(logprob, policy.evaluate(raw)[0])
    torch.testing.assert_close(decoded[:, 13:] * decoded[:, 7:13], torch.zeros_like(decoded[:, 13:]))
    torch.testing.assert_close(baseline, policy.baseline.expand(128))
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(policy.evaluate(raw)[1]).all()
    assert policy.fixed_raw(2).shape == (2, 14)
    torch.testing.assert_close(policy.decode(policy.fixed_raw(2)),
                               torch.as_tensor(np.stack([policy.space.normalize(policy.space.center)] * 2), dtype=torch.float32))


def test_active_angles_are_marginalized_but_frozen_angles_have_gradients():
    policy = TopologyHardwarePolicy(DesignSpace.quadruped(joint_topology=True), hidden_dims=(8,))
    raw = torch.zeros(2, 14)
    raw[0, 0] = 63  # all active
    raw[1, 0] = 0  # all frozen
    changed = raw.clone()
    changed[:, 8:] = 5
    original_lp, _, _ = policy.evaluate(raw)
    changed_lp, _, _ = policy.evaluate(changed)
    torch.testing.assert_close(changed_lp[:1], original_lp[:1])
    assert changed_lp[1] < original_lp[1]
    original_lp.sum().backward()
    assert policy.topology_logits.grad is not None
    assert policy.conditioner[-1].bias.grad is not None
    assert policy.conditioner[-1].bias.grad[7:13].abs().sum() > 0 or policy.conditioner[-1].bias.grad[20:].abs().sum() > 0
    # A fully active action gives zero gradient to all angle parameters.
    policy.zero_grad()
    policy.evaluate(raw[:1])[0].sum().backward()
    torch.testing.assert_close(policy.conditioner[-1].bias.grad[7:13], torch.zeros(6))
    torch.testing.assert_close(policy.conditioner[-1].bias.grad[20:], torch.zeros(6))
    changed[0, 8:] = 1e30
    assert torch.isfinite(policy.evaluate(changed[:1])[0]).all()


def test_exact_joint_kl_and_probability_normalization():
    policy = TopologyHardwarePolicy(DesignSpace.quadruped(joint_topology=True), hidden_dims=(8,))
    state = policy.distribution_state()
    torch.testing.assert_close(policy.topology_logits.softmax(-1).sum(), torch.tensor(1.))
    torch.testing.assert_close(policy.kl_from_state(state), torch.tensor(0.), atol=1e-6, rtol=0)
    with torch.no_grad():
        policy.topology_logits[0] += .3
        policy.conditioner[-1].bias[0] += .2
    kl = policy.kl_from_state(state)
    assert math.isfinite(kl.item()) and kl.item() > 0
    kl.backward()
    assert policy.topology_logits.grad is not None
    assert policy.conditioner[-1].bias.grad[0] != 0


def test_nominal_lock_angle_means_and_invalid_policy_configuration():
    space = DesignSpace.quadruped(joint_topology=True)
    policy = TopologyHardwarePolicy(space)
    frozen = policy.decode(torch.cat((torch.zeros(1), policy.conditioner[-1].bias[:13])))
    physical = space.denormalize(frozen.detach().numpy())
    np.testing.assert_allclose(physical[13:], [0, .5, -1., 0, .5, -1.], atol=1e-6)
    narrow = DesignSpace.quadruped(joint_topology=True,
                                     lock_angle_bounds={"front_knee": (-2.8, -2.7)})
    clipped = TopologyHardwarePolicy(narrow)
    frozen = clipped.decode(torch.cat((torch.zeros(1), clipped.conditioner[-1].bias[:13])))
    assert -2.8 < narrow.denormalize(frozen.detach().numpy())[15] < -2.7
    for kwargs in ({"hidden_dims": ()}, {"hidden_dims": (True,)},
                   {"initial_active_probability": True}, {"initial_std": True}):
        with pytest.raises(ValueError):
            TopologyHardwarePolicy(space, **kwargs)
