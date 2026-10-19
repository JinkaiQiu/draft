import pytest

torch = pytest.importorskip("torch")

from draft.codesign.policy import HardwarePolicy, NativeControlPolicy


def test_hardware_dimensions_decoder_and_raw_likelihood():
    policy = HardwarePolicy(dimensions=11, initial_std=.3)
    raw, logprob, normalized, baseline = policy.sample(5)
    assert raw.shape == (5, 11)
    assert normalized.shape == (5, 11)
    torch.testing.assert_close(normalized, raw.tanh())
    torch.testing.assert_close(logprob, policy.distribution().log_prob(raw).sum(-1))
    torch.testing.assert_close(baseline, policy.baseline.expand(5))
    recomputed, entropy, values = policy.evaluate(raw)
    torch.testing.assert_close(recomputed, logprob)
    assert entropy.shape == values.shape == (5,)
    recomputed.sum().backward()
    assert policy.mean.grad is not None
    assert not normalized.requires_grad


@pytest.mark.parametrize("dimensions,std", [(0, .5), (True, .5), (7, 0), (7, float("nan"))])
def test_invalid_hardware_distribution(dimensions, std):
    with pytest.raises(ValueError):
        HardwarePolicy(dimensions, std)


def test_scalar_baseline_cannot_condition_on_draw():
    policy = HardwarePolicy(3)
    with torch.no_grad():
        policy.baseline.fill_(12)
    _, _, values = policy.evaluate(torch.tensor([[100., -50., 0.], [-100., 50., 1.]]))
    assert values.tolist() == [12., 12.]


def test_initial_reference_can_be_off_center_with_asymmetric_bounds():
    policy = HardwarePolicy(2, initial_normalized=[.25, -.7])
    torch.testing.assert_close(policy.decode(policy.mean), torch.tensor([.25, -.7]))


def test_fixed_conditioning_retains_scale_when_new_morphologies_are_seen():
    pytest.importorskip("rsl_rl")
    observations = {name: torch.randn(32, 56) for name in ("actor", "critic")}
    for values in observations.values():
        values[:, -8:-1] = 0.
    actor = dict(hidden_dims=(8,), obs_normalization=True,
                 distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    critic = dict(hidden_dims=(8,), obs_normalization=True)
    exclusions = {name: tuple(range(-8, -1)) for name in observations}
    policy = NativeControlPolicy.from_observation(observations, 12, actor, critic,
                                                normalization_exclusions=exclusions)
    policy.train()
    for _ in range(3):
        policy.update_normalization(observations)
    varied = torch.randn(4, 56)
    varied[:, -8:-1] = .5
    for model in (policy.actor, policy.critic):
        torch.testing.assert_close(model.obs_normalizer(varied)[:, -8:-1], varied[:, -8:-1])
        assert model.obs_normalizer.count.item() == 96
        assert not torch.equal(model.obs_normalizer._std[:, :48], torch.ones(1, 48))
    restored = NativeControlPolicy.from_observation(observations, 12, actor, critic,
                                                  normalization_exclusions=exclusions)
    restored.load_state_dict(policy.state_dict())
    torch.testing.assert_close(restored.actor.obs_normalizer(varied), policy.actor.obs_normalizer(varied))


def test_conditioning_exclusions_reject_out_of_range_indices():
    pytest.importorskip("rsl_rl")
    with pytest.raises(ValueError, match="outside actor"):
        NativeControlPolicy.from_observation({"actor": torch.zeros(1, 2), "critic": torch.zeros(1, 2)}, 1,
                                             dict(hidden_dims=(8,)), dict(hidden_dims=(8,)),
                                             normalization_exclusions={"actor": (-8,)})
