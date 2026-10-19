from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mjlab")
from draft.tasks.quadruped.hardware_mdp import TerrainRelativeCollapse, TerrainRelativeHeightReward


def environment(root_z, feet_z, terrain_z):
    root_z, feet_z, terrain_z = (torch.as_tensor(value, dtype=torch.float64)
                                for value in (root_z, feet_z, terrain_z))
    count, feet, rays = terrain_z.shape
    root = torch.zeros(count, 3, dtype=torch.float64)
    root[:, 2] = root_z
    frames = torch.zeros(count, feet, 3, dtype=torch.float64)
    frames[..., 2] = feet_z
    hits = torch.zeros(count, feet, rays, 3, dtype=torch.float64)
    hits[..., 2] = terrain_z
    normals = torch.zeros_like(hits)
    normals[..., 2] = 1.
    sensor = SimpleNamespace(frame_pos_w=frames, distances=torch.ones(count, feet * rays),
                             hit_pos_w=hits.flatten(1, 2), normals_w=normals.flatten(1, 2),
                             heights=feet_z - terrain_z.max(-1).values)
    return SimpleNamespace(num_envs=count, device="cpu", scene={
        "robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root)),
        "foot_height_scan": SimpleNamespace(data=sensor)})


def terms(env, nominal_heights):
    cfg = SimpleNamespace(params=dict(nominal_heights=nominal_heights))
    return TerrainRelativeHeightReward(cfg, env), TerrainRelativeCollapse(cfg, env), cfg.params


def test_world_translation_and_foot_swing_do_not_change_height_terms():
    env = environment([.475, .475], [[.02] * 4] * 2, torch.zeros(2, 4, 4))
    reward, collapse, params = terms(env, .5)
    before = reward(env, **params).clone()
    torch.testing.assert_close(before, torch.ones(2, dtype=torch.float64))
    assert not collapse(env, **params).any()
    # Same ground while feet swing up: native reduced clearance changes with
    # foot z, but the reconstructed support elevation does not.
    scan = env.scene["foot_height_scan"].data
    swing = torch.tensor([[0., .3, .1, .4], [.2, 0., .3, .1]])
    scan.frame_pos_w[..., 2] += swing
    scan.heights += swing
    torch.testing.assert_close(reward(env, **params), before)
    # Translate both terrain and robot, including below world z=0.
    offset = torch.tensor([12., -7.], dtype=torch.float64)
    env.scene["robot"].data.root_link_pos_w[:, 2] += offset
    scan.frame_pos_w[..., 2] += offset[:, None]
    scan.hit_pos_w[..., 2] += offset[:, None]
    torch.testing.assert_close(reward(env, **params), before)
    assert not collapse(env, **params).any()


def test_per_design_nominal_heights_match_single_world_and_contiguous_replicas():
    nominal = torch.tensor([.3, .3, .8, .8], dtype=torch.float64)
    env = environment(.95 * nominal, torch.zeros(4, 4), torch.zeros(4, 4, 4))
    reward, collapse, params = terms(env, nominal.tolist())
    torch.testing.assert_close(reward(env, **params), torch.ones_like(nominal))
    env.scene["robot"].data.root_link_pos_w[:, 2] = nominal * torch.tensor([.49, .5, .49, .5])
    assert collapse(env, **params).tolist() == [True, False, True, False]
    for i, height in enumerate(nominal):
        one = environment([env.scene["robot"].data.root_link_pos_w[i, 2]], [[0.] * 4], torch.zeros(1, 4, 4))
        one_reward, one_collapse, one_params = terms(one, float(height))
        torch.testing.assert_close(one_reward(one, **one_params)[0], reward(env, **params)[i])
        assert one_collapse(one, **one_params)[0] == collapse(env, **params)[i]


def test_sloped_ground_uses_highest_valid_ring_then_equal_weight_per_foot():
    # Four elevated/slope samples per foot. Native min height means highest
    # local ground, not the average ray or the current foot position.
    terrain = torch.tensor([[[1., 1.1, 1.2, 1.3], [1.2, 1.3, 1.4, 1.5],
                             [.8, .9, 1., 1.1], [1.4, 1.5, 1.6, 1.7]]])
    ground = (1.3 + 1.5 + 1.1 + 1.7) / 4
    env = environment([ground + .475], [[1.4, 1.6, 1.2, 1.8]], terrain)
    reward, collapse, params = terms(env, .5)
    assert reward(env, **params).item() == pytest.approx(1.)
    assert not collapse(env, **params).item()
    env.scene["robot"].data.root_link_pos_w[:, 2] = ground + .24
    assert collapse(env, **params).item()


def test_partial_misses_ignore_fabricated_heights_and_unsupported_feet():
    env = environment([2.475], [[2.1, 20., 30., 40.]], torch.full((1, 4, 4), 2.))
    scan = env.scene["foot_height_scan"].data
    scan.distances[:, 1:] = -1.  # Only one actual terrain hit remains.
    scan.hit_pos_w[:, 1:, 2] = float("nan")
    scan.heights[:] = 0.  # Native all-miss fallback must not establish support.
    reward, collapse, params = terms(env, .5)
    assert reward(env, **params).item() == pytest.approx(1.)
    assert not collapse(env, **params).item()


def test_supported_feet_have_equal_weight_with_different_numbers_of_valid_rays():
    terrain = torch.tensor([[[0.] * 4, [2.] * 4, [20.] * 4, [30.] * 4]])
    env = environment([1.475], [[.1, 2.1, 20.1, 30.1]], terrain)
    scan = env.scene["foot_height_scan"].data
    scan.distances[:, 1:4] = -1.  # One ray for foot0, four for foot1.
    scan.distances[:, 8:] = -1.  # Other feet have no observable support.
    reward, collapse, params = terms(env, .5)
    assert reward(env, **params).item() == pytest.approx(1.)
    assert not collapse(env, **params).item()


@pytest.mark.parametrize("fault", ["miss", "distance_nan", "hit_nan", "normal_nan", "backface", "vertical", "frame_nan", "root_nan"])
def test_unobservable_or_invalid_ground_fails_closed_without_nan(fault):
    env = environment([.475], [[.02] * 4], torch.zeros(1, 4, 4))
    scan = env.scene["foot_height_scan"].data
    if fault == "miss":
        scan.distances.fill_(-1.)
    elif fault == "distance_nan":
        scan.distances.fill_(float("nan"))
    elif fault == "hit_nan":
        scan.hit_pos_w.fill_(float("nan"))
    elif fault == "normal_nan":
        scan.normals_w.fill_(float("nan"))
    elif fault in ("backface", "vertical"):
        scan.normals_w[..., 2] = -1. if fault == "backface" else 0.
    elif fault == "frame_nan":
        scan.frame_pos_w.fill_(float("nan"))
    else:
        env.scene["robot"].data.root_link_pos_w.fill_(float("nan"))
    reward, collapse, params = terms(env, .5)
    assert reward(env, **params).tolist() == [0.]
    assert collapse(env, **params).tolist() == [True]


@pytest.mark.parametrize("heights", [0., -1., float("nan"), float("inf"), [.5], [.5, 0.]])
def test_invalid_nominal_height_config_rejected_once_at_construction(heights):
    env = environment([.475] * 2, [[.02] * 4] * 2, torch.zeros(2, 4, 4))
    with pytest.raises(ValueError, match="nominal_heights"):
        terms(env, heights)


def test_nominal_values_are_cached_once_and_not_aliased_to_mutable_config():
    env = environment([.475], [[.02] * 4], torch.zeros(1, 4, 4))
    nominal = torch.tensor([.5], dtype=torch.float64)
    reward, _, params = terms(env, nominal)
    pointer = reward.nominal_heights.data_ptr()
    nominal.fill_(10.)
    assert reward(env, **params).item() == pytest.approx(1.)
    assert reward.nominal_heights.data_ptr() == pointer
