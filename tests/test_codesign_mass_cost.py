"""Hardware mass charges preserve task scale and native controller learning."""

import copy
import json

import pytest
import torch

from draft.codesign.mass_cost import apply_mass_cost, validate_mass_cost


STOCK_MASS = 32.63809926450245
LEARNED_MASS = 39.37773037


def charge(returns, masses, **overrides):
    cfg = dict(reference_kg=STOCK_MASS, weight=.05,
               discounted_seconds=256 * .02)
    cfg.update(overrides)
    return apply_mass_cost(torch.tensor(returns, dtype=torch.float64), masses, **cfg)


def test_real_mass_cost_is_small_and_task_advantage_dominates():
    net, metrics, records = charge([[17., 17.], [18., 18.]], [STOCK_MASS, LEARNED_MASS])
    assert records[0]["mass_penalty"] == pytest.approx(.256)
    assert records[1]["mass_penalty"] == pytest.approx(.3088629771)
    assert net[1].mean() > net[0].mean()
    assert records[1]["mass_penalty"] / 18 < .02
    assert metrics["hardware_mass_penalty_mean"] == pytest.approx((.256 + .3088629771) / 2)
    json.dumps(records, allow_nan=False)


def test_mass_pressure_is_independent_of_task_score_and_preserves_replica_differences():
    raw = torch.tensor([[1., 3.], [-10., -20.], [0., 0.]], dtype=torch.float64)
    original = raw.clone()
    net, _, records = apply_mass_cost(raw, [STOCK_MASS] * 3,
        reference_kg=STOCK_MASS, weight=.05, discounted_seconds=5.12)
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    torch.testing.assert_close(net, raw - .256)
    assert all(record["mass_penalty"] == pytest.approx(.256) for record in records)
    # A heavier design pays more even when task performance is poor or negative.
    heavier, _, heavy_records = charge(raw.tolist(), [LEARNED_MASS] * 3)
    assert all(record["mass_penalty"] > .256 for record in heavy_records)
    assert (heavier < net).all()


@pytest.mark.parametrize("settings", [(-1, STOCK_MASS), (True, STOCK_MASS),
    (.05, None), (.05, 0.), (float("inf"), STOCK_MASS), (.05, float("nan"))])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        validate_mass_cost(*settings)


@pytest.mark.parametrize("masses", [None, [STOCK_MASS], [STOCK_MASS, -1.], [STOCK_MASS, float("inf")]])
def test_invalid_environment_mass(masses):
    with pytest.raises(ValueError, match="hardware_masses_kg"):
        charge([[1.], [2.]], masses)


def runner_with_mass(*, weight=.05, enabled=False, **overrides):
    pytest.importorskip("rsl_rl")
    from test_codesign_vtrace_integration import make_runner
    runner = make_runner(enabled=enabled, hardware_mass_cost_weight=weight,
                         hardware_mass_reference_kg=STOCK_MASS, **overrides)
    factory = runner.env_factory

    def with_mass(*args, **kwargs):
        env = factory(*args, **kwargs)
        env.hardware_masses_kg = (STOCK_MASS, LEARNED_MASS)
        env.step_dt = .02
        return env

    runner.env_factory = with_mass
    return runner


@pytest.mark.parametrize("vtrace", [False, True])
def test_mass_changes_only_hardware_score_after_task_estimation(vtrace, tmp_path):
    from draft.codesign.metrics import tensorboard_scalars
    plain = runner_with_mass(weight=0., enabled=vtrace)
    scored = runner_with_mass(enabled=vtrace)
    scored.log_dir = tmp_path
    scored.policy.control.load_state_dict(copy.deepcopy(plain.policy.control.state_dict()))
    try:
        # Both windows retain the same cohort; a static mass charge cannot accumulate.
        for window in range(2):
            torch.manual_seed(412 + window)
            raw = plain.collect()
            rng = torch.get_rng_state().clone()
            expected = copy.deepcopy(plain.policy.control.state_dict())
            torch.manual_seed(412 + window)
            batch = scored.collect()
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, value in scored.policy.control.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            if vtrace:
                for key, value in scored._vtrace_scorer.value_model.state_dict().items():
                    torch.testing.assert_close(value, plain._vtrace_scorer.value_model.state_dict()[key], rtol=0, atol=0)
            records = batch["control"]["hardware_objective_records"]
            penalties = torch.tensor([r["mass_penalty"] for r in records])
            torch.testing.assert_close(batch["scores"], raw["scores"] - penalties)
            training = batch["controller_training"]
            assert training["hardware_task_score_mean"] == pytest.approx(raw["scores"].mean().item())
            assert records[0]["mass_penalty"] == pytest.approx(.05 * .64)
            metrics = scored.update(batch)
            plain.update(raw)
            tags = tensorboard_scalars(metrics)
            assert {tag for tag in tags if tag.startswith("HardwareReward/")} == {
                "HardwareReward/task", "HardwareReward/mass_cost"}
            assert tags["HardwareReward/task"] - tags["HardwareReward/mass_cost"] == pytest.approx(
                tags["codesign/design_score"])
        rows = [json.loads(line) for line in (tmp_path / "design_records.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        for row in rows:
            assert row["score"] == pytest.approx(row["task_score"] - row["mass_penalty"])
            assert row["mass_kg"] == pytest.approx((STOCK_MASS, LEARNED_MASS)[row["draw_index"]])
    finally:
        plain.close()
        scored.close()


def test_old_disabled_checkpoint_resumes_and_cost_changes_require_warmstart(tmp_path):
    source, target, changed = runner_with_mass(weight=0.), runner_with_mass(weight=0.), runner_with_mass()
    try:
        path = tmp_path / "old.pt"
        source.config["hardware_mass_reference_kg"] = target.config["hardware_mass_reference_kg"] = None
        source.save(path)
        payload = torch.load(path, weights_only=False)
        for key in ("hardware_mass_cost_weight", "hardware_mass_reference_kg"):
            del payload["config"][key]
        torch.save(payload, path)
        target.load(path)
        with pytest.raises(ValueError, match="protected.*hardware_mass"):
            changed.load(path)
    finally:
        source.close()
        target.close()
        changed.close()


def test_typed_cost_requires_coadaptation_and_explicit_reference():
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg
    with pytest.raises(ValueError, match="requires training_schedule"):
        CodesignRunnerCfg(hardware_mass_cost_weight=.05, hardware_mass_reference_kg=STOCK_MASS).validate()
    with pytest.raises(ValueError, match="reference_kg is required"):
        CodesignRunnerCfg(training_schedule="coadaptation", hardware_mass_cost_weight=.05).validate()
