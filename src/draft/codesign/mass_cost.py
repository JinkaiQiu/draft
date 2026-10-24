"""Fixed-weight mass costs applied only to hardware design scores."""

import math
from numbers import Real


def validate_mass_cost(weight, reference_kg):
    def finite(value):
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)

    if not finite(weight) or weight < 0:
        raise ValueError("hardware_mass_cost_weight must be finite and nonnegative")
    if reference_kg is not None and (not finite(reference_kg) or reference_kg <= 0):
        raise ValueError("hardware_mass_reference_kg must be positive and finite")
    if weight > 0 and reference_kg is None:
        raise ValueError("hardware_mass_reference_kg is required when mass cost is enabled")


def apply_mass_cost(replica_returns, masses_kg, *, reference_kg, weight, discounted_seconds):
    """Subtract one static design cost after raw/V-trace task estimation.

    Weight is reward per simulated second for a robot at reference mass. The
    intended scoring exposure is used even when a replica fails, rather than
    making failure a way to shorten the mass charge. The cost is independent
    of task performance and applies to positive and negative scores alike.
    Each replica gets the same cost; hardware scoring then averages replicas
    as usual.
    """
    import torch

    validate_mass_cost(weight, reference_kg)
    if (isinstance(discounted_seconds, bool) or not isinstance(discounted_seconds, Real)
            or not math.isfinite(discounted_seconds) or discounted_seconds <= 0):
        raise ValueError("Mass scoring requires positive finite discounted_seconds")
    if (replica_returns.ndim != 2 or min(replica_returns.shape) < 1
            or not replica_returns.is_floating_point() or not torch.isfinite(replica_returns).all()):
        raise ValueError("Mass scoring requires finite [designs, replicas] task returns")
    if reference_kg is None:
        raise ValueError("Mass scoring requires hardware_mass_reference_kg")
    if masses_kg is None:
        raise ValueError("Mass scoring requires hardware_masses_kg from the compiled environment")
    with torch.no_grad():
        task = replica_returns.detach()
        masses = torch.as_tensor(masses_kg, device=task.device, dtype=task.dtype)
        if masses.shape != (task.shape[0],) or not torch.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError("hardware_masses_kg must contain one positive finite mass per valid design")
        task_scores = task.mean(-1)
        ratio = masses / reference_kg
        penalty = ratio * (weight * discounted_seconds)
        if not torch.isfinite(penalty).all():
            raise ValueError("Mass cost exceeds the scoring dtype's finite range")
        net = task - penalty[:, None]
        records = [dict(task_score=score, mass_kg=mass, mass_ratio=relative,
                        mass_penalty=cost)
                   for score, mass, relative, cost in zip(
                       task_scores.cpu().tolist(), masses.cpu().tolist(), ratio.cpu().tolist(),
                       penalty.cpu().tolist())]
        metrics = dict(hardware_task_score_mean=task_scores.mean().item(),
                       hardware_mass_penalty_mean=penalty.mean().item())
    return net, metrics, records
