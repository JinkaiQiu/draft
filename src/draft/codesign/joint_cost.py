"""Static active-joint cost for hardware design scores."""

import math
from numbers import Real


def validate_joint_cost(weight):
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
        raise ValueError("hardware_joint_cost_weight must be finite and nonnegative")


def active_joint_counts(action_mask, *, num_designs, replicas, reference_count=12):
    """Read the immutable, design-major native action mask as physical counts."""
    import torch

    if action_mask is None:
        raise ValueError("hardware_action_mask is required when joint cost is enabled")
    if reference_count != 12 or isinstance(reference_count, bool):
        raise ValueError("Quadruped hardware_action_mask requires twelve joints")
    try:
        mask = torch.as_tensor(action_mask)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("hardware_action_mask must be a rectangular numeric matrix") from exc
    if mask.shape != (num_designs * replicas, reference_count):
        raise ValueError("hardware_action_mask must have one twelve-joint row per world")
    if torch.is_complex(mask) or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("hardware_action_mask must be binary and finite")
    rows = mask.reshape(num_designs, replicas, reference_count)
    if not (rows == rows[:, :1]).all():
        raise ValueError("hardware_action_mask replicas must be identical within each design")
    design_masks = rows[:, 0]
    if not ((design_masks[:, :3] == design_masks[:, 3:6]).all()
            and (design_masks[:, 6:9] == design_masks[:, 9:12]).all()):
        raise ValueError("hardware_action_mask must preserve paired left/right joint gates")
    return design_masks.sum(-1)


def apply_joint_cost(replica_returns, active_counts, *, weight, discounted_seconds,
                     reference_count=12):
    """Charge each replica for the full intended window after task estimation."""
    import torch

    validate_joint_cost(weight)
    if (isinstance(discounted_seconds, bool) or not isinstance(discounted_seconds, Real)
            or not math.isfinite(discounted_seconds) or discounted_seconds <= 0):
        raise ValueError("Joint scoring requires positive finite discounted_seconds")
    if isinstance(reference_count, bool) or not isinstance(reference_count, int) or reference_count <= 0:
        raise ValueError("reference_count must be a positive integer")
    if (replica_returns.ndim != 2 or min(replica_returns.shape) < 1
            or not replica_returns.is_floating_point() or not torch.isfinite(replica_returns).all()):
        raise ValueError("Joint scoring requires finite [designs, replicas] task returns")
    if active_counts is None:
        raise ValueError("active_counts must contain one physical count per valid design")
    with torch.no_grad():
        task = replica_returns.detach()
        try:
            counts = torch.as_tensor(active_counts, device=task.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("active_counts must contain numeric physical counts") from exc
        if (counts.dtype == torch.bool or torch.is_complex(counts)
                or counts.shape != (task.shape[0],)
                or not torch.isfinite(counts).all() or (counts < 0).any()
                or (counts > reference_count).any() or (counts != counts.floor()).any()
                or (reference_count == 12 and (counts.remainder(2) != 0).any())):
            raise ValueError("active_counts must contain one paired physical count in [0, reference_count] per valid design")
        counts = counts.to(dtype=task.dtype)
        penalty = counts / reference_count * (weight * discounted_seconds)
        if not torch.isfinite(penalty).all():
            raise ValueError("Joint cost exceeds the scoring dtype's finite range")
        net = task - penalty[:, None]
        records = [dict(task_score=score, active_joint_count=int(count), joint_penalty=cost)
                   for score, count, cost in zip(task.mean(-1).cpu().tolist(),
                                                 counts.cpu().tolist(), penalty.cpu().tolist())]
        metrics = dict(hardware_task_score_mean=task.mean(-1).mean().item(),
                       hardware_joint_penalty_mean=penalty.mean().item(),
                       hardware_active_joint_count_mean=counts.mean().item())
    return net, metrics, records
