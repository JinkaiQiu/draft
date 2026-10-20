"""Hardware-conditioned control and joint robot/controller optimization.

Design definitions and compilation do not import the GPU training stack.
"""

from .space import DesignParameter, DesignSpace

__all__ = ["DesignParameter", "DesignSpace", "get_runner_class"]


def get_runner_class(config):
    """Resolve the saved training schedule for training, resume and playback."""
    schedule = config.get("training_schedule", "full_window")
    if schedule == "alternating":
        from .alternating import AlternatingCodesignRunner
        return AlternatingCodesignRunner
    if schedule == "coadaptation":
        from .coadaptation import CoadaptationCodesignRunner
        return CoadaptationCodesignRunner
    if schedule == "full_window":
        from .runner import CodesignRunner
        return CodesignRunner
    raise ValueError(f"Unknown co-design training schedule: {schedule}")
