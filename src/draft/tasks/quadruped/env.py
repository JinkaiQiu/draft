"""Flat-ground quadruped task: the base policy the challenge tasks warm-start from.

``make_base_env_cfg`` with no curriculum. Point a challenge task's
``warmstart_checkpoint`` at this policy's checkpoint.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from draft.tasks.quadruped.base_env import make_base_env_cfg


def make_env_cfg(*, morphology_conditioning: bool | dict = False, **base) -> ManagerBasedRlEnvCfg:
    """The base environment, with no curriculum on top.

    Base kwargs pass unchanged to ``make_base_env_cfg``. Optional morphology
    conditioning appends only seven physical-design features to actor/critic.
    """
    if not isinstance(morphology_conditioning, (bool, dict)):
        raise ValueError("morphology_conditioning must be a boolean or configuration mapping")
    if morphology_conditioning is True or isinstance(morphology_conditioning, dict):
        from draft.codesign.sampling import make_conditioned_cfg
        return make_conditioned_cfg(morphology_conditioning, **base)
    cfg, _ctx = make_base_env_cfg(**base)
    return cfg
