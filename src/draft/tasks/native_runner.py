"""Native PPO lifecycle with atomic, self-describing conditioning checkpoints."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import tempfile

from mjlab.rl import MjlabOnPolicyRunner

def conditioning_transfer_metadata(env, train_cfg):
    """Describe inputs from managers, without drawing extra noisy observations."""
    conditioning = getattr(env.unwrapped.cfg, "morphology_conditioning_metadata", None)
    if conditioning is None:
        return {}
    shapes = {key: list(shape) for key, shape in env.unwrapped.observation_manager.group_obs_dim.items()}
    layout = {}
    for role in ("actor", "critic"):
        dimensions = sum(shapes[group][0] for group in train_cfg["obs_groups"][role])
        layout[role] = list(range(dimensions - 7, dimensions))
    transfer = dict(version=1, space_schema=conditioning["space_schema"],
                    control_schema=dict(observation_shapes=shapes, num_actions=env.num_actions),
                    actor_cfg=copy.deepcopy(train_cfg["actor"]), critic_cfg=copy.deepcopy(train_cfg["critic"]),
                    obs_groups=copy.deepcopy(train_cfg["obs_groups"]), conditioning_layout=layout)
    return dict(codesign_transfer=transfer, morphology_conditioning=copy.deepcopy(conditioning))


class MetadataOnPolicyRunner(MjlabOnPolicyRunner):
    """Inherit native learning and logging; extend checkpoint publishing."""

    def __init__(self, *args, checkpoint_metadata=None, **kwargs):
        self.checkpoint_metadata = copy.deepcopy(checkpoint_metadata or {})
        super().__init__(*args, **kwargs)

    def save(self, path, infos=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=destination.parent)
        os.close(fd)
        upload = self.cfg["upload_model"]
        try:
            self.cfg["upload_model"] = False
            super().save(temporary, infos={**self.checkpoint_metadata, **(infos or {})})
            os.replace(temporary, destination)
            if destination.name != "model_latest.pt":
                shutil.copyfile(destination, temporary)
                os.replace(temporary, destination.with_name("model_latest.pt"))
        finally:
            self.cfg["upload_model"] = upload
            if os.path.exists(temporary):
                os.unlink(temporary)
        if upload:
            self.logger.save_model(str(destination), self.current_learning_iteration)
