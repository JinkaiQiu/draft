"""CPU configuration checks for paired joint locks in the native task."""

import numpy as np
import pytest

from draft.codesign.compiler import DesignCompiler
from draft.codesign.space import DesignSpace


def test_topology_native_configuration_binds_pose_masks_and_rewards(tmp_path):
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.codesign.native_env import make_native_env_cfg
    from draft.tasks.quadruped.codesign.topology_mdp import (
        active_action_rate_l2, active_joint_pos_limits, paired_encoder_bias,
        reset_joints_topology,
    )

    space = DesignSpace.quadruped(joint_topology=True)
    physical = np.asarray(space.center)
    physical[7:13] = [0, 0, 0, 1, 1, 1]
    physical[13:19] = [.2, -.4, -1.2, 0, 0, 0]
    with DesignCompiler(space, tmp_path) as compiler:
        design = compiler.compile(physical)
    cfg = make_native_env_cfg([design], 2, design.xml_path,
                              hardware_aware=True, cpu_compile_workers=1)
    expected = [False] * 6 + [True] * 6
    assert cfg.hardware_action_mask == [expected, expected]
    assert cfg.hardware_joint_mask == [expected, expected]
    np.testing.assert_allclose(
        cfg.hardware_lock_angles, [[.2, -.4, -1.2, -.2, -.4, -1.2, 0, 0, 0, 0, 0, 0]] * 2)
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    assert np.allclose([pose[name] for name in (
        "fl_hip_roll", "fl_hip_pitch", "fl_knee", "fr_hip_roll", "fr_hip_pitch", "fr_knee")],
        [.2, -.4, -1.2, -.2, -.4, -1.2])
    assert cfg.hardware_nominal_heights[0] > 0
    assert cfg.events["reset_robot_joints"].func is reset_joints_topology
    assert cfg.events["base_com"].params["ranges"][1] == (0., 0.)
    assert cfg.events["foot_friction"].params["shared_random"] is True
    assert cfg.events["encoder_bias"].func is paired_encoder_bias
    assert cfg.rewards["action_rate_l2"].func is active_action_rate_l2
    assert cfg.rewards["dof_pos_limits"].func is active_joint_pos_limits
    for group in ("actor", "critic"):
        assert np.asarray(cfg.observations[group].terms["morphology"].params["values"]).shape == (2, 19)
