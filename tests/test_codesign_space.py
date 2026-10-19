"""Physical design schema and normalization are stable across serialization."""

import numpy as np
import pytest

from draft.codesign.space import DesignParameter, DesignSpace, GEOMETRY_NAMES


def test_seven_geometry_coordinates_and_reference_are_preserved():
    space = DesignSpace.quadruped()
    assert space.names == GEOMETRY_NAMES
    np.testing.assert_allclose(space.center, [.25, .23, .52, .20, .18, .16, .10])
    assert DesignSpace.from_dict(space.to_dict()) == space
    reference = space.reference_parameters
    reference["L_motor_effort"] = 1
    assert space.reference_parameters["L_motor_effort"] == 72
    assert not space.reference_parameters["allow_hypothetical"]


def test_physical_and_normalized_batches_round_trip_at_bounds():
    space = DesignSpace.quadruped()
    physical = np.array([space.lower, space.center, space.upper])
    normalized = space.normalize(physical)
    np.testing.assert_allclose(normalized, [-np.ones(7), np.zeros(7), np.ones(7)], atol=1e-14)
    np.testing.assert_allclose(space.denormalize(normalized), physical)
    overrides = space.overrides(space.center)
    assert overrides["hip_roll_mot"] == overrides["hip_pitch_mot"] == overrides["knee_mot"] == "L"


@pytest.mark.parametrize("values", [[0] * 6, [float("nan")] * 7, [float("inf")] * 7])
def test_invalid_physical_coordinates_are_rejected(values):
    with pytest.raises(ValueError):
        DesignSpace.quadruped().normalize(values)


def test_bounds_and_feasibility_bypasses_are_rejected():
    with pytest.raises(ValueError, match="Unknown geometry"):
        DesignSpace.quadruped(bounds={"torso_mass": [1, 2]})
    with pytest.raises(ValueError, match="feasibility"):
        DesignSpace.quadruped(reference_overrides={"allow_hypothetical": True})
    with pytest.raises(ValueError):
        DesignParameter("length", 1, 2, 3)
    with pytest.raises(ValueError):
        DesignSpace.quadruped().denormalize([1.1] * 7)


def test_custom_bounds_preserve_explicit_coordinate_order():
    space = DesignSpace.quadruped(bounds={"torso_lx": [.4, .8]})
    assert space.parameters[2].lower == .4
    assert space.parameters[2].upper == .8
    np.testing.assert_allclose(space.denormalize(space.normalize(space.center)), space.center)


def test_serialized_schema_cannot_change_order_or_bypass_feasibility():
    data = DesignSpace.quadruped().to_dict()
    data["parameters"] = list(reversed(data["parameters"]))
    with pytest.raises(ValueError, match="order"):
        DesignSpace.from_dict(data)
    data = DesignSpace.quadruped().to_dict()
    data["reference"]["allow_motor_overlap"] = True
    with pytest.raises(ValueError, match="feasibility"):
        DesignSpace.from_dict(data)
