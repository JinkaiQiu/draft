"""Generation remains authoritative, with reusable and verified cache assets."""

import numpy as np
import pytest

from draft.codesign.compiler import DesignCompiler
from draft.codesign.space import DesignSpace


def test_generated_design_cache_and_physical_properties(tmp_path):
    import mujoco
    space = DesignSpace.quadruped()
    with DesignCompiler(space, tmp_path) as compiler:
        records = compiler.compile_many([space.center, space.center])
        assert all(record.error is None for record in records)
        first, second = [record.design for record in records]
        assert first == second
        assert first.xml_path.is_file()
        assert first.parameters["L_motor_effort"] == 72
        assert first.leg_length == pytest.approx(.48)
        model = mujoco.MjModel.from_xml_path(str(first.xml_path))
        assert model.nv == 18 and model.body_mass.sum() > 0
        np.testing.assert_allclose(first.morphology, np.zeros(7), atol=1e-14)
        assert len(list(tmp_path.glob("*/design.json"))) == 1
        assert compiler.compile(space.center).xml_path == first.xml_path
        first.xml_path.write_text("corrupt")
        with pytest.raises(ValueError, match="validation"):
            compiler.compile(space.center)


def test_compiler_preserves_rejected_rows_and_validates_before_generation(tmp_path):
    space = DesignSpace.quadruped(bounds={"hip_roll_link_length": [.0001, .2]})
    invalid = list(space.center)
    invalid[-1] = .0001
    with DesignCompiler(space, tmp_path) as compiler:
        results = compiler.compile_many([invalid, space.center, invalid])
        assert len(results) == 3 and results[1].design is not None
        assert results[0].design is results[2].design is None
        assert results[0].error == results[2].error
        assert results[0].error
        with pytest.raises(ValueError, match="bounds"):
            compiler.compile_many([[0] * 7])


def test_cache_pruning_retains_current_cohort(tmp_path):
    space = DesignSpace.quadruped()
    rows = np.asarray([space.center, space.center, space.center])
    rows[1, 0] *= 1.02
    rows[2, 0] *= 1.04
    with DesignCompiler(space, tmp_path, max_cache_entries=1) as compiler:
        cohort = compiler.compile_many(rows[:2])
        assert all(record.design.xml_path.exists() for record in cohort)
        assert len(list(tmp_path.glob("*/design.json"))) == 2  # live cohort wins over cache limit
        compiler.compile_many(rows[2:])
        assert len(list(tmp_path.glob("*/design.json"))) == 1


def test_retain_assets_survives_new_batch_and_releases_on_error(tmp_path):
    space = DesignSpace.quadruped()
    other = list(space.center)
    other[0] *= 1.02
    with DesignCompiler(space, tmp_path, max_cache_entries=1) as compiler:
        live = compiler.compile(space.center)
        with pytest.raises(RuntimeError, match="stop"):
            with compiler.retain_assets([live.design_id]):
                evaluation = compiler.compile(other)
                assert live.xml_path.is_file()
                assert evaluation.xml_path.is_file()
                assert len(list(tmp_path.glob("*/design.json"))) == 2
                raise RuntimeError("stop")
        compiler.prune()
        assert not live.xml_path.exists()
        assert evaluation.xml_path.is_file()
        assert len(list(tmp_path.glob("*/design.json"))) == 1


def test_spawned_workers_produce_ordered_deduplicated_results(tmp_path):
    space = DesignSpace.quadruped()
    rows = np.asarray([space.center, space.center])
    rows[1, 0] *= 1.02
    with DesignCompiler(space, tmp_path, workers=2) as compiler:
        records = compiler.compile_many(rows)
    assert all(record.design is not None for record in records)
    assert [record.design.physical for record in records] == [tuple(row) for row in rows]


def test_shared_cache_preserves_other_compilers_live_cohort(tmp_path):
    space = DesignSpace.quadruped()
    other = list(space.center)
    other[0] *= 1.02
    with DesignCompiler(space, tmp_path, max_cache_entries=1) as first:
        design = first.compile(space.center)
        with DesignCompiler(space, tmp_path, max_cache_entries=1) as second:
            second.compile(other)
            assert design.xml_path.is_file()
        first.prune()
        assert design.xml_path.is_file()
        assert len(list(tmp_path.glob("*/design.json"))) == 1
