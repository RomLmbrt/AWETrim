"""Unit tests for awetrim.aerostructural.forces

distribute_total_force_by_particle_mass spreads a single 3D resultant (inertial
or gravity) over the structural nodes in proportion to their mass. The split
must conserve the total force exactly.
"""

import numpy as np
import pytest

from awetrim.aerostructural.forces import distribute_total_force_by_particle_mass


class TestDistributeTotalForce:
    def test_output_shape(self):
        out = distribute_total_force_by_particle_mass([0.0, 0.0, -10.0], [1.0, 1.0, 2.0])
        assert out.shape == (3, 3)

    def test_conserves_total_force(self):
        # Sum the nodal forces with an elementwise add rather than
        # ``out.sum(axis=0)``: the reduction form forwards numpy's ``_NoValue``
        # sentinel, which breaks when casadi has re-imported numpy under some
        # test/coverage import orders.
        out = distribute_total_force_by_particle_mass([1.0, -2.0, -10.0], [1.0, 1.0, 2.0])
        total_out = out[0] + out[1] + out[2]
        assert total_out == pytest.approx([1.0, -2.0, -10.0])

    def test_force_proportional_to_mass(self):
        out = distribute_total_force_by_particle_mass([0.0, 0.0, -8.0], [1.0, 3.0])
        # Node masses 1:3 -> z-forces split -2 and -6.
        assert out[0, 2] == pytest.approx(-2.0)
        assert out[1, 2] == pytest.approx(-6.0)

    def test_equal_masses_split_evenly(self):
        out = distribute_total_force_by_particle_mass([6.0, 0.0, 0.0], [2.0, 2.0])
        assert out[0, 0] == pytest.approx(3.0)
        assert out[1, 0] == pytest.approx(3.0)

    def test_zero_total_mass_raises(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            distribute_total_force_by_particle_mass([0.0, 0.0, -1.0], [0.0, 0.0])
