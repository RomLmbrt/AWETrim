import copy

import matplotlib.pyplot as plt
import numpy as np

from awetrim.timeseries.phase_parametrized import PhaseParameterized


class ReelinSimple:
    """Encapsulates the simple reel-in workflow for parametrized simulations."""

    def __init__(
        self,
        *,
        system_model,
        beta0=np.radians(30),
        distance_radial_start=330,
        distance_radial_end=220,
        radial_parameters=None,
        depower=1.0,
    ):
        # Physical configuration
        self.beta0 = beta0
        self.distance_radial_start = distance_radial_start
        self.distance_radial_end = distance_radial_end
        self.depower = depower

        default_radial_parameters = {
            "reeling_strategy": "force",
            "force_model": "quadratic",
            "reeling_speed": 1.0,
            "max_tether_force": 2e4,
            "min_tether_force": 5000.0,
            "softplus": True,
            "softplus_beta": 1e-4,
            "softminus": True,
            "softminus_beta": 1e-3,
            "slope": 2716,
            "offset": -3,
        }

        # Derived configuration/state containers
        self.variables_to_plot = [
            "speed_tangential",
            "tension_tether_ground",
            "s",
            "distance_radial",
        ]
        self.radial_parameters = radial_parameters or default_radial_parameters
        self.base_start_state_ri = {
            "t": 0,
            "s": 0,
            "s_dot": 0.2,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": self.distance_radial_start,
            "speed_radial": -6,
        }
        self.pattern_config_ri = {
            "pattern_type": "reel_in_simple",
            "path_parameters": {"beta0": self.beta0},
            "radial_parameters": self.radial_parameters,
            "sim_parameters": {
                "start_angle": 0,
                "end_angle": 0.8,
                "n_points": 200,
            },
            "optimization_parameters": ["end_angle"],
        }

        # Components and state placeholders
        self.system_model = system_model

        self._phase_reel_in = None
        self._phase_transition = None
        self._opti = None
        self._opti_vars_reel_in = None
        self._objective_reel_in = None
        self._opti_vars_transition = None
        self._objective_transition = None
        self._solution = None
        self._transition_pattern_config = None
        self._transition_start_state = None

    def _reset_transition_state(self):
        self._phase_transition = None
        self._opti_vars_transition = None
        self._objective_transition = None
        self._solution = None
        self._transition_pattern_config = None
        self._transition_start_state = None

    def initialize_reel_in_phase(self):
        """Prepare the initial reel-in optimization phase."""
        if self._phase_reel_in is not None:
            return self._phase_reel_in

        self._opti = None
        self._reset_transition_state()
        self.system_model.input_depower = self.depower
        self._phase_reel_in = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=self.pattern_config_ri,
        )
        (
            self._opti,
            self._opti_vars_reel_in,
            self._objective_reel_in,
        ) = self._phase_reel_in.opti_phase(start_state=self.base_start_state_ri)

        s_transition = self._phase_reel_in.return_variable("s")[-1]
        elevation_transition = self._phase_reel_in.return_variable("angle_elevation")[
            -1
        ]
        t_start = self._phase_reel_in.return_variable("t")[-1]
        r_start = self._phase_reel_in.return_variable("distance_radial")[-1]

        self._transition_pattern_config = {
            "pattern_type": "transition_simple",
            "path_parameters": {"beta0": elevation_transition},
            "radial_parameters": self.radial_parameters,
            "sim_parameters": {
                "start_angle": 0,
                "end_angle": s_transition,
                "n_points": 200,
            },
            "optimization_parameters": [],
        }
        self._transition_start_state = copy.deepcopy(self.base_start_state_ri)
        self._transition_start_state["distance_radial"] = r_start
        self._transition_start_state["t"] = t_start
        return self._phase_reel_in

    def initialize_transition_phase(self):
        """Extend the optimization problem with the transition phase setup."""
        self.initialize_reel_in_phase()
        if self._phase_transition is not None:
            return self._phase_transition

        pattern_config_opti = copy.deepcopy(self._transition_pattern_config)
        start_state_opti = copy.deepcopy(self._transition_start_state)
        pattern_config_opti["sim_parameters"]["end_angle"] = self._opti_vars_reel_in[
            "end_angle"
        ]
        pattern_config_opti["path_parameters"]["beta0"] = (
            self.beta0 + self._opti_vars_reel_in["end_angle"]
        )
        start_state_opti["distance_radial"] = self._opti_vars_reel_in[
            "distance_radial"
        ][-1]

        self._phase_transition = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=self._transition_pattern_config,
            pattern_config_opti=pattern_config_opti,
        )
        (
            self._opti,
            self._opti_vars_transition,
            self._objective_transition,
        ) = self._phase_transition.opti_phase(
            start_state=self._transition_start_state,
            opti=self._opti,
            start_state_opti=start_state_opti,
        )
        return self._phase_transition

    def get_opti_components(self, phase="combined"):
        """Return the optimization problem components (opti, variables, objective).

        Args:
            phase: One of ``\"reel_in\"``, ``\"transition\"``, or ``\"combined\"``.
        """
        phase = phase.lower()
        valid = {"reel_in", "transition", "combined"}
        if phase not in valid:
            raise ValueError(f"phase must be one of {valid}, received '{phase}'.")

        self.initialize_reel_in_phase()

        if phase == "reel_in":
            return (
                self._opti,
                self._copy_phase_dict(self._opti_vars_reel_in),
                self._copy_phase_dict(self._objective_reel_in),
            )

        self.initialize_transition_phase()

        if phase == "transition":
            return (
                self._opti,
                self._copy_phase_dict(self._opti_vars_transition),
                self._copy_phase_dict(self._objective_transition),
            )

        combined_opti_vars = self._merge_phase_dicts(
            self._opti_vars_reel_in, self._opti_vars_transition
        )
        combined_objective = self._merge_phase_dicts(
            self._objective_reel_in, self._objective_transition
        )
        return self._opti, combined_opti_vars, combined_objective

    def run_opti(self):
        """Solve the optimization problem for the transition phase."""
        if self._solution is not None:
            return self._solution

        self.initialize_transition_phase()

        self._opti.subject_to(
            self._opti_vars_transition["distance_radial"][-1]
            == self.distance_radial_end
        )
        self._solution = self.run_simulation_opti(
            self._opti, 0, phase=self._phase_transition
        )
        return self._solution

    def run_simulation_opti(self, opti, objective, *, phase=None):
        """Solve the supplied opti problem and return the solution."""
        phase = phase or self._phase_transition or self._phase_reel_in
        if phase is None:
            raise ValueError(
                "No phase available to solve optimization. "
                "Initialize a reel-in or transition phase first."
            )

        opti.minimize(objective)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "bound_relax_factor": 1e-8,
                    "tol": 1e-4,
                    "acceptable_iter": 3,
                    "acceptable_tol": 1e-4,
                    "constr_viol_tol": 1e-4,
                    "dual_inf_tol": 1e-4,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                }
            },
        )

        try:
            solution = opti.solve()

            print("\nOptimized Pattern Variables:")
            optimized_config = phase.pattern_config.copy()
            for var_name, mx in phase.optimization_vars.items():
                val = solution.value(mx)
                print(f"  {var_name}: {val}")

                if var_name in optimized_config.get("path_parameters", {}):
                    optimized_config["path_parameters"][var_name] = val
                elif var_name in optimized_config.get("radial_parameters", {}):
                    optimized_config["radial_parameters"][var_name] = val
                elif var_name in optimized_config.get("sim_parameters", {}):
                    optimized_config["sim_parameters"][var_name] = val
            phase.pattern_config = optimized_config
            if phase is self._phase_transition:
                self._transition_pattern_config = optimized_config.copy()
            elif phase is self._phase_reel_in:
                self.pattern_config_ri = optimized_config.copy()
            return solution

        except Exception as exc:
            print("Debug optimization information:")
            for var_name, mx in phase.optimization_vars.items():
                try:
                    print(f"  {var_name}: {opti.debug.value(mx)}")
                except Exception:
                    pass
            print("Optimization failed:", exc)
            return None

    def run_simulation(self, *, solution=None, run_plots=False):
        """Execute the reel-in and transition simulations.

        Args:
            solution: Optional CasADi solution produced via `run_opti`. If omitted,
                the method uses the latest stored solution when available, otherwise
                it relies on the current pattern configuration.
            run_plots: When True, produce overview plots using Matplotlib.
        """
        self.initialize_reel_in_phase()

        active_solution = solution or self._solution
        if solution is not None:
            self._solution = solution

        if active_solution is not None:
            end_angle = active_solution.value(self._opti_vars_reel_in["end_angle"])
            self.pattern_config_ri["sim_parameters"]["end_angle"] = end_angle

        phase_reel_in = self._run_parametrized_phase(
            label_prefix="a",
            depower=self.depower,
            start_state=self.base_start_state_ri,
            pattern_config=self.pattern_config_ri,
            phase_sym=True,
        )
        fig = axes = None
        if run_plots:
            fig, axes, _ = phase_reel_in.plot_overview_3d(
                x_param="t",
                variables=self.variables_to_plot,
            )

        s_transition = phase_reel_in.return_variable("s")[-1]
        elevation_transition = phase_reel_in.return_variable("angle_elevation")[-1]
        t_start = phase_reel_in.return_variable("t")[-1]
        r_start = phase_reel_in.return_variable("distance_radial")[-1]
        pattern_config = {
            "pattern_type": "transition_simple",
            "path_parameters": {"beta0": elevation_transition},
            "radial_parameters": self.radial_parameters,
            "sim_parameters": {
                "start_angle": 0,
                "end_angle": s_transition,
                "n_points": 200,
            },
        }
        start_state = copy.deepcopy(self.base_start_state_ri)
        start_state["distance_radial"] = r_start
        start_state["t"] = t_start
        phase_transition = self._run_parametrized_phase(
            label_prefix="a",
            depower=self.depower,
            start_state=start_state,
            pattern_config=pattern_config,
            phase_sym=True,
        )
        if run_plots and axes is not None:
            phase_transition.plot_overview_3d(
                x_param="t",
                variables=self.variables_to_plot,
                axes=axes,
            )
        plt.show()

        print(
            "final radial distance:",
            phase_transition.return_variable("distance_radial")[-1],
        )

        return {
            "phase_reel_in": phase_reel_in,
            "phase_transition": phase_transition,
            "solution": active_solution,
        }

    def _run_parametrized_phase(
        self,
        label_prefix,
        depower,
        start_state,
        pattern_config,
        phase_sym=False,
    ):
        """Run a parametrized phase simulation and return the PhaseParameterized object."""
        sim_type = "quasi steady"
        print(f"Running simulation for {sim_type} with label: {label_prefix}")

        self.system_model.input_depower = depower
        phase = PhaseParameterized(
            self.system_model,
            quasi_steady=True,
            pattern_config=pattern_config,
        )
        if phase_sym:
            phase.run_simulation_phase(start_state=start_state)
        else:
            phase.run_simulation(start_state=start_state)
        return phase

    @staticmethod
    def _merge_phase_dicts(primary, secondary):
        def coerce_copy(value):
            if isinstance(value, list):
                return value.copy()
            if isinstance(value, tuple):
                return list(value)
            return value

        def to_list(value):
            if isinstance(value, list):
                return value.copy()
            if isinstance(value, tuple):
                return list(value)
            return [value]

        merged = {}
        for source in (primary, secondary):
            if not source:
                continue
            for key, value in source.items():
                prepared = coerce_copy(value)
                if key not in merged:
                    merged[key] = prepared
                    continue
                existing = merged[key]
                merged[key] = to_list(existing) + to_list(prepared)
        return merged

    @staticmethod
    def _copy_phase_dict(source):
        if not source:
            return {}

        def coerce(value):
            if isinstance(value, list):
                return value.copy()
            if isinstance(value, tuple):
                return list(value)
            return value

        return {key: coerce(value) for key, value in source.items()}
