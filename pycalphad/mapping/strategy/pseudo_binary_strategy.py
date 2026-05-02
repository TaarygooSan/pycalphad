"""
PseudoBinaryStrategy

A mapping strategy for pseudo-binary phase diagrams such as CaO-SiO2, where
the system contains more than two elements, but the overall composition is
constrained to move along a straight line (join) in mole-fraction space.

Coordinate convention
---------------------
The pseudo-binary join is parameterised by z ∈ [0, 1]:

    X_i(z) = X_i(0) * (1 - z) + X_i(1) * z

where X_i(0) and X_i(1) are the element mole fractions at the two endpoints
(``comp_start`` and ``comp_end`` respectively).

The *primary* element is the one whose mole fraction spans the widest range
across the join; it is used as the composition axis variable (v.X(primary)).
All remaining elements whose fractions vary along the join are *secondary*
elements; their conditions are derived from the primary at every step so the
system is always constrained to the join.

Example – CaO-SiO₂
-------------------
::

    from pycalphad import Database, variables as v
    from pycalphad.mapping import PseudoBinaryStrategy
    from pycalphad.mapping import plot_binary

    dbf = Database('cao_sio2.tdb')
    comps = ['CA', 'SI', 'O', 'VA']
    phases = list(dbf.phases.keys())

    # SiO2 endpoint (z=0): 1 mol SiO2 → 1/3 Si, 2/3 O
    # CaO endpoint  (z=1): 1 mol CaO  → 1/2 Ca, 1/2 O
    comp_start = {'CA': 0.0,  'SI': 1/3, 'O': 2/3}   # pure SiO2
    comp_end   = {'CA': 0.5,  'SI': 0.0, 'O': 0.5 }   # pure CaO

    conds = {v.T: (1200, 3000, 10), v.P: 101325}

    strategy = PseudoBinaryStrategy(
        dbf, comps, phases, conds,
        comp_start=comp_start,
        comp_end=comp_end,
    )
    strategy.do_map()
    ax = plot_binary(strategy)
"""

from typing import Union
import logging
import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.core.utils import get_pure_elements

from pycalphad.mapping.primitives import (
    Node, Point, ExitHint, Direction, MIN_COMPOSITION,
)
from pycalphad.mapping.starting_points import point_from_equilibrium
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.binary_strategy import BinaryStrategy
from pycalphad.mapping.strategy.step_strategy import StepStrategy

_log = logging.getLogger(__name__)


class PseudoBinaryStrategy(BinaryStrategy):
    """
    Phase-diagram mapping strategy for pseudo-binary systems.

    In a pseudo-binary system the overall composition moves along a straight
    *join* in mole-fraction space as a function of the pseudo-binary
    parameter z ∈ [0, 1].  The user defines the two endpoints of the join
    (``comp_start`` at z = 0, ``comp_end`` at z = 1).  The strategy then:

    * chooses the element with the largest compositional range as the *primary*
      axis composition variable;
    * derives all other (secondary) element mole fractions from the primary via
      the linear join relation at every step; and
    * uses standard ZPF-line tracing (inherited from
      :class:`~pycalphad.mapping.strategy.binary_strategy.BinaryStrategy`)
      to find phase boundaries in (T, X_primary) space.

    Parameters
    ----------
    dbf : Database
    components : list[str]
    phases : list[str]
    conditions : dict
        Must contain at least a temperature range ``v.T: (T_min, T_max, dT)``
        and pressure ``v.P: value``.  Composition conditions are constructed
        automatically from *comp_start*, *comp_end* and *z_limits*.
        Any pre-existing composition conditions in *conditions* are silently
        replaced by the join-derived values.
    comp_start : dict[str, float]
        Element mole fractions at z = 0 (the "left" / low-z endpoint).
        Keys are element names (case-insensitive); values must sum to 1.
    comp_end : dict[str, float]
        Element mole fractions at z = 1 (the "right" / high-z endpoint).
        Keys are element names (case-insensitive); values must sum to 1.
    z_limits : tuple[float, float], optional
        Lower and upper bounds of z to map.  Defaults to ``(0, 1)``.
    z_step : float, optional
        Step size as a fraction of the full z range [0, 1].
        Defaults to 0.02 (50 steps across the full range).
    **kwargs
        Forwarded to :class:`MapStrategy` (e.g. ``models``,
        ``phase_record_factory``, ``GLOBAL_MIN_PDENS``, …).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        dbf: Database,
        components: list[str],
        phases: list[str],
        conditions: dict,
        *,
        comp_start: dict[str, float],
        comp_end: dict[str, float],
        z_limits: tuple[float, float] = (0.0, 1.0),
        z_step: float = 0.02,
        **kwargs,
    ):
        # Normalise element-name keys to uppercase.
        self.comp_start: dict[str, float] = {k.upper(): float(v_) for k, v_ in comp_start.items()}
        self.comp_end:   dict[str, float] = {k.upper(): float(v_) for k, v_ in comp_end.items()  }

        if set(self.comp_start) != set(self.comp_end):
            raise ValueError(
                "comp_start and comp_end must contain the same set of elements."
            )
        self.z_limits: tuple[float, float] = (float(z_limits[0]), float(z_limits[1]))

        # ------------------------------------------------------------------
        # Identify primary element (largest |ΔX| across the join).
        # Secondary elements are those that also change but are not the axis.
        # Elements with zero range (constant mole fraction) are excluded from
        # conditions entirely—pycalphad will derive them via X_sum = 1.
        # ------------------------------------------------------------------
        ranges = {
            el: abs(self.comp_end[el] - self.comp_start[el])
            for el in self.comp_start
        }
        self.primary_el: str = max(ranges, key=ranges.get)
        if ranges[self.primary_el] == 0:
            raise ValueError(
                "All element mole fractions are constant across the join; "
                "cannot build a pseudo-binary diagram."
            )
        self.primary_var = v.X(self.primary_el.capitalize())

        # Secondary: elements that vary (excluding primary) AND excluding the
        # element whose fraction equals (1 - sum_of_others), i.e. the one
        # that is implicitly derived by pycalphad.
        # We include ALL varying elements except the one we choose to omit
        # (which is the element with the second-largest range, to maximise
        # numerical stability — but for correctness, we can simply include all
        # varying secondary elements; pycalphad will handle over-specification
        # gracefully since X_sum = 1 is always a hard constraint).
        #
        # In practice pycalphad needs exactly C-1 independent composition
        # conditions.  We provide the primary and all secondary elements that
        # are present in the database as elements, letting pycalphad determine
        # the remaining one from the sum-to-one constraint.
        pure_els = {el.upper() for el in get_pure_elements(dbf, components)}

        self.secondary_vars: dict = {}   # var → (x_start, x_end)
        for el, x_start in self.comp_start.items():
            if el == self.primary_el:
                continue
            if el not in pure_els:
                continue
            x_end = self.comp_end[el]
            if abs(x_end - x_start) > 0:                       # only varying
                var = v.X(el.capitalize())
                self.secondary_vars[var] = (x_start, x_end)

        # ------------------------------------------------------------------
        # Build full conditions dict:
        #   • strip any existing X conditions from the caller's dict
        #   • add X(primary) as an axis variable
        #   • add X(secondary) as single-value (fixed) conditions at z_limits[0]
        # ------------------------------------------------------------------
        full_conds = {
            key: val for key, val in conditions.items()
            if not isinstance(key, v.X)
        }

        z_lo, z_hi = self.z_limits
        x_p_start = self.comp_start[self.primary_el]
        x_p_end   = self.comp_end  [self.primary_el]
        x_p_lo = x_p_start * (1 - z_lo) + x_p_end * z_lo
        x_p_hi = x_p_start * (1 - z_hi) + x_p_end * z_hi
        # Axis-variable step size for primary comp (as mole fraction)
        x_p_step = abs(x_p_end - x_p_start) * z_step

        # Ensure ordering: low ≤ high
        x_p_lo, x_p_hi = min(x_p_lo, x_p_hi), max(x_p_lo, x_p_hi)
        full_conds[self.primary_var] = (x_p_lo, x_p_hi, x_p_step)

        # Secondary elements: single value at z = z_lo
        for sec_var, (xs, xe) in self.secondary_vars.items():
            full_conds[sec_var] = xs * (1 - z_lo) + xe * z_lo

        super().__init__(dbf, components, phases, full_conds, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _x_primary_to_z(self, x_primary: float) -> float:
        """Convert primary-element mole fraction to pseudo-binary parameter z."""
        x_start = self.comp_start[self.primary_el]
        x_end   = self.comp_end  [self.primary_el]
        denom = x_end - x_start
        if abs(denom) < 1e-15:
            return 0.0
        return (x_primary - x_start) / denom

    def _z_to_secondary_conditions(self, z: float) -> dict:
        """
        Return a dict {var: value} for all secondary composition variables
        corresponding to pseudo-binary parameter *z*.
        """
        return {
            var: xs * (1 - z) + xe * z
            for var, (xs, xe) in self.secondary_vars.items()
        }

    def _build_conds_at_z(self, z: float, T_val: float) -> dict:
        """
        Return a complete conditions dict (single-point, no ranges) for a
        given z and temperature, suitable for passing to
        :func:`point_from_equilibrium`.
        """
        x_primary = (
            self.comp_start[self.primary_el] * (1 - z)
            + self.comp_end  [self.primary_el] * z
        )
        # Clamp slightly inside composition bounds to avoid pure-component
        # singularities.
        x_primary = max(x_primary, MIN_COMPOSITION)
        x_primary = min(x_primary, 1.0 - MIN_COMPOSITION)

        conds = {
            key: val for key, val in self.conditions.items()
            if not isinstance(key, v.X)        # strip all X conditions
            and not map_utils.is_state_variable(key)  # strip T, P, N, ...
        }
        # Re-add state variables as single values
        for key, val in self.conditions.items():
            if map_utils.is_state_variable(key):
                conds[key] = np.atleast_1d(val)[0]  # take first value if range

        conds[v.T] = T_val
        conds[self.primary_var] = x_primary
        for var, val in self._z_to_secondary_conditions(z).items():
            conds[var] = val

        if v.N not in conds:
            conds[v.N] = 1
        return conds

    # ------------------------------------------------------------------
    # Core overrides – maintaining the join constraint
    # ------------------------------------------------------------------

    def _step_conditions(
        self,
        point: Point,
        axis_var: v.StateVariable,
        axis_delta: float,
        axis_lims: tuple,
        direction: Direction,
    ):
        """
        Override: when stepping in the primary composition variable, also
        update all secondary composition conditions to maintain the join.
        """
        new_conds, hit_limit = super()._step_conditions(
            point, axis_var, axis_delta, axis_lims, direction
        )

        if axis_var == self.primary_var and not hit_limit:
            z = self._x_primary_to_z(new_conds[axis_var])
            new_conds.update(self._z_to_secondary_conditions(z))

        return new_conds, hit_limit

    def _take_step(
        self,
        point: Point,
        axis_var: v.StateVariable,
        axis_delta: float,
        axis_lims: tuple,
        direction: Direction,
    ):
        """
        Override: after any equilibrium step, project the secondary
        composition conditions back onto the join.

        When stepping in the primary composition direction the join constraint
        is already maintained by :meth:`_step_conditions`.  When stepping in
        the potential direction (T) pycalphad frees the primary composition
        variable; the returned primary X may not lie exactly on the join.
        The post-hoc correction here keeps the conditions dict consistent for
        subsequent steps.
        """
        result = super()._take_step(point, axis_var, axis_delta, axis_lims, direction)
        if result is None:
            return None

        new_point, orig_cs = result

        # Update secondary conditions from whatever X(primary) was found
        x_primary = float(new_point.get_property(self.primary_var))
        z = self._x_primary_to_z(x_primary)
        for sec_var, val in self._z_to_secondary_conditions(z).items():
            new_point.global_conditions[sec_var] = val

        return new_point, orig_cs

    # ------------------------------------------------------------------
    # Starting-point generation
    # ------------------------------------------------------------------

    def generate_automatic_starting_points(self):
        """
        Find starting points for ZPF tracing along all four axis boundaries.

        * **Composition boundaries** (X_primary = x_min or x_max):
          Step in T at fixed composition, updating secondary X to match the
          boundary value.  Uses the standard :class:`StepStrategy`.

        * **Potential boundaries** (T = T_min or T_max):
          Scan the join at fixed T.  Because StepStrategy does not know about
          the join constraint, we instead directly query equilibrium at
          evenly-spaced z values and look for phase changes.
        """
        T_var = next(
            (av for av in self.axis_vars if map_utils.is_state_variable(av)),
            None,
        )
        comp_var = self.primary_var

        # --- Composition boundaries: step in T at fixed X(primary) ----------
        for x_val in self.axis_lims[comp_var]:
            conds = copy.deepcopy(self.conditions)
            conds[comp_var] = max(x_val, MIN_COMPOSITION) if x_val < MIN_COMPOSITION else x_val

            # Update secondary variables to the value corresponding to this x_val
            z = self._x_primary_to_z(conds[comp_var])
            for sec_var, val in self._z_to_secondary_conditions(z).items():
                conds[sec_var] = val

            # Make T an axis (step variable)
            if T_var is not None:
                T_range = np.amax(self.axis_lims[T_var]) - np.amin(self.axis_lims[T_var])
                conds[T_var] = (
                    self.axis_lims[T_var][0],
                    self.axis_lims[T_var][1],
                    T_range / 20,
                )

            map_kwargs = self._constant_kwargs()
            step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
            step.do_map()
            self.add_starting_points_from_step(step)

        # --- Potential boundaries: scan z at fixed T -----------------------
        if T_var is not None:
            for T_val in self.axis_lims[T_var]:
                self._scan_join_at_fixed_t(T_val)

    def _scan_join_at_fixed_t(self, T_val: float, n_scan: int = 20):
        """
        Scan the pseudo-binary join at a fixed temperature *T_val*.

        Equilibrium is computed at *n_scan + 1* evenly-spaced z values.
        When a change in the stable phase set is detected between adjacent
        points, the two-phase point is added as a starting node in both
        step directions.
        """
        z_lo, z_hi = self.z_limits
        z_values = np.linspace(z_lo, z_hi, n_scan + 1)

        prev_point: Point | None = None

        for z in z_values:
            conds = self._build_conds_at_z(z, T_val)
            point = point_from_equilibrium(
                self.dbf, self.components, self.phases, conds,
                models=self.models,
                phase_record_factory=self.phase_records,
            )

            if point is not None and prev_point is not None:
                curr_phases  = frozenset(point.stable_phases)
                prev_phases  = frozenset(prev_point.stable_phases)
                if curr_phases != prev_phases and len(point.stable_composition_sets) == 2:
                    _log.info(
                        f"T={T_val}: phase change at z≈{z:.4f}: "
                        f"{set(prev_phases)} → {set(curr_phases)}"
                    )
                    for d in (Direction.POSITIVE, Direction.NEGATIVE):
                        node = self._create_node_from_point(
                            point, None, None, d, ExitHint.POINT_IS_EXIT
                        )
                        self.node_queue.add_node(node, True)

            prev_point = point

    # ------------------------------------------------------------------
    # Public helpers for z ↔ x_primary conversion (useful for plotting)
    # ------------------------------------------------------------------

    def z_to_x_primary(self, z: float) -> float:
        """Convert pseudo-binary parameter z to primary-element mole fraction."""
        x_start = self.comp_start[self.primary_el]
        x_end   = self.comp_end  [self.primary_el]
        return x_start * (1 - z) + x_end * z

    def x_primary_to_z(self, x_primary: float) -> float:
        """Convert primary-element mole fraction to pseudo-binary parameter z."""
        return self._x_primary_to_z(x_primary)
