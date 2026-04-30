"""
Single-sublattice Modified Quasichemical Model (MQM) in the Pair Approximation.

This module provides the ``ModelMQM`` class, a pycalphad-compatible thermodynamic
model for liquid solutions where short-range ordering is described by nearest-
neighbour pair exchange reactions.  Typical applications are oxide melts such as
the CaO-SiO2 system.

The formulation follows Pelton & Blander [1]_ and Pelton *et al.* [2]_.  The key
difference from the two-sublattice MQMQA (quadruplet approximation) is that MQM
uses a *single sublattice* and *pairs* as its internal structural unit.

Thermodynamic framework
-----------------------
One formula unit is defined as **one mole of pairs**.  With pair fractions
:math:`X_{ij}` (summing to 1 by the internal constraint), the Gibbs energy per
formula unit is:

.. math::

    G = G_{\\text{ref}} + G_{\\text{config}} + G_{\\text{xs}}

**Reference energy**

.. math::

    G_{\\text{ref}} = \\sum_i n_i G^\\circ_i

where

.. math::

    n_i = \\frac{2 X_{ii} + \\sum_{j \\neq i} X_{ij}}{Z_i}

is the moles of component *i* per mole of pairs and :math:`Z_i` is its
coordination number.

**Configurational entropy** (SUBG-type pair entropy, no quadruplet term)

.. math::

    G_{\\text{config}} = RT \\left[
        \\sum_i n_i \\ln X_i
        + \\sum_{i \\leq j} X_{ij} \\ln \\frac{X_{ij}}{C_{ij}\\, Y_i\\, Y_j}
    \\right]

with :math:`C_{ij} = 2` for :math:`i \\neq j` and 1 otherwise, and the
coordination-equivalent fraction

.. math::

    Y_i = \\frac{2 X_{ii} + \\sum_{j \\neq i} X_{ij}}{2}

**Excess energy** (binary pair exchange, Redlich-Kister-like)

.. math::

    G_{\\text{xs}} = \\sum_{i < j} \\frac{X_{ij}}{2}\\,
        \\Delta G^\\circ_{ij}\\, \\chi_{ij}^{p}\\, \\chi_{ji}^{q}

where :math:`\\chi_{ij} = Y_i / (Y_i + Y_j)`.

Database setup
--------------
All phases that use this model must have:

* ``model_hints['mqm']`` set in the ``Database``; it must contain a
  ``'coordination_numbers'`` dict mapping each ``v.Species`` component to its
  coordination number :math:`Z_i`.
* **Reference parameters** of type ``MQMPG`` with ``constituent_array = [[A]]``
  and an optional ``stoichiometry`` list (defaults to ``[1.0]``).
* **Excess parameters** of type ``MQMPX`` with ``constituent_array = [[A, B]]``
  (alphabetically sorted), an ``exponents`` list ``[p, q]`` (defaults to
  ``[0, 0]``).

Example (CaO-SiO2)::

    from pycalphad import Database
    import pycalphad.variables as v
    from symengine import S

    dbf = Database()
    cao = v.Species("CAO", constituents={"CAO": 1.0})
    sio2 = v.Species("SIO2", constituents={"SIO2": 1.0})
    for sp in [cao, sio2]:
        dbf.species.add(sp)
    dbf.elements |= {"CAO", "SIO2"}
    dbf.refstates["CAO"]  = {"mass": 56.077, "phase": "BLANK", "H298": 0, "S298": 0}
    dbf.refstates["SIO2"] = {"mass": 60.084, "phase": "BLANK", "H298": 0, "S298": 0}

    model_hints = {"mqm": {"coordination_numbers": {cao: 6.0, sio2: 4.0}}}
    dbf.add_phase("LIQUID", model_hints=model_hints, sublattices=[1.0])
    dbf.add_phase_constituents("LIQUID", [["CAO", "SIO2"]])

    dbf.add_parameter("MQMPG", "LIQUID", [["CAO"]],  None, -600000 + 200*v.T)
    dbf.add_parameter("MQMPG", "LIQUID", [["SIO2"]], None, -855000 + 300*v.T)
    dbf.add_parameter("MQMPX", "LIQUID", [["CAO", "SIO2"]], None,
                      -120000 + 5*v.T, exponents=[0, 0])

References
----------
.. [1] Pelton, A.D. & Blander, M. (1987). Thermodynamic analysis of ordered
       liquid alloys with a modified quasichemical model.
       *CALPHAD* 11(4) 371–395. doi:10.1016/0364-5916(87)90046-7
.. [2] Pelton, A.D., Chartrand, P., & Eriksson, G. (2001).
       The modified quasi-chemical model: Part IV. Two-sublattice quadruplet
       approximation. *Metall. Mater. Trans. A* 32(6) 1409–1416.
       doi:10.1007/s11661-001-0230-7
"""

import itertools
from collections import OrderedDict
from functools import partial
from typing import List

from symengine import S, Symbol, log
from tinydb import where

import pycalphad.variables as v
from pycalphad import Model
from pycalphad.core.errors import DofError
from pycalphad.core.utils import unpack_species, wrap_symbol
from pycalphad.model import _MAX_PARAM_NESTING


def _get_pair_species(i, j) -> v.Species:
    """Return a canonically-sorted ``Species`` representing the pair *(i, j)*.

    Both orderings ``(i, j)`` and ``(j, i)`` map to the same ``Species``
    object, so pair fractions stored as ``v.Y`` site fractions are unique.

    Parameters
    ----------
    i, j : v.Species or str
        The two components forming the pair.

    Returns
    -------
    v.Species
        A species whose name is the alphabetically-sorted concatenation of the
        two component names and whose ``constituents`` dict counts how many
        times each component name appears.
    """
    if all(isinstance(c, v.Species) for c in [i, j]):
        names = sorted([c.name for c in [i, j]])
    else:
        names = sorted([str(i), str(j)])
    name = "".join(names)
    constituent_dict: dict = {}
    for c in names:
        constituent_dict[c] = constituent_dict.get(c, 0.0) + 1.0
    return v.Species(name, constituents=constituent_dict)


class ModelMQM(Model):
    """Single-sublattice Modified Quasichemical Model (MQM) in the Pair Approximation.

    See the module docstring for a full description of the formulation, the
    required database layout, and a worked example.

    This class is only semantically a subclass of ``Model``.  It implements the
    full pycalphad Model API in a self-contained way without relying on the
    ``Model`` superclass internals.  Subclassing ``Model`` is kept only to
    satisfy ``isinstance`` checks in the broader pycalphad infrastructure.
    """

    contributions = [
        ("ref",   "reference_energy"),
        ("idmix", "ideal_mixing_energy"),
        ("xsmix", "excess_mixing_energy"),
    ]

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, dbe, comps, phase_name, parameters=None):
        self._dbe = dbe
        self._reference_model = None
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = tuple(list(phase.sublattices))

        active_species = unpack_species(dbe, comps)

        # MQM requires exactly one sublattice
        if len(dbe.phases[phase_name].constituents) != 1:
            raise ValueError(
                f"MQM model requires exactly one sublattice, "
                f"got {len(dbe.phases[phase_name].constituents)} for phase {phase_name!r}."
            )

        constituents_set = (
            set(dbe.phases[phase_name].constituents[0])
            .intersection(active_species)
        )

        if len(constituents_set) == 0:
            raise DofError(
                f"{self.phase_name}: Sublattice has no active species "
                f"from components {comps}."
            )

        self.components = sorted(constituents_set)

        # All canonical pairs (i, j) with i ≤ j
        self._pairs = list(itertools.combinations_with_replacement(self.components, 2))
        pair_species_list = [_get_pair_species(A, B) for A, B in self._pairs]
        self.constituents = [sorted(pair_species_list)]

        # Symbol substitution table
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}

        if parameters is not None:
            self._parameters_arg = parameters
            if isinstance(parameters, dict):
                symbols.update(
                    [(wrap_symbol(s), val) for s, val in parameters.items()]
                )
            else:
                for s in parameters:
                    symbols.pop(wrap_symbol(s))
        else:
            self._parameters_arg = None

        self._symbols = {wrap_symbol(key): value for key, value in symbols.items()}

        self.models = OrderedDict()
        self.build_phase(dbe)

        for name, value in self.models.items():
            self.models[name] = self.symbol_replace(value, symbols)

    # ------------------------------------------------------------------ #
    # Equality / hashing                                                   #
    # ------------------------------------------------------------------ #

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) != type(other):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))

    # ------------------------------------------------------------------ #
    # Core properties                                                      #
    # ------------------------------------------------------------------ #

    @property
    def state_variables(self) -> List[v.StateVariable]:
        """Sorted list of state variables (non-SiteFraction) in the AST."""
        return sorted(
            (x for x in self.ast.free_symbols
             if not isinstance(x, v.SiteFraction) and isinstance(x, v.StateVariable)),
            key=str,
        )

    @property
    def site_fractions(self) -> List[v.SiteFraction]:
        """Sorted list of site fractions (pair fractions) in the AST."""
        return sorted(
            (x for x in self.ast.free_symbols if isinstance(x, v.SiteFraction)),
            key=str,
        )

    @property
    def ast(self):
        """Full abstract syntax tree of the model (sum of all contributions)."""
        return sum(self.models.values())

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pair_test(self, constituent_array):
        """Return ``True`` if *constituent_array* is a single-species single-sublattice pair.

        Used to find MQMPG (pure-component reference) parameters.
        """
        if len(constituent_array) != 1:
            return False
        if len(constituent_array[0]) != 1:
            return False
        return constituent_array[0][0] in self.components

    def _array_validity(self, constituent_array):
        """Return ``True`` if all species in *constituent_array* are active components."""
        for subl in constituent_array:
            for constituent in subl:
                if constituent not in self.components:
                    return False
        return True

    def _X_ij(self, i: v.Species, j: v.Species) -> v.SiteFraction:
        """Return the pair-fraction site fraction :math:`X_{ij}` for the pair *(i, j)*."""
        return v.Y(self.phase_name, 0, _get_pair_species(i, j))

    def _bonds_i(self, i: v.Species):
        """Return the bond-count for component *i*:

        .. math::

            \\text{bonds}_i = 2 X_{ii} + \\sum_{j \\neq i} X_{ij}

        Because :math:`\\sum_{kl} X_{kl} = 1`, we have
        :math:`\\sum_i \\text{bonds}_i = 2`.
        """
        bonds = S.Zero
        for a, b in self._pairs:
            bonds += self._X_ij(a, b) * ((a == i) + (b == i))
        return bonds

    def Z(self, dbe, species: v.Species) -> float:
        """Return the coordination number of *species* from ``model_hints``."""
        return dbe.phases[self.phase_name].model_hints["mqm"]["coordination_numbers"][species]

    def _n_i(self, dbe, species: v.Species):
        """Moles of component *species* per mole of pairs:

        .. math::

            n_i = \\frac{\\text{bonds}_i}{Z_i}
        """
        return self._bonds_i(species) / self.Z(dbe, species)

    def _X_i(self, dbe, species: v.Species):
        """Mole fraction of component *species* (among all non-VA components)."""
        return self._n_i(dbe, species) / sum(
            self._n_i(dbe, s) for s in self.components
        )

    def _Y_i(self, species: v.Species):
        """Coordination-equivalent fraction:

        .. math::

            Y_i = \\frac{\\text{bonds}_i}{2}

        :math:`\\sum_i Y_i = 1` because :math:`\\sum_{ij} X_{ij} = 1`.
        """
        return self._bonds_i(species) / 2

    # ------------------------------------------------------------------ #
    # Constraints & normalisation                                          #
    # ------------------------------------------------------------------ #

    def get_internal_constraints(self):
        """Return the pair-fraction normalisation constraint :math:`\\sum_{ij} X_{ij} - 1 = 0`."""
        pair_sum = S.Zero
        for i, j in self._pairs:
            pair_sum += self._X_ij(i, j)
        return [pair_sum - 1]

    @property
    def normalization(self):
        """Total moles of (non-VA) components per mole of pairs.

        Dividing *G* (energy / mole-pairs) by ``normalization`` gives *GM*
        (energy / mole-components).
        """
        return sum(
            self._n_i(self._dbe, i)
            for i in self.components
            if "VA" not in i.constituents
        )

    def moles(self, species, per_formula_unit=False):
        """Moles of element or pseudo-element *species*.

        For each active MQM component that contains the requested element,
        the contribution :math:`n_i` is accumulated.  Divide by
        ``normalization`` to obtain the mole fraction.

        Parameters
        ----------
        species : str or v.Species
            Element or pseudo-element name (e.g. ``"CA"``, ``"CAO"``).
        per_formula_unit : bool, optional
            If ``True``, return the raw amount (per mole of pairs) without
            dividing by ``normalization``.  Defaults to ``False``.
        """
        species_obj = v.Species(species)
        element = list(species_obj.constituents.keys())[0]
        result = S.Zero
        for comp in self.components:
            if element in comp.constituents:
                result += self._n_i(self._dbe, comp)
        if per_formula_unit:
            return result
        return result / self.normalization

    # ------------------------------------------------------------------ #
    # Thermodynamic quantities (standard pycalphad API)                   #
    # ------------------------------------------------------------------ #

    degree_of_ordering = DOO = S.Zero
    curie_temperature   = TC  = S.Zero
    beta                = BMAG = S.Zero
    neel_temperature    = NT  = S.Zero

    GM            = property(lambda self: self.ast / self.normalization)
    G             = property(lambda self: self.ast)
    energy        = GM
    entropy       = SM    = property(lambda self: -self.GM.diff(v.T))
    enthalpy      = HM    = property(lambda self: self.GM - v.T * self.GM.diff(v.T))
    heat_capacity = CPM   = property(lambda self: -v.T * self.GM.diff(v.T, v.T))
    mixing_energy       = GM_MIX  = property(lambda self: self.GM - self.reference_model.GM)
    mixing_enthalpy     = HM_MIX  = property(lambda self: self.GM_MIX - v.T * self.GM_MIX.diff(v.T))
    mixing_entropy      = SM_MIX  = property(lambda self: -self.GM_MIX.diff(v.T))
    mixing_heat_capacity= CPM_MIX = property(lambda self: -v.T * self.GM_MIX.diff(v.T, v.T))

    # ------------------------------------------------------------------ #
    # Contribution methods                                                 #
    # ------------------------------------------------------------------ #

    def reference_energy(self, dbe):
        """Reference energy :math:`G_{\\text{ref}} = \\sum_i n_i G^\\circ_i`.

        Looks up parameters of type ``MQMPG`` with
        ``constituent_array = [[species_i]]``.  An optional ``stoichiometry``
        field (list of floats, default ``[1.0]``) scales the energy.
        """
        query = (
            (where("phase_name") == self.phase_name)
            & (where("parameter_type") == "MQMPG")
            & (where("constituent_array").test(self._pair_test))
        )
        params = dbe._parameters.search(query)
        terms = S.Zero
        for param in params:
            component = param["constituent_array"][0][0]
            if component not in self.components:
                continue
            n_A   = self._n_i(dbe, component)
            G_A   = param["parameter"]
            stoich = param.get("stoichiometry", [1.0])
            terms += n_A * G_A / stoich[0]
        return terms

    def ideal_mixing_energy(self, dbe):
        """Configurational entropy (SUBG-type pair entropy):

        .. math::

            G_{\\text{config}} = RT \\left[
                \\sum_i n_i \\ln X_i
                + \\sum_{i \\leq j} X_{ij}\\, \\ln
                  \\frac{X_{ij}}{C_{ij}\\, Y_i\\, Y_j}
            \\right]

        where :math:`C_{ij} = 2` for :math:`i \\neq j` and 1 for
        :math:`i = j`.
        """
        n_i  = partial(self._n_i, dbe)
        X_i  = partial(self._X_i, dbe)
        Y_i  = self._Y_i
        X_ij = self._X_ij

        Sid = S.Zero
        # Component (reference-composition) terms
        for A in self.components:
            Sid += n_i(A) * log(X_i(A))
        # Pair-correlation terms
        for i, j in self._pairs:
            C_ij = 1 + int(i != j)   # 2 for off-diagonal, 1 for diagonal
            Sid += X_ij(i, j) * log(X_ij(i, j) / (C_ij * Y_i(i) * Y_i(j)))
        return Sid * v.T * v.R

    def excess_mixing_energy(self, dbe):
        """Excess energy from pair exchange reactions.

        Looks up parameters of type ``MQMPX`` with
        ``constituent_array = [[A, B]]`` (alphabetically sorted, ``A < B``).
        Each parameter may carry an ``exponents`` key ``[p, q]`` for the
        Redlich–Kister-like expansion :math:`\\chi_{AB}^p \\chi_{BA}^q`.

        .. math::

            G_{\\text{xs}} = \\sum_{A < B}
                \\frac{X_{AB}}{2}\\,
                \\Delta G^\\circ_{AB}(T)\\,
                \\chi_{AB}^{p}\\, \\chi_{BA}^{q}
        """
        query = (
            (where("phase_name") == self.phase_name)
            & (where("parameter_type") == "MQMPX")
            & (where("constituent_array").test(self._array_validity))
        )
        params = dbe._parameters.search(query)

        Y_i  = self._Y_i
        X_ij = self._X_ij

        energy = S.Zero
        for param in params:
            ca = param["constituent_array"]
            if len(ca) != 1 or len(ca[0]) != 2:
                continue
            A, B = ca[0][0], ca[0][1]
            if A not in self.components or B not in self.components:
                continue

            exponents = param.get("exponents", [0, 0])
            p_alpha = exponents[0]
            q_alpha = exponents[1]

            Y_A = Y_i(A)
            Y_B = Y_i(B)
            Chi_AB = Y_A / (Y_A + Y_B)
            Chi_BA = Y_B / (Y_A + Y_B)

            mixing_term = Chi_AB ** p_alpha * Chi_BA ** q_alpha
            energy += X_ij(A, B) * param["parameter"] * mixing_term / 2

        return energy

    # ------------------------------------------------------------------ #
    # Build / symbol helpers                                               #
    # ------------------------------------------------------------------ #

    def build_phase(self, dbe):
        """Populate ``self.models`` with each contribution term."""
        self.models.clear()
        for key, value in self.__class__.contributions:
            self.models[key] = S(getattr(self, value)(dbe))

    @staticmethod
    def symbol_replace(obj, symbols):
        """Substitute *symbols* into *obj* up to ``_MAX_PARAM_NESTING`` times."""
        try:
            for _ in range(_MAX_PARAM_NESTING):
                obj = obj.xreplace(symbols)
                undefs = [
                    x for x in obj.free_symbols
                    if not isinstance(x, v.StateVariable)
                ]
                if not undefs:
                    break
        except AttributeError:
            pass
        return obj

    def shift_reference_state(
        self, reference_states, dbe, contrib_mods=None,
        output=("GM", "HM", "SM", "CPM"), fmt_str="{}R",
    ):
        raise NotImplementedError(
            "shift_reference_state is not implemented for ModelMQM."
        )

    def _build_reference_model(self, preserve_ideal=True):
        raise NotImplementedError(
            "_build_reference_model is not implemented for ModelMQM."
        )

    @property
    def reference_model(self):
        raise NotImplementedError(
            "Endmember reference models do not have a physical meaning for MQM models."
        )
