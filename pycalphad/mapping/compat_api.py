import numpy as np

from pycalphad.mapping import BinaryStrategy, TernaryStrategy, plot_binary, plot_ternary
from pycalphad.mapping import PseudoBinaryStrategy, plot_pseudo_binary
import pycalphad.mapping.utils as map_utils

def binplot(database, components, phases, conditions, return_strategy=False, plot_kwargs=None, **map_kwargs):
    """
    Calculate the binary isobaric phase diagram.

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conditions : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For binplot only one changing composition and one potential coordinate each is supported.
    return_strategy : bool, optional
        Return the BinaryStrategy object in addition to the Axes. Defaults to False.
    map_kwargs : dict, optional
        Additional keyword arguments to BinaryStrategy().
    plot_kwargs : dict, optional
        Keyword arguments to plot_binary()
        Possible key,val pairs in plot_kwargs

            label_nodes : bool
                Whether to plot points for phases on three-phase regions
                Default = False
            tieline_color : tuple
                Color for tielines
                Default = (0,1,0,1)
            tie_triangle_color : tuple
                Color for tie triangles
                Default = (1,0,0,1)

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram
    (Axes, BinaryStrategy)
        If return_strategy is True.

    """
    indep_comps = [key for key, value in conditions.items() if not map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conditions.items() if map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 1) or (len(indep_pots) != 1):
        raise ValueError('binplot() requires exactly one composition coordinate and one potential coordinate')

    strategy = BinaryStrategy(database, components, phases, conditions, **map_kwargs)
    strategy.do_map()

    plot_kwargs = plot_kwargs if plot_kwargs is not None else dict()
    ax = plot_binary(strategy, **plot_kwargs)
    ax.grid(plot_kwargs.get("gridlines", False))

    if return_strategy:
        return ax, strategy
    else:
        return ax


def ternplot(dbf, comps, phases, conds, x=None, y=None, return_strategy=False, map_kwargs=None, **plot_kwargs):
    """
    Calculate the ternary isothermal, isobaric phase diagram.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conds : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Maps StateVariables to values and/or iterables of values.
        For ternplot only two changing composition coordinates is supported.
    x : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the x-axis.
        Must correspond to an independent condition.
    y : v.MoleFraction
        instance of a pycalphad.variables.composition to plot on the y-axis.
        Must correspond to an independent condition.
    return_strategy : bool, optional
        Return the TernaryStrategy object in addition to the Axes. Defaults to False.
    label_nodes : bool (optional)
        Whether to plot points for phases on three-phase regions
        Default = False
    tieline_color : tuple (optional)
        Color for tielines
        Default = (0,1,0,1)
    tie_triangle_color : tuple (optional)
        Color for tie triangles
        Default = (1,0,0,1)
    map_kwargs : dict, optional
        Additional keyword arguments to TernaryStrategy().
    plot_kwargs : dict, optional
        Keyword arguments to plot_ternary().

    Returns
    -------
    Axes
        Matplotlib Axes of the phase diagram
    (Axes, TernaryStrategy)
        If return_strategy is True.

    """
    indep_comps = [key for key, value in conds.items() if not map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if map_utils.is_state_variable(key) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comps) != 2) or (len(indep_pots) != 0):
        raise ValueError('ternplot() requires exactly two composition coordinates')

    map_kwargs = map_kwargs if map_kwargs is not None else dict()
    strategy = TernaryStrategy(dbf, comps, phases, conds, **map_kwargs)
    strategy.do_map()

    ax = plot_ternary(strategy, x, y, **plot_kwargs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if return_strategy:
        return ax, strategy
    else:
        return ax


def pseudobinplot(
    database,
    components,
    phases,
    conditions,
    comp_start,
    comp_end,
    z_limits=(0.0, 1.0),
    z_step=0.02,
    return_strategy=False,
    plot_kwargs=None,
    **map_kwargs,
):
    """
    Calculate and plot a pseudo-binary isobaric phase diagram.

    A *pseudo-binary* system is one in which the composition of the mixture
    is constrained to a straight line (*join*) in mole-fraction space, even
    though the underlying system has more than two elements.  Examples include
    CaO–SiO₂, Al₂O₃–SiO₂, and MgO–FeO.

    The join is defined by two endpoint compositions (*comp_start* at z = 0
    and *comp_end* at z = 1).  The x-axis of the resulting diagram shows the
    primary-element mole fraction (the element with the largest range across
    the join) together with a secondary z-axis.  The y-axis shows temperature.

    Parameters
    ----------
    database : Database
        Thermodynamic database containing the relevant parameters.
    components : Sequence[str]
        Names of components to consider in the calculation.
    phases : Sequence[str]
        Names of phases to consider in the calculation.
    conditions : Mapping[v.StateVariable, Union[float, Tuple[float, float, float]]]
        Must contain at minimum a temperature range ``v.T: (T_min, T_max, dT)``
        and pressure ``v.P: value``.  Any composition conditions present are
        ignored; they are constructed automatically from *comp_start*,
        *comp_end*, and *z_limits*.
    comp_start : dict[str, float]
        Element mole fractions at z = 0 (the low-z endpoint).
        Keys are element names (case-insensitive) and must sum to 1.
        Example for pure SiO₂: ``{'SI': 1/3, 'O': 2/3}``.
    comp_end : dict[str, float]
        Element mole fractions at z = 1 (the high-z endpoint).
        Keys are element names (case-insensitive) and must sum to 1.
        Example for pure CaO: ``{'CA': 0.5, 'O': 0.5}``.
    z_limits : tuple[float, float], optional
        Lower and upper bounds of *z* to map.  Defaults to ``(0, 1)`` (full
        join).
    z_step : float, optional
        Composition step size as a fraction of the full z range.
        Defaults to 0.02 (50 points across the join).
    return_strategy : bool, optional
        If ``True``, return ``(Axes, PseudoBinaryStrategy)`` instead of just
        ``Axes``.  Defaults to ``False``.
    plot_kwargs : dict, optional
        Keyword arguments forwarded to :func:`plot_pseudo_binary`.
        Recognised keys include ``tielines``, ``label_nodes``,
        ``tieline_color``, ``tie_triangle_color``, ``z_axis``, ``z_label``.
    **map_kwargs
        Additional keyword arguments forwarded to
        :class:`PseudoBinaryStrategy` (e.g. ``models``,
        ``GLOBAL_MIN_PDENS``).

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the phase diagram.
    (matplotlib.axes.Axes, PseudoBinaryStrategy)
        If *return_strategy* is ``True``.

    Examples
    --------
    Plot the CaO–SiO₂ pseudo-binary phase diagram::

        from pycalphad import Database, variables as v
        from pycalphad.mapping.compat_api import pseudobinplot

        dbf = Database('cao_sio2.tdb')
        comps = ['CA', 'SI', 'O', 'VA']
        phases = list(dbf.phases.keys())

        comp_start = {'CA': 0.0, 'SI': 1/3, 'O': 2/3}   # z=0 → SiO₂
        comp_end   = {'CA': 0.5, 'SI': 0.0, 'O': 0.5 }   # z=1 → CaO

        ax = pseudobinplot(
            dbf, comps, phases,
            {v.T: (1200, 2800, 10), v.P: 101325},
            comp_start=comp_start,
            comp_end=comp_end,
        )
    """
    strategy = PseudoBinaryStrategy(
        database, components, phases, conditions,
        comp_start=comp_start,
        comp_end=comp_end,
        z_limits=z_limits,
        z_step=z_step,
        **map_kwargs,
    )
    strategy.do_map()

    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
    ax = plot_pseudo_binary(strategy, **plot_kwargs)

    if return_strategy:
        return ax, strategy
    else:
        return ax