from plot_import_and_params import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
np.seterr(all="ignore")
from TOV import txt_to_arr

"""This file contains the plotting routines to create the mass-radius plots found throughout the thesis. Most of the 
lines of code here are for nice plot formatting. As most of the plots require their own formatting to be acceptably 
pretty, we dedicate one function for each of them.

Note that the naming conventions follows the thesis: hybrid is the first order phase transition, while unified
is the interpolating phase.

The functions require mrp-iterable inputs. These are arrays of (M, R, p_c)-tuples which we calculate with 
get_mass_radius_p_c_triplets() from TOV.py. The best way to do this, is to calculate the mrp-array, save it as a .txt
with arr_to_txt(). When we plot, we read the .txt-files with txt_to_arr(), also found in TOV.py."""


def color_plot_insert(fig, ax, cax, mrp_arr_iterable, title, cmap_str="viridis", instability=None, orientation="vertical"):
    """Assumes the array mass_radius_pressure_arr is sorted by pressure.
    NB: For some strange reason, this does not work with np.seterr("raise"). Must be changed before function call.
    This function is used as a template for many mass-radius plots"""
    if instability:
        print("instability pressure provided")
        assert len(instability) == len(mrp_arr_iterable)

    # Finding maximum p_c:
    max_pc = max(max(arr[:, 2]) for arr in mrp_arr_iterable)
    min_pc = min(min(arr[:, 2]) for arr in mrp_arr_iterable)

    # Makes a norm for numbers to be mapped to colours.
    color_norm = colors.LogNorm(vmin=min_pc, vmax=max_pc)
    # Choosing which colour map to use
    color_map = plt.get_cmap(cmap_str)
    # Creates a mapping function from a number to a spesific colour map.
    number_to_color_map = cm.ScalarMappable(norm=color_norm, cmap=color_map)

    for j in range(len(mrp_arr_iterable)):
        if instability:
            add_curve_with_color(mrp_arr_iterable[j], number_to_color_map, ax, instability[j])
        else:
            add_curve_with_color(mrp_arr_iterable[j], number_to_color_map, ax)
    cbar = fig.colorbar(number_to_color_map, cax=cax, orientation=orientation, pad=0.065, aspect=40)
    cbar.ax.set_ylabel(r"Central pressure $p_c \, / \, \unit{\pascal}$", rotation=90)

    ax.set_ylabel(r"$M \, / \,  M_\odot$")
    ax.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    ax.grid(which="minor", alpha=0.2)
    ax.grid(which="major", alpha=0.5)

    ax.set_title(r"{}".format(title), fontdict={"fontsize": 10})
    return fig, ax, number_to_color_map


def add_curve_with_color(arr, scalarMap, ax, unstable_p=None):
    """arr is an array on the form of mass_radius_pressure_arr.
    scaLarMap is an instance of the ScalarMappable from matplotlib.cm
    ax is where we would like to plot the data from arr."""
    colorvals = scalarMap.to_rgba(arr[:, 2])
    for i in range(len(arr) - 1):
        # Division by 1000 because we want km
        ax.plot(arr[i:i+2, 1] / 1000, arr[i:i+2, 0], color=colorvals[i])
    if unstable_p:
        mrp_over_unstable_p = [(M, R, p) for (M, R, p) in zip(arr[:, 0], arr[:, 1], arr[:, 2]) if p - unstable_p > 0]
        print(mrp_over_unstable_p[0])
        ax.scatter(mrp_over_unstable_p[0][1] / 1000, mrp_over_unstable_p[0][0], color="black", marker="x", zorder=2)
    return ax


def color_plot_different_color_maps(mrp_pair, mrp_names_pair, color_map_name_pair):
    """Used to create the second plot in chapter 5.2.
    mrp_pair passed are mrp-arrays for NR improved and analytic."""
    fig, ax = plt.subplots(nrows=1, figsize=(6, 4.5))
    scalarMaps = []
    positioning, rotation = ("vertical", "vertical"), (90, 90)
    for n, (mrp_arr, cmap_name) in enumerate(zip(mrp_pair, color_map_name_pair)):
        vmin, vmax = min(mrp_arr[:, 2]), max(mrp_arr[:, 2])
        lognorm = colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        scalarMap = cm.ScalarMappable(norm=lognorm, cmap=cmap)
        scalarMaps.append(scalarMap)
        add_curve_with_color(mrp_arr, scalarMap, ax)
        cbar = fig.colorbar(scalarMap, orientation=positioning[n], pad=0.01, aspect=30)
        cbar.ax.set_ylabel(r"{}, $p_c \, / \, \unit{}$".format(mrp_names_pair[n], "{\pascal}"), rotation=rotation[n])

    ax.set_xlim(0.95 / 1000 * min(min(arr[:, 1]) for arr in mrp_pair), 1.025 / 1000 * max(max(arr[:, 1]) for arr in mrp_pair))
    ax.set_ylim(max(0.35, 0.95 * min(min(arr[:, 0]) for arr in mrp_pair)), 1.05 * max(max(arr[:, 0]) for arr in mrp_pair))
    ax.set_ylabel(r"$M \, / \, M_\odot$")
    ax.set_xlabel(r"$R\, / \, \unit{\kilo \metre}$")
    ax.grid()
    ax.set_title(r"Mass-radius relations for ideal neutron stars", fontdict={"fontsize": 10})

    plt.tight_layout()
    plt.savefig("Mass_radius_exact_and_low_p_bar_tight.eps", format="eps", dpi=600)
    plt.show()


def plot_ideal(arrs):
    """Used to create the first plot in chapter 5.2. The argument arrs must contain mrp-arrays for
    UR improved, NR, NR improved and analytic."""
    fig, (ax, cax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [1, 0.03]}, figsize=(5.6, 7.1))
    _, _, num_to_col = color_plot_insert(fig, ax, cax, arrs, r"Mass-radius relations for ideal neutron stars",
                                         orientation="horizontal")
    max_R = 1 / 1000 * max(max(arr[:, 1]) for arr in arrs)
    max_M = max(max(arr[:, 0]) for arr in arrs)

    # Manipulating ticks (In a sensible way for max_R around 40:
    x_ticks_large = np.arange(0, 40 + 10, 10)
    x_ticks_small = np.arange(0, 40 + 5, 5)

    ax.set_xticks(x_ticks_large)
    ax.set_xticks(x_ticks_small, minor=True)

    ax.set_yticks(np.arange(0, 1 + 0.2, 0.2))
    ax.set_yticks(np.arange(0, 1 + 0.1, 0.1), minor=True)

    ax.set_xlim(0, 1.05 * max_R)
    ax.set_ylim(0, min(1, 1.05 * max_M))

    # Next, we annotate the figure.
    # NR peak:
    ax.annotate(r"$M = {} M_{}$,{}$R = {} \,  \unit{}$,{}$p_c = {} \, \unit{}$.".format("0.97", "\odot", "\n", "8.0",
                                                                                        "{\kilo \metre}", "\n",
                                                                                        r"\num{1.1d35}", "{\pascal}"),
                xy=(8.0, 0.9655), xytext=(80, -39), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=180,angleB=90"))
    # NR label:
    ax.annotate("NR limit", xy=(11.6, 0.83), xytext=(100, -4), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # UR improved peak:
    ax.annotate(r"$M = {} M_{}$,{}$R = {} \, \unit{}$,{}$p_c = {} \, \unit{}$.".format("0.41", "\odot", "\n", "3.8", "{\kilo \metre}", "\n",
                                                                       r"\num{2.7d35}", "{\pascal}"),
                xy=(3.8, 0.41), xytext=(7, 0.08), textcoords="data",
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=180,angleB=90"))
    # It would seem angleB is the angle that the arrow connects to the point with, and angleA is the angle the
    # line leaves the textbox with.
    # UR improved label:
    ax.annotate(r"Large $\{}$-expansion".format("bar{p}"), xy=(5.32, 0.25), xytext=(35, -4), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # Analytic peak:
    ax.annotate(r"$M = {}M_{}$,{}$R = {} \, \unit{}$,{}$p_c = {} \, \unit{}$.".format("0.71", "\odot", "\n", "9.2", "{\kilo \metre}", "\n", r"\num{3.6d34}", "{\pascal}"),
                xy=(9.18, 0.713), xytext=(80, -50), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=180,angleB=90"))
    # Analytic label:
    ax.annotate("Analytic", xy=(14.5, 0.55), xytext=(100, -4), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    # plt.savefig("Mass_radius_vertical_cbar.eps", format="eps", dpi=600)
    plt.show()


def plot_quark_star_bag_compare(mrp_arr_iterable, bag_shifts, fname):
    """This function is used to create the first plot in chapter 11.4.
    For the annotations to be placed correctly, we must use m_sigma = 700 and bag shifts (69 MeV)^4, (96 MeV)^4,
    and (117 MeV)^4 in the incr quark-meson model equation of state."""
    assert len(mrp_arr_iterable) == len(bag_shifts)
    fig = plt.figure(figsize=(5, 4), layout="constrained")
    ax, cax = fig.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.03]})
    _, _, _ = color_plot_insert(fig, ax, cax, mrp_arr_iterable,
                                r"Mass-radius relations for quark stars, $m_\sigma = 700 \, \unit{\mega \eV}$")
    for mrp_arr in mrp_arr_iterable:
        mrp_max = max([(M, R, p) for M, R, p in zip(mrp_arr[:, 0], mrp_arr[:, 1], mrp_arr[:, 2])],
                      key=lambda triplet: triplet[0])
        print(mrp_max)
        ax.scatter(mrp_max[1]/1000, mrp_max[0], s=10, zorder=3, color="black")
    # Converting from units of bag shifts in dimensionless for to dimensionfull: * 93^4.
    # Adding annotations
    bag1, bag2, bag3 = round((bag_shifts[0] * 93**4)**(1/4)), round((bag_shifts[1] * 93**4)**(1/4)), round((bag_shifts[2] * 93**4)**(1/4))
    ax.annotate(r"$B=({} \, \unit{})^4$".format(bag1, "{\mega \eV}"), xy=(10.35, 1.95), xytext=(7, 1.935),
                textcoords="data",
                arrowprops=dict(arrowstyle="->"), size=8)
    ax.annotate(r"$B=({} \, \unit{})^4$".format(bag2, "{\mega \eV}"), xy=(9.85, 1.1), xytext=(7, 1.085),
                textcoords="data",
                arrowprops=dict(arrowstyle="->"), size=8)
    ax.annotate(r"$B=({} \, \unit{})^4$".format(bag3, "{\mega \eV}"), xy=(8.7, 0.9), xytext=(7, 0.885),
                textcoords="data",
                arrowprops=dict(arrowstyle="->"), size=8)
    ax.set_xlim(6.5, 12)
    ax.set_ylim(0.4, 2.1)
    plt.savefig(fname=fname, format="eps", dpi=600)
    plt.show()


def plot_quark_star_incr(mrp_arr_iterable, fname):
    """Creates the second plot in chapter 11.4. mrp_arr_iterable are for m_sigma = 600, 700, 800, minimal bag
    constant."""
    fig = plt.figure(figsize=(5.5, 5.5), layout="constrained")
    ax, cax = fig.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.03]})
    _, _, _ = color_plot_insert(fig, ax, cax, mrp_arr_iterable,
                                r"Mass-radius relations for quark stars, inconsistent renormalising")
    for mrp_arr in mrp_arr_iterable:
        mrp_max = max([(M, R, p) for M, R, p in zip(mrp_arr[:, 0], mrp_arr[:, 1], mrp_arr[:, 2])],
                      key=lambda triplet: triplet[0])
        print(mrp_max)
        ax.scatter(mrp_max[1] / 1000, mrp_max[0], s=10, zorder=3, color="black")

    ax.annotate(r"$m_\sigma = 600 \, \unit{\mega \eV}$", xy=(10.25, 1.05), xytext=(6.5, 1.035),
                textcoords="data", arrowprops=dict(arrowstyle="->"), size=9)
    ax.annotate(r"$m_\sigma = 700 \, \unit{\mega \eV}$", xy=(9.55, 0.75), xytext=(11, 0.735),
                textcoords="data", arrowprops=dict(arrowstyle="->"), size=9)
    ax.annotate(r"$m_\sigma = 800 \, \unit{\mega \eV}$", xy=(12.6, 1.45), xytext=(11.67, 1.035),
                textcoords="data", arrowprops=dict(arrowstyle="->"), size=9)

    ax.set_xlim(6, 15)
    ax.set_ylim(0.25, 2.1)

    plt.savefig(fname=fname, format="eps", dpi=600)
    plt.show()


def plot_quark_star_cr(mrp_arr_iterable, sigma_masses, fname):
    """Creates the mass-radius plot in chapter 11.5. mrp-arrays for m_sigma = 400, 500, 550, and 600. Minimal bag
    constant."""
    superfig = plt.figure(figsize=(6, 5), layout="constrained")
    mrp_fig, zoom_legend_fig = superfig.subfigures(ncols=2, width_ratios=[0.65, 0.35])
    ax, cax = mrp_fig.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.03]})
    zoom, legend_fig = zoom_legend_fig.subfigures(nrows=2, height_ratios=[0.5, 0.5])
    ax2 = zoom.subplots()
    superfig.suptitle("Consistent renormalising", fontsize=10)

    # Zoom box configuring:
    M_lower, M_upper = 1.7, 2.05
    R_lower, R_upper = 10, 12
    square_aspect = (R_upper - R_lower) / (M_upper - M_lower)
    ax2.axes.set_aspect(aspect=square_aspect * 1.5)

    ax.plot((R_lower, R_upper, R_upper, R_lower, R_lower), (M_lower, M_lower, M_upper, M_upper, M_lower),
            color="black", linewidth=1.0)
    _, _, scalar_map = color_plot_insert(mrp_fig, ax, cax, mrp_arr_iterable,
                                r"Mass-radius relations for quark stars")
    cmap = cm.get_cmap("plasma")
    norm = colors.Normalize(vmin=0, vmax=3)
    num_to_col = cm.ScalarMappable(norm, cmap)

    point_handles = []

    for n, (sigma_mass, mrp_arr) in enumerate(zip(sigma_masses, mrp_arr_iterable)):
        mrp_max = max([(M, R, p) for M, R, p in zip(mrp_arr[:, 0], mrp_arr[:, 1], mrp_arr[:, 2])],
                      key=lambda triplet: triplet[0])
        print(mrp_max)
        top_point = ax2.scatter(mrp_max[1] / 1000, mrp_max[0], s=10, zorder=3, color=num_to_col.to_rgba(n),
                               label=r"$m_\sigma = {} \, \unit{}$".format(sigma_mass, "{\mega \eV}"))
        ax.scatter(mrp_max[1] / 1000, mrp_max[0], s=10, zorder=3, color=num_to_col.to_rgba(n),
                     label=r"$m_\sigma = {} \, \unit{}$".format(sigma_mass, "{\mega \eV}"))
        add_curve_with_color(mrp_arr, scalar_map, ax2)
        point_handles.append(top_point)

    ax.annotate(r"$m_\sigma = 400 \, \unit{\mega \eV}$", xy=(10.3, 1.05), xytext=(6.2, 1.035),
                textcoords="data", arrowprops=dict(arrowstyle="->"), size=8)
    ax.annotate(r"$m_\sigma = 500 \, \unit{\mega \eV}$", xy=(9, 0.65), xytext=(10.8, 0.635),
                textcoords="data", arrowprops=dict(arrowstyle="->"), size=8)
    ax.annotate(r"$m_\sigma = 550 \, \unit{\mega \eV}$", xy=(9.15, 0.55), xytext=(10.8, 0.535),
                textcoords="data", arrowprops=dict(arrowstyle="->"), size=8)
    ax.annotate(r"$m_\sigma = 600 \, \unit{\mega \eV}$", xy=(13.15, 1), xytext=(10.8, 0.77),
                textcoords="data", arrowprops=dict(arrowstyle="->",
                                                   connectionstyle="angle,angleA=90,angleB=180"),
                size=8)

    legend_fig.legend(handles=point_handles, fontsize="small", handlelength=1.0, loc="upper center")

    ax.set_xlim(6, 15)
    ax.set_ylim(0.25, 2.1)

    ax2.set_xlim(R_lower, R_upper)
    ax2.set_ylim(M_lower, M_upper)
    ax2.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    ax2.set_ylabel(r"$M \, / \, M_\odot$")
    ax2.grid()
    ax2.set_title("Zoom-in on mass maxima", fontdict={"fontsize": 10})

    plt.savefig(fname=fname, format="eps", dpi=600)
    plt.show()


def hybrid_star_color_plot(mrp_arr_iterable, fname="", instability=None):
    """Creates the mass-radius plot in chapter 12.2.
    NB: This function requires that the mrp_arry for APR matter comes FIRST. Then it assumes that m_sigma = 400 follows,
    500, and finally 600. This is for the annotations on the curves to be correct."""
    name_and_color = {"Pure APR": "black", r"Hybrid, $m_\sigma=400 \, \unit{\mega \eV}$": "red",
                      r"Hybrid, $m_\sigma=500 \, \unit{\mega \eV}$": "blue",
                      r"Hybrid, $m_\sigma=600 \, \unit{\mega \eV}$": "yellow"}
    superfig = plt.figure(layout="constrained", figsize=(6.5, 4.5))
    fig1, fig2 = superfig.subfigures(ncols=2, width_ratios=[1.05, 0.6])
    ax1, ax2 = fig1.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.04]})
    subfig1, subfig2 = fig2.subfigures(nrows=2, height_ratios=[0.7, 0.3])
    ax3 = subfig1.subplots()
    _, _, num_to_col = color_plot_insert(superfig, ax1, ax2, mrp_arr_iterable,
                                         "Mass-radius relations for hybrid stars")
    ax3.plot((0, 1), (0, 1), color="black")
    # APR-matter becomes non-causal at p = 7.0 * 10^34
    p_non_causal = 6.283 * 10**34   # APR for shifted m_n turns non-causal
    APR_index = 0
    M_non_causal = np.interp(p_non_causal, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 0])
    R_non_causal = np.interp(p_non_causal, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 1]/1000)
    cross = ax1.scatter(R_non_causal, M_non_causal, color="black", marker="x", s=10, zorder=3,
                        label="Non-causal APR-matter")
    print("(M [M_odot], R [km], p_c [Pa]) for non-causal APR-matter: ({}, {}, {})".format(M_non_causal, R_non_causal,
                                                                                          p_non_causal))
    ax3.scatter(R_non_causal, M_non_causal, color="black", marker="x", s=15, zorder=3)
    lower_M, upper_M = 1.9, 2.3
    lower_R, upper_R = 10, 11.5
    ax1.plot((lower_R, upper_R, upper_R, lower_R, lower_R), (lower_M, lower_M, upper_M, upper_M, lower_M),
             color="black", linewidth=0.8)
    ax3.set_xlim(lower_R, upper_R)
    ax3.set_ylim(lower_M, upper_M)
    ax3.set_ylabel(r"$M \, / \, M_\odot$")
    ax3.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    square_aspect = (upper_R - lower_R) / (upper_M - lower_M)   # Using this gives a sqare plot
    ax3.axes.set_aspect(aspect=square_aspect * 1.4)
    ax3.set_title("Zoom-in on maximum masses", fontdict={"fontsize": 10})

    handles = []
    for name, mrp_arr in zip(name_and_color.keys(), mrp_arr_iterable):
        add_curve_with_color(mrp_arr, num_to_col, ax3)
        max_m_r = max([(mass, radius) for mass, radius in zip(mrp_arr[:, 0], mrp_arr[:, 1])],
                      key=lambda mr_pair: mr_pair[0])
        dot = ax3.scatter(max_m_r[1] / 1000, max_m_r[0], color=name_and_color[name], label=name,
                          zorder=3, s=10)
        handles.append(dot)
    subfig2.legend(handles=handles, fontsize="small", loc="upper right")
    ax3.grid()
    ax1.set_xlim(8, 15)
    ax1.set_ylim(0.25, 2.35)
    ax1.legend(handles=[cross], fontsize="small", loc="lower left")
    # At last, annotations:
    ax1.annotate(r"Quark core branches", xy=(10, 1.75), xytext=(8.6, 1.25), textcoords="data",
                 arrowprops=dict(arrowstyle="->"), size=8)
    ax1.annotate(r"Pure APR branch", xy=(11, 2.2), xytext=(12.3, 2.185), textcoords="data",
                 arrowprops=dict(arrowstyle="->"), size=8)
    plt.savefig(fname=fname, format="eps", dpi=600)
    plt.show()


def unified_star_colour_plot(mrp_arr_iterable, fname):
    """Creates the first mass-radius plot in chapter 12.3. m_n = 900 MeV.
    This is exactly as for the hybrid star, but this time for a unified star.
    The zoom-in box and the plot boundaries have been changed."""
    name_and_color = {"Pure APR": "black", r"Unified, $m_\sigma=400 \, \unit{\mega \eV}$": "red",
                      r"Unified, $m_\sigma=500 \, \unit{\mega \eV}$": "blue",
                      r"Unified, $m_\sigma=600 \, \unit{\mega \eV}$": "yellow"}
    superfig = plt.figure(layout="constrained", figsize=(6.5, 4.5))
    superfig.suptitle(r"$m_n = 900 \, \unit{\mega \eV}$", fontsize=10)
    fig1, fig2 = superfig.subfigures(ncols=2, width_ratios=[1.05, 0.6])
    ax1, ax2 = fig1.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.04]})
    subfig1, subfig2 = fig2.subfigures(nrows=2, height_ratios=[0.7, 0.3])
    ax3 = subfig1.subplots()
    _, _, num_to_col = color_plot_insert(superfig, ax1, ax2, mrp_arr_iterable,
                                         "Mass-radius relations for unified hybrid stars")
    ax3.plot((0, 1), (0, 1), color="black")
    # APR-matter becomes non-causal at p = 7.0 * 10^34
    p_non_causal = 6.283 * 10 ** 34  # APR for shifted m_n turns non-causal
    p_unified = 2.3 * 10 ** 33  # The splitting into branches happens at p = 2.3 * 10^33 Pa.
    APR_index = 0
    M_non_causal = np.interp(p_non_causal, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 0])
    R_non_causal = np.interp(p_non_causal, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 1] / 1000)
    M_unified = np.interp(p_unified, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 0])
    R_unified = np.interp(p_unified, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 1] / 1000)
    print("(M [M_odot], R [km], p_c [Pa]) for non-causal APR-matter: ({}, {}, {})".format(M_non_causal, R_non_causal,
                                                                                          p_non_causal))
    cross = ax1.scatter(R_non_causal, M_non_causal, color="black", marker="x", s=10, zorder=3,
                        label="Non-causal APR-matter")
    plus = ax1.scatter(R_unified, M_unified, color="black", marker="$+$", s=15, zorder=3,
                       label="Unified EoS")
    ax1.legend(handles=[plus, cross], fontsize="small", loc="lower left", handlelength=0.8)
    ax3.scatter(R_non_causal, M_non_causal, color="black", marker="x", s=15, zorder=3)
    lower_M, upper_M = 1.6, 2.0
    lower_R, upper_R = 10, 12.6
    ax1.plot((lower_R, upper_R, upper_R, lower_R, lower_R), (lower_M, lower_M, upper_M, upper_M, lower_M),
             color="black", linewidth=0.8)
    ax3.set_xlim(lower_R, upper_R)
    ax3.set_ylim(lower_M, upper_M)
    ax3.set_ylabel(r"$M \, / \, M_\odot$")
    ax3.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    square_aspect = (upper_R - lower_R) / (upper_M - lower_M)  # Using this gives a sqare plot
    ax3.axes.set_aspect(aspect=square_aspect * 1.2)
    ax3.set_title("Zoom-in on maximum masses", fontdict={"fontsize": 10})

    handles = []
    for name, mrp_arr in zip(name_and_color.keys(), mrp_arr_iterable):
        add_curve_with_color(mrp_arr, num_to_col, ax3)
        max_m_r = max([(mass, radius) for mass, radius in zip(mrp_arr[:, 0], mrp_arr[:, 1])],
                      key=lambda mr_pair: mr_pair[0])
        dot = ax3.scatter(max_m_r[1] / 1000, max_m_r[0], color=name_and_color[name], label=name,
                          zorder=3, s=10)
        handles.append(dot)
    handles.pop(0)  # Removing APR-dot legend for this plot
    subfig2.legend(handles=handles, fontsize="small", loc="upper right")
    ax3.grid()
    ax1.set_xlim(8, 16)
    ax1.set_ylim(0.2, 2.35)
    # At last, annotations:
    ax1.annotate(r"Quark-hybrid branches", xy=(9.8, 1.76), xytext=(8.05, 1.17), textcoords="data",
                 arrowprops=dict(arrowstyle="->"), size=8)
    ax1.text(8.05, 1.09, r"$m_\sigma \in \{400,\, 500 \} \, \unit{\mega \eV}$", size=8)
    ax1.annotate(r"Quark-hybrid branch", xy=(12.35, 1.375),
                 xytext=(12.5, 1.125),
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=90,angleB=180"), size=8)
    ax1.text(12.5, 1.045, r"$m_\sigma = 600 \, \unit{\mega \eV}$", size=8)
    ax1.annotate(r"Pure APR branch", xy=(10.75, 2.23), xytext=(12.3, 2.215), textcoords="data",
                 arrowprops=dict(arrowstyle="->"), size=8)
    plt.savefig(fname=fname, format="eps", dpi=600)
    plt.show()


def unified_star_colour_plot_non_shifted(mrp_arr_iterable, fname):
    """Creates the second mass-radius plot in chapter 12.3. m_n = 939.6 MeV. Otherwise, as the function above."""
    name_and_color = {"Pure APR": "black", r"Unified, $m_\sigma=400 \, \unit{\mega \eV}$": "red",
                      r"Unified, $m_\sigma=500 \, \unit{\mega \eV}$": "blue",
                      r"Unified, $m_\sigma=600 \, \unit{\mega \eV}$": "yellow"}
    superfig = plt.figure(layout="constrained", figsize=(6.5, 4.5))
    superfig.suptitle(r"$m_n = 939.6 \, \unit{\mega\eV}$", fontsize=10)
    fig1, fig2 = superfig.subfigures(ncols=2, width_ratios=[1.05, 0.6])
    ax1, ax2 = fig1.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.04]})
    subfig1, subfig2 = fig2.subfigures(nrows=2, height_ratios=[0.7, 0.3])
    ax3 = subfig1.subplots()
    _, _, num_to_col = color_plot_insert(superfig, ax1, ax2, mrp_arr_iterable,
                                         "Mass-radius relations for unified hybrid stars")
    p_non_causal = 7.04 * 10 ** 34  # APR for non-shifted m_n turns non-causal.
    p_unified = 2.33 * 10 ** 33  # The splitting into branches happens at p = 2.33 * 10^33 Pa for non-shifted m_n.
    APR_index = 0
    M_non_causal = np.interp(p_non_causal, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 0])
    R_non_causal = np.interp(p_non_causal, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 1] / 1000)
    M_unified = np.interp(p_unified, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 0])
    R_unified = np.interp(p_unified, mrp_arr_iterable[APR_index][:, 2], mrp_arr_iterable[APR_index][:, 1] / 1000)
    print("(M [M_odot], R [km], p_c [Pa]) for non-causal APR-matter: ({}, {}, {})".format(M_non_causal, R_non_causal,
                                                                                          p_non_causal))
    cross = ax1.scatter(R_non_causal, M_non_causal, color="black", marker="x", s=10, zorder=3,
                        label="Non-causal APR-matter")
    plus = ax1.scatter(R_unified, M_unified, color="black", marker="$+$", s=15, zorder=3,
                       label="Unified EoS")
    ax1.legend(handles=[plus, cross], fontsize="small", loc="lower left", handlelength=0.8)
    ax3.scatter(R_non_causal, M_non_causal, color="black", marker="x", s=15, zorder=3)
    lower_M, upper_M = 1.3, 1.75
    lower_R, upper_R = 8.25, 10.6
    ax1.plot((lower_R, upper_R, upper_R, lower_R, lower_R), (lower_M, lower_M, upper_M, upper_M, lower_M),
             color="black", linewidth=0.8)
    ax3.set_xlim(lower_R, upper_R)
    ax3.set_ylim(lower_M, upper_M)
    ax3.set_ylabel(r"$M \, / \, M_\odot$")
    ax3.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    square_aspect = (upper_R - lower_R) / (upper_M - lower_M)  # Using this gives a sqare plot
    ax3.axes.set_aspect(aspect=square_aspect * 1.3)
    ax3.set_title("Zoom-in on maximum masses", fontdict={"fontsize": 10})

    handles = []
    for name, mrp_arr in zip(name_and_color.keys(), mrp_arr_iterable):
        add_curve_with_color(mrp_arr, num_to_col, ax3)
        max_m_r = max([(mass, radius) for mass, radius in zip(mrp_arr[:, 0], mrp_arr[:, 1])],
                      key=lambda mr_pair: mr_pair[0])
        dot = ax3.scatter(max_m_r[1] / 1000, max_m_r[0], color=name_and_color[name], label=name,
                          zorder=3, s=10)
        handles.append(dot)
    handles.pop(0)  # Removing APR-dot legend for this plot
    subfig2.legend(handles=handles, fontsize="small", loc="upper right")
    ax3.grid()
    ax1.set_xlim(6.5, 13.5)
    ax1.set_ylim(0.25, 2.25)
    # At last, annotations:
    ax1.annotate(r"Quark-hybrid branches", xy=(11, 1.15), xytext=(7.8, 1.0), textcoords="data",
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=90,angleB=0"), size=8)
    ax1.text(7.8, 0.92, r"$m_\sigma \in \{400,\, 500 \} \, \unit{\mega \eV}$", size=8)
    ax1.annotate(r"Quark-hybrid branch", xy=(10.04, 1.71),
                 xytext=(6.6, 1.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=270"), size=8)
    ax1.text(6.6, 1.82, r"$m_\sigma = 600 \, \unit{\mega \eV}$", size=8)
    ax1.annotate(r"Pure APR branch", xy=(10.55, 2.125), xytext=(7.1, 2.11), textcoords="data",
                 arrowprops=dict(arrowstyle="->"), size=8)
    plt.savefig(fname=fname, format="eps", dpi=600)
    plt.show()


def summarising_plot(mrp_dict, line_props_dict, fname=""):
    """Creates the mass-radius plot in the Master's thesis summary, chapter 13.1.
    mrp_dict is a dictionary with names and mrp_arrays. line_prop_dict is a dictionary of names and
    line colour and style for plotting."""
    superfig = plt.figure(figsize=(6, 4.5), layout="constrained")
    fig, legend_superfig = superfig.subfigures(ncols=2, width_ratios=[1, 0.30])
    ax = fig.subplots()
    ax.set_title("Mass-radius relations summary", fontdict={"fontsize": 10})
    fig_legend1, fig_legend2, fig_legend3 = legend_superfig.subfigures(nrows=3, height_ratios=[0.3, 0.1, 0.6])
    # Empirical data:
    star_1_lower, star_1_upper = 1.93, 2.01
    star_2_lower, star_2_upper = 1.97, 2.05
    star_3_lower, star_3_upper = 2.18, 2.52
    common_R_lower, common_R_upper = 9.9, 11.2
    common_M_lower, common_M_upper = 1.17, 2.0
    star_ranges = [[star_1_lower, star_1_upper], [star_2_lower, star_2_upper], [star_3_lower, star_3_upper]]
    colors = {"star1": "green", "star2": "yellow", "star3": "red", "common": "black"}

    # Calculated data:
    p_non_causal_shifted = 6.28 * 10 ** 34
    p_non_causal_unshifted = 7.04 * 10 ** 34

    # Name APR shifted neutron mass "APR shifted" and APR unshifted neutron mass "APR unshifted".
    M_non_causal_shift = np.interp(p_non_causal_shifted, mrp_dict["APR shifted"][:, 2], mrp_dict["APR shifted"][:, 0])
    M_non_causal_unshift = np.interp(p_non_causal_unshifted, mrp_dict["APR unshifted"][:, 2],
                                     mrp_dict["APR unshifted"][:, 0])
    R_non_causal_shift = np.interp(p_non_causal_shifted, mrp_dict["APR shifted"][:, 2],
                                   mrp_dict["APR shifted"][:, 1] / 1000)
    R_non_causal_unshift = np.interp(p_non_causal_unshifted, mrp_dict["APR unshifted"][:, 2],
                                     mrp_dict["APR unshifted"][:, 1] / 1000)
    ax.scatter(R_non_causal_unshift, M_non_causal_unshift, color="black", marker="x", s=15, zorder=3)
    cross = ax.scatter(R_non_causal_shift, M_non_causal_shift, color="black", marker="x", s=15, zorder=3,
                       label="Non-causal APR-matter")

    p_branches_shifted = 2.3 * 10 ** 33  # Branching in unified hybrid model
    p_branches_unshifted = 2.33 * 10 ** 33
    M_branches_shift = np.interp(p_branches_shifted, mrp_dict["APR shifted"][:, 2], mrp_dict["APR shifted"][:, 0])
    M_branches_unshift = np.interp(p_branches_unshifted, mrp_dict["APR unshifted"][:, 2],
                                     mrp_dict["APR unshifted"][:, 0])
    R_branches_shift = np.interp(p_branches_shifted, mrp_dict["APR shifted"][:, 2],
                                   mrp_dict["APR shifted"][:, 1] / 1000)
    R_branches_unshift = np.interp(p_branches_unshifted, mrp_dict["APR unshifted"][:, 2],
                                     mrp_dict["APR unshifted"][:, 1] / 1000)
    ax.scatter(R_branches_unshift, M_branches_unshift, color="black", marker=r"$+$", s=15, zorder=3)
    plus = ax.scatter(R_branches_shift, M_branches_shift, color="black", marker=r"$+$", s=15, zorder=3,
                      label="Unified branching")
    ax_handles = [cross, plus]
    # Relevant limits to fit a practical portion of each curve inside ax:
    x_lim_lower, x_lim_upper = 7.5, 13
    y_lim_lower, y_lim_upper = 0.4, 2.35
    ax.set_xlim(x_lim_lower, x_lim_upper)
    ax.set_ylim(y_lim_lower, y_lim_upper)
    ax.set_ylabel(r"$M \, / \, M_\odot$")
    ax.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    thin_line, thicker_line = 1.0, 1.2
    patch_handles = []
    # Zip cuts away the last element when one list is one longer than the other:
    counter = 0
    for (lim_lower, lim_upper), star_name in zip(star_ranges, colors.keys()):
        counter += 1
        R_points = (x_lim_lower, x_lim_upper)   # R-values limiting the plot
        ax.plot(R_points, (lim_lower, lim_lower), color=colors[star_name], linewidth=thin_line, alpha=0.5)   # Adding lower M-limit line
        ax.plot(R_points, (lim_upper, lim_upper), color=colors[star_name], linewidth=thin_line, alpha=0.5)   # Adding upper M-limit line
        ax.fill_between(R_points, (lim_lower, lim_lower), (lim_upper, lim_upper), color=colors[star_name],
                                alpha=0.3)
        patch = Patch(color=colors[star_name], alpha=0.3,
                      label=r"$\text{}_{}$".format("{Star}", "{" + r"\textcolor{}".format("{" + colors[star_name] + "}{" + str(counter) + "}}")))
        patch_handles.append(patch)
    fig_legend3.legend(handles=patch_handles, loc="upper left", fontsize="small", handlelength=1.5)
    M_points = (y_lim_lower, y_lim_upper)   # M-values limiting the plot
    # ax.plot((common_R_lower, common_R_lower), M_points, color=colors["common"], linewidth=thin_line)  # Lower common R-line
    # ax.plot((common_R_upper, common_R_upper), M_points, color=colors["common"], linewidth=thin_line)  # Upper common R-line
    ax.fill_between((common_R_lower, common_R_upper), (y_lim_lower, y_lim_lower), (y_lim_upper, y_lim_upper), color=colors["common"], alpha=0.3)

    dashed_line = Line2D([0], [0], linewidth=thicker_line, linestyle=(0, (3, 3)), color="silver",
                         label=r"$m_\sigma = 500 \, \unit{\mega \eV}$")
    solid_line = Line2D([0], [0], linewidth=thicker_line, linestyle="solid", color="silver",
                        label=r"$m_\sigma = 600 \, \unit{\mega \eV}$")
    handles = [dashed_line, solid_line]
    APR_shifted = Line2D([0], [0], linewidth=thicker_line, color="black", label=r"APR shifted $m_n$")
    APR_unshifted = Line2D([0], [0], linewidth=thicker_line, color="dimgrey", label=r"APR original $m_n$")
    APR_handles = [APR_shifted, APR_unshifted]

    for n, name in enumerate(mrp_dict.keys()):
        label_name = name.split(" ")[0]
        Rs, Ms = mrp_dict[name][:, 1] / 1000, mrp_dict[name][:, 0]
        colour, linelook = line_props_dict[name]
        line, = ax.plot(Rs, Ms, color=colour, linestyle=linelook, linewidth=thicker_line, label=label_name)
        if linelook == "solid" and n > 1:
            # Excluding n = 0, 1 to not add APR-lines
            handles.append(line)
    fig_legend1.legend(handles=handles, loc="lower left", fontsize="small", handlelength=1.5)
    lgnd = fig_legend2.legend(handles=APR_handles, loc="upper left", fontsize="small", handlelength=1.5)
    ax.legend(handles=ax_handles, loc="lower left", fontsize="small", handlelength=1.0)
    ax.grid(alpha=0.2)
    plt.savefig(fname=fname, format="svg", dpi=600)
    plt.show()


def front_page_plot(mrp_dict):
    """As the name suggests, this is the front page plot. mrp_dict is a dictionary with a name which appears in the
    legend, paired with an array of (Mass, radius, central pressure)-tuples."""
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    norm = colors.Normalize(vmin=0, vmax=len(mrp_dict) - 1)
    num_to_col = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("viridis"))
    counter = 0
    for name, mrp_arr in mrp_dict.items():
        ax.plot(mrp_arr[:, 1] / 1000, mrp_arr[:, 0], color=num_to_col.to_rgba(counter), label=name)
        counter += 1
    ax.legend(fontsize="small", handlelength=1.2, loc="upper left")
    ax.set_xlim(3, 14)
    ax.set_ylim(0.3, 2.3)
    ax.grid(alpha=0.5)
    ax.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    ax.set_ylabel(r"$M \, / \, M_\odot$")
    ax.set_title("Mass-Radius Relations for Compact Stars", fontdict={"fontsize": 10})
    plt.tight_layout()
    plt.savefig(fname="Front_page_plot.pdf", format="pdf", dpi=600)
    plt.show()
