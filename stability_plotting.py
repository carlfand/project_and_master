from plot_import_and_params import *
from TOV import txt_to_arr
import math     # Using math.floor()

"""This file includes the functions which we have used to generate the plots in the stability section of the thesis."""


def plot_mass_of_p_c(EoS_name, p_arr, M_arr):
    """This function creates a plot of M(log(p_c)), which is included in chapter 6.1. In this chapter, we have used
    p_arr and M_arr for pressures around 10**34.5 Pa for the ideal neutron star.
    This central pressure yields the largest mass."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    # Assumes sorted p_arr and M_arr according to increasing p and a length > 1:
    increasing_M_indices = [i for i in range(1, len(M_arr)) if M_arr[i] >= M_arr[i - 1]]    # Will lose the final M
    if M_arr[0] < M_arr[1]:
        increasing_M_indices.insert(0, 0)
    decreasing_M_indices = [i for i in range(len(M_arr) - 1) if M_arr[i] >= M_arr[i + 1]]
    if M_arr[-1] < M_arr[-2]:
        decreasing_M_indices.insert(-1, len(M_arr) - 1)
    increasing_M, decreasing_M = [M_arr[i] for i in range(len(M_arr)) if i in increasing_M_indices], \
                                 [M_arr[k] for k in range(len(M_arr)) if k in decreasing_M_indices]
    # increasing_M_pressures, decreasing_M_pressures:
    iM_p, dM_p = [p_arr[i] for i in range(len(M_arr)) if i in increasing_M_indices], \
                 [p_arr[k] for k in range(len(M_arr)) if k in decreasing_M_indices]

    # Running with plasma-theme
    colornorm = colors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("viridis")
    number_to_color = cm.ScalarMappable(norm=colornorm, cmap=cmap)
    color_list = [number_to_color.to_rgba(0.2),
                  number_to_color.to_rgba(0.8)]

    ax.plot(np.log10(iM_p), increasing_M, label=r"$\frac{dM}{dp_c} > 0$, stable", color=color_list[0])
    ax.plot(np.log10(dM_p), decreasing_M, label=r"$\frac{dM}{dp_c} < 0$, unstable", color=color_list[-1])

    # Marking some points on the curve:
    n_sep = int(max(3, len(p_arr) / 2 * 1 / 5))
    increasing_M_index = math.floor(len(increasing_M_indices) / 2) - n_sep//2
    decreasing_M_index = math.floor(len(decreasing_M_indices) / 2) - n_sep//2

    marked_p = np.log10([iM_p[increasing_M_index], iM_p[increasing_M_index + n_sep]])
    marked_M = (increasing_M[increasing_M_index], increasing_M[increasing_M_index + n_sep])
    # -1 because otherwise there is a mismatch.
    marked_p2 = np.log10([dM_p[decreasing_M_index - 1], dM_p[decreasing_M_index + n_sep - 1]])
    marked_M2 = (decreasing_M[increasing_M_index], decreasing_M[increasing_M_index + n_sep])

    rel_coord_length = np.log10(max(p_arr)) - np.log10(min(p_arr))
    rel_coord_height = max(M_arr) - min(M_arr)
    ax.annotate("", xy=(marked_p[1], marked_M[1]), xytext=(marked_p[0], marked_M[0]), xycoords="data",
                textcoords="data", size=8,
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90", color=color_list[0]))
    ax.annotate(r"$M_{+}'$", xy=(marked_p[1] - 0.06 * rel_coord_length, marked_M[1] - 0.015 * rel_coord_height),
                color=color_list[0], size=8)
    ax.annotate(r"$M_{+}$", xy=(marked_p[0] - 0.06 * rel_coord_length, marked_M[0] - 0.015 * rel_coord_height),
                color=color_list[0], size=8)

    ax.annotate("", xy=(marked_p2[1], marked_M2[1]), xytext=(marked_p2[0], marked_M2[0]), xycoords="data",
                textcoords="data", size=8,
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90", color=color_list[-1]))
    ax.annotate(r"$M_{-}'$", xy=(marked_p2[1] - 0.06 * rel_coord_length, marked_M2[1] - 0.015 * rel_coord_height),
                color=color_list[-1], size=8)
    ax.annotate(r"$M_{-}$", xy=(marked_p2[0] - 0.06 * rel_coord_length, marked_M2[0] - 0.015 * rel_coord_height),
                color=color_list[-1], size=8)

    ax.scatter(marked_p, marked_M, color=color_list[0])
    ax.scatter(marked_p2, marked_M2, color=color_list[-1])

    ax.set_xlim(min(np.log10(p_arr)), max(np.log10(p_arr)))
    ax.set_ylim(min(M_arr), max(M_arr) + rel_coord_height * 0.05)
    ax.set_xlabel(r"$\log (p_c)$")
    ax.set_ylabel(r"$M \, / \, M_\odot$")
    ax.set_title(r"{} equation of state".format(EoS_name), fontdict={"fontsize": 10})
    ax.legend(fontsize="small")
    plt.grid()
    plt.tight_layout()
    # Uncomment to save figure
    # plt.savefig("Stability_plot.svg", format="svg", dpi=600)
    plt.show()


"""Using the radial perturbation stability theory in stability_ideal_stars.py, we can save arrays of 
(p_c, omega_0, omega_1, ..., omega_n, M [solar masses], R [10 km]). We calculate these tuples in find_modes().

The plotting functions below reads a .txt-file containing such tuples.
Naturally, for other eigenfrequency-data, the functions may be used to investigate other equations of state."""


def frequency_plot(fname):
    """This function generates the eigenfrequency-log(p_c) plot in chapter 6.3."""
    # Constants:
    r_0, c = 10 ** 4, 3 * 10 ** 8
    # Fetching calculated data:
    eigenfreq_array = txt_to_arr(fname)
    # Counting number of frequencies:
    n_freq = len(eigenfreq_array[0, :]) - 3   # The array contains p_c, M and R in addition to frequencies.
    # Making a norm for the desired number of frequencies
    colors_norm = colors.Normalize(vmin=0, vmax=n_freq - 1)
    # Get colormap:
    color_map = plt.get_cmap("viridis")   # "Plasma" brings another nice colormap
    number_to_color = cm.ScalarMappable(norm=colors_norm, cmap=color_map)
    # List of colors:
    color_list = [number_to_color.to_rgba(n) for n in range(0, n_freq)]
    fig, ax = plt.subplots(figsize=(6, 4))

    omega_max = 0
    for i in range(n_freq):
        # The frequencies are normalised with a factor of r_0**2 /c ** 2
        ax.plot(np.log10(eigenfreq_array[:, 0]), c ** 2 / r_0 ** 2 * eigenfreq_array[:, i + 1], color=color_list[i],
                label=r"$\omega^2_{}$".format(i))
        # Finding maximal omega for cropping the plot.
        omega_max = max(max(eigenfreq_array[:, i + 1] * c**2/r_0**2), omega_max)
    ax.set_xlim(np.log10(eigenfreq_array[0, 0]), np.log10(eigenfreq_array[-1, 0]))    # Array sorted by p_c
    ax.set_ylim(-omega_max * 1.05, omega_max * 1.05)
    ax.set_xlabel(r"$\log(p_c)$")
    ax.set_ylabel(r"$\omega^2 \, / \, \unit{\per\square\second}$")
    ax.set_title(r"Smallest eigenfrequencies $\omega_n^2$ for ideal neutron stars".format(n_freq),
                 fontdict={"fontsize": 10})
    ax.set_xticks(np.arange(31, 43, 2), minor=True)
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.2)
    plt.legend(handlelength=1.5, fontsize="small")
    plt.tight_layout()
    # Uncomment to save figure:
    # plt.savefig("Eigenfrequencies_ideal_neutron_star.svg", format="svg", dpi=600)
    plt.show()


def mass_radius_stability_plot(fname):
    """This function creates the mass-radius plot in chapter 6.3. Unlike the other mass-radius plots, we colour the
    curve by how many unstable nodes (omega_n^2 < 0) there are."""
    # Fetching the mass-radius data
    array = txt_to_arr(fname)
    max_nodes = 5   # For the calculated range of p_c
    # Values for critical pressures investigated with bisection method.
    # These values were found using critical_p_c_for_nth_freq(), found in stability_ideal_stars.py
    critical_p_cs = [0, 3.638 * 10 ** 34, 2.05 * 10 ** 37, 7.19 * 10 ** 38, 4.48 * 10 ** 40]    # Units of Pa.
    # The usual color-handling
    colornorm = colors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("viridis")
    number_to_color = cm.ScalarMappable(norm=colornorm, cmap=cmap)
    color_list = [number_to_color.to_rgba((i / (max_nodes - 1))**(2/3)) for i in range(max_nodes)]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for n_nodes in range(max_nodes):
        # Plotting each mass and radius with colours changing when the p_c values exceed one of the critical pressures
        ax.plot([R * 10 for p_c, R in zip(array[:, 0], array[:, -1]) if p_c > critical_p_cs[n_nodes]],
                [M for p_c, M in zip(array[:, 0], array[:, -2]) if p_c > critical_p_cs[n_nodes]],
                color=color_list[n_nodes], label="{} $\omega_n^2 < 0$".format(n_nodes))
    ax.set_yticks([(i + 0.5)/10 for i in range(1, 8)], minor=True)
    ax.set_xticks([i + 2.5 for i in range(5, 25, 5)], minor=True)
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.2)
    ax.set_title("Mass-radius relations with number of unstable modes", fontdict={"fontsize": 10})
    ax.set_xlim(3.5, 27.5)
    ax.set_ylim(0.15, 0.73)

    ax.set_ylabel(r"$M \, / \, M_{\odot}$")
    ax.set_xlabel(r"$R \, / \, \unit{\kilo \metre}$")
    plt.legend(loc="upper right", handlelength=1.5, fontsize="small")
    plt.tight_layout()
    # Uncomment to save figure.
    # plt.savefig("Mass_radius_relation_unstable_modes.svg", format="svg", dpi=600)
    plt.show()
