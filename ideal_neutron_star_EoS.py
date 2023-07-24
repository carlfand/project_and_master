from plot_import_and_params import *

"""The objective for this file is to plot pressure and energy density as a function of x_F. We consider a free Fermi
gas. After this, we turn to the equation of state for a free fermi gas.
We plot the analytic solution as well as expansions for small and large x_F. The functions denoted improved, takes 
into account one more order of the expansion than their 'unimproved' counterparts.
The NR-limit (non-relativistic) takes into account the leading order x_F-term when x_F is small.
The UR-limit (ultra-relativistic) takes into account the leading order x_F-term when x_F is large.

In the thesis, these plots are displayed in chapter 4.2."""


def energy_density_free_Fermi_gas_analytic(x_F, eps_g):
    """Analytic solution for the energy density for the Free fermi gas as a function of
    the dimensionless parameter x_F."""
    return eps_g / 8 * (2 * x_F**3 * np.sqrt(1 + x_F**2) + x_F * np.sqrt(1 + x_F**2) - np.arcsinh(x_F))


def pressure_free_Fermi_gas_analytic(x_F, eps_g):
    """Analytic solution for the pressure for the Free fermi gas as a function of
    the dimensionless parameter x_F."""
    return eps_g * (x_F**3 * np.sqrt(1 + x_F**2)/12 + 1/8 * (np.arcsinh(x_F) - x_F * np.sqrt(1 + x_F**2)))


def diff_pressure_free_Fermi_gas_analytic(x_F, eps_g):
    """pressure_free_Fermi_gas_analytic differentiated with respect to x_F."""
    return eps_g / 8 * (np.sqrt(1 + x_F ** 2) * (2 * x_F - 1) + 1 / np.sqrt(1 + x_F ** 2) * (2/3 * x_F ** 4 + 1))


def pressure_free_Fermi_gas_NR(x_F, eps_g):
    """Expansion for small Fermi energies, x_F << 1."""
    return eps_g * (x_F**5 / 15)


def energy_density_free_Fermi_gas_NR(x_F, eps_g):
    """Expansion for small Fermi energies, x_F << 1."""
    return eps_g * (x_F**3/3 + x_F**5/10)


def pressure_free_Fermi_gas_NR_improved(x_F, eps_g):
    """Expansion for small Fermi energies, x_F << 1."""
    return eps_g * (x_F ** 5 / 15 - x_F**7 / 42)


def energy_density_free_Fermi_gas_NR_improved(x_F, eps_g):
    """Expansion for small Fermi energies, x_F << 1."""
    return eps_g * (x_F**3/3 + x_F**5/10 - x_F**7/56)


def pressure_free_Fermi_gas_UR(x_F, eps_g):
    """Expansion for large Fermi energies, x_F >> 1."""
    return eps_g * x_F**4 / 12


def pressure_free_Fermi_gas_UR_improved(x_F, eps_g):
    """Expansion for large Fermi energies, x_F >> 1."""
    return eps_g * (x_F**2 * (x_F ** 2 - 1)/12 + np.log(x_F)/8)


def energy_density_free_Fermi_gas_UR(x_F, eps_g):
    """Expansion for large Fermi energies, x_F >> 1."""
    return eps_g * x_F**4 / 4


def energy_density_free_Fermi_gas_UR_improved(x_F, eps_g):
    """Expansion for large Fermi energies, x_F >> 1."""
    return eps_g * (x_F**2 * (x_F ** 2 + 1)/4 - np.log(x_F)/8)


def plot_p_and_eps(x_F_min, x_F_max, n, UR=False, NR=False, UR_improved=False, NR_improved=False, save_name=""):
    """This function allows us to plot p(x_F) and eps(x_F) as a """
    eps_g = 1   # Normalised pressures and energy densities
    x_F_axis = np.linspace(0, x_F_max, n)
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(7, 3.2))
    ax1.plot(x_F_axis, pressure_free_Fermi_gas_analytic(x_F_axis, eps_g), label=r"Analytic $\{}$".format("bar{p}"))
    max1 = [pressure_free_Fermi_gas_analytic(x_F_max, eps_g)]
    if UR:
        # Adding p(x_F) in the UR limit
        max1.append(pressure_free_Fermi_gas_UR(x_F_max, eps_g))
        ax1.plot(x_F_axis, pressure_free_Fermi_gas_UR(x_F_axis, eps_g), label=r"UR limit $\{}$".format("bar{p}"))
    if UR_improved:
        # Adding p(x_F)-expansion for large x_F
        max1.append(pressure_free_Fermi_gas_UR_improved(x_F_max, eps_g))
        ax1.plot(x_F_axis, pressure_free_Fermi_gas_UR_improved(x_F_axis, eps_g),
                 label=r"Large $x_F$-expansion of $\{}$".format("bar{p}"))
    if NR:
        # Adding p(x_F) in the NR limit
        max1.append(pressure_free_Fermi_gas_NR(x_F_max, eps_g))
        ax1.plot(x_F_axis, pressure_free_Fermi_gas_NR(x_F_axis, eps_g), label=r"NR limit $\{}$".format("bar{p}"))
    if NR_improved:
        # Adding p(x_F)-expansion for small x_F
        max1.append(max(pressure_free_Fermi_gas_NR_improved(x_F_axis, eps_g)))
        ax1.plot(x_F_axis, pressure_free_Fermi_gas_NR_improved(x_F_axis, eps_g),
                 label=r"Small $x_F$-expansion of $\{}$".format("bar{p}"))
    ax1.set_xlim([x_F_min, x_F_max])
    ax1.set_ylim([0, 1.05 * max(max1)])
    ax1.set_title(r"(b) $\quad \{}(x_F)$".format("bar{p}"), fontdict={"fontsize": 10})
    ax1.set_xlabel(r"$x_F$")
    ax1.set_ylabel(r"$p \, / \, \varepsilon_g$", rotation=90)
    ax1.grid()
    ax1.legend(loc="upper left", handlelength=1.5, fontsize="small")

    max2 = [energy_density_free_Fermi_gas_analytic(x_F_max, eps_g)]
    ax2.plot(x_F_axis, energy_density_free_Fermi_gas_analytic(x_F_axis, eps_g),
             label=r"Analytic $\{}$".format("bar{\epsilon}"))
    if UR:
        # Adding eps(x_F) in the UR limit
        max2.append(energy_density_free_Fermi_gas_UR(x_F_max, eps_g))
        ax2.plot(x_F_axis, energy_density_free_Fermi_gas_UR(x_F_axis, eps_g),
                 label=r"UR limit $\{}$".format("bar{\epsilon}"))
    if UR_improved:
        # Adding eps(x_F)-expansion for large x_F
        max2.append(energy_density_free_Fermi_gas_UR_improved(x_F_max, eps_g))
        ax2.plot(x_F_axis, energy_density_free_Fermi_gas_UR_improved(x_F_axis, eps_g),
                 label=r"Large $x_F$-expansion of $\{}$".format("bar{\epsilon}"))
    if NR:
        # Adding eps(x_F) in the NR limit
        max2.append(energy_density_free_Fermi_gas_NR(x_F_max, eps_g))
        ax2.plot(x_F_axis, energy_density_free_Fermi_gas_NR(x_F_axis, eps_g),
                 label=r"NR limit $\{}$".format("bar{\epsilon}"))
    if NR_improved:
        # Adding eps(x_F)-expansion for small x_F
        max2.append(max(energy_density_free_Fermi_gas_NR_improved(x_F_axis, eps_g)))
        ax2.plot(x_F_axis, energy_density_free_Fermi_gas_NR_improved(x_F_axis, eps_g),
                 label=r"Small $x_F$-expansion of $\{}$".format("bar{\epsilon}"))
    ax2.set_xlim([x_F_min, x_F_max])
    ax2.set_ylim([0, 1.05 * max(max2)])
    ax2.set_title(r"(a) $\quad \{}(x_F)$".format("bar{\epsilon}"), fontdict={"fontsize": 10})
    ax2.set_xlabel(r"$x_F$")
    ax2.set_ylabel(r"$\epsilon \, / \, \varepsilon_g$", rotation=90)
    ax2.legend(loc="upper left", handlelength=1.5, fontsize="small")
    ax2.grid()
    plt.tight_layout()

    if not save_name:
        plt.show()
    else:
        plt.savefig("{}.svg".format(save_name), format="svg", dpi=600)


"""Uncomment to plot eps_bar(x_F) and p_bar(x_F)"""
# plot_p_and_eps(0, 1, 400, NR=True, NR_improved=True, save_name="Small_x_F_eps_and_p")    # NR, NR-improved
# plot_p_and_eps(1, 3, 400, UR=True, UR_improved=True, save_name="Large_x_F_eps_and_p")    # UR, UR-improved


def bisection_root_finder(f, x_min, x_max, accuracy, *f_params):
    """Self-implemented bisection root finder. Useful to find eps(p) in the analytical case."""
    assert_string = "f(x_min)*f(x_max) < 0 was not fulfilled. " \
                    "f(x_min) = f({0}) = {1}, f(x_max) = f({2}) = {3}".format(x_min, f(x_min, *f_params), x_max,
                                                                              f(x_max, *f_params))
    assert f(x_min, *f_params) * f(x_max, *f_params) <= 0, assert_string
    x0, x1 = x_min, x_max
    while abs(x0 - x1) > accuracy:
        c = (x0 + x1) / 2
        if f(x0, *f_params) * f(c, *f_params) <= 0:
            x1 = c
        else:
            x0 = c
    return (x0 + x1)/2


def eps_of_p(ax, p_arr, plot_title, **eps_arr):
    """Adds the curves eps(p) as supplied in **eps_arr to ax. The keyword is passed to the legend."""
    max1 = 0
    min1 = 0
    for name, eps in eps_arr.items():
        max1 = max(max1, eps[-1])
        min1 = min(min1, eps[0])
        ax.plot(p_arr, eps, label=r"{}".format(name))
        ax.legend(fontsize="small", handlelength=1.5)
    ax.set_xlim(p_arr[0], p_arr[-1])
    ax.set_ylim(0, max1 * 1.05)
    ax.grid()
    ax.set_title(r"{}".format(plot_title), fontdict={"fontsize": 10})

    ax.set_xlabel(r"$\{}$".format("bar{p}"))
    ax.set_ylabel(r"$\{}$".format("bar{\epsilon}"))

    # Adding some more ticks if there are five or less
    locs, labels = plt.xticks()
    locs_len = len(locs)
    if locs_len <= 5:
        intermediate = [(locs[i] + locs[i+1])/2 for i in range(locs_len - 1)]
        new_locs = [element for pair in zip(locs[0:locs_len - 1], intermediate) for element in pair]
        new_locs.append(locs[-1])
        ax.set_xticks(new_locs)

    return ax


def EoS_bar_NR(p_bar):
    """Free Fermi gas equation of state in the NR limit."""
    return 15**(3/5) / 3 * p_bar**(3/5)


def EoS_bar_UR(p_bar):
    """Free Fermi gas equation of state in the UR limit."""
    return 3 * p_bar


def EoS_bar_large_p_bar(p_bar):
    """Free Fermi gass equation of state from the large x_F-expansion."""
    return 3 * p_bar + np.sqrt(3 * p_bar)


def EoS_bar_small_p_bar(p_bar):
    """Free Fermi gass equation of state from the small x_F-expansion."""
    return 15**(3/5) / 3 * p_bar**(3/5) + 18/5 * p_bar


def EoS_bar_analytic(p_arr_bar):
    x_F_arr = np.zeros_like(p_arr_bar)
    for n, pressure_bar in enumerate(p_arr_bar):
        x_F_arr[n] = bisection_root_finder(lambda x: pressure_free_Fermi_gas_analytic(x, 1) - pressure_bar, 0, 100,
                                           10 ** (-7))
    return energy_density_free_Fermi_gas_analytic(x_F_arr, 1)


"""Uncomment to plot EoS for UR, NR and analytical."""
"""
p_arr1 = np.linspace(0, 0.3, 200)
p_arr2 = np.linspace(1, 4, 200)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
ax1 = eps_of_p(ax1, p_arr1, "(a) $\quad \{}(\{})$".format("bar{\epsilon}", "bar{p}"),
               **{"Analytic $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_analytic(p_arr1),
                  "UR limit $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_UR(p_arr1),
                  "NR limit $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_NR(p_arr1)})
ax2 = eps_of_p(ax2, p_arr2, "(b) $\quad \{}(\{})$".format("bar{\epsilon}", "bar{p}"),
               **{"Analytic $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_analytic(p_arr2),
                  "UR limit $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_UR(p_arr2),
                  "NR limit $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_NR(p_arr2)})
fig.axes.append(ax1)
fig.axes.append(ax2)
plt.tight_layout()
# plt.savefig("analytical_UR_NR_EoS_plot.svg", format="svg", dpi=600)
plt.show()
"""


"""Uncomment to plot EoS for small p_bar, large p_bar and analytical."""
"""
p_arr1 = np.linspace(0, 0.3, 200)
p_arr2 = np.linspace(1, 4, 200)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
ax1 = eps_of_p(ax1, p_arr1, "(a) $\quad \{}(\{})$".format("bar{\epsilon}", "bar{p}"),
               **{"Analytic $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_analytic(p_arr1),
                  "Large $\{1}$-expansion $\{0}(\{1})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_large_p_bar(p_arr1),
                  "Small $\{1}$-expansion $\{0}(\{1})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_small_p_bar(p_arr1)})
ax2 = eps_of_p(ax2, p_arr2, "(b) $\quad \{}(\{})$".format("bar{\epsilon}", "bar{p}"),
               **{"Analytic $\{}(\{})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_analytic(p_arr2),
                  "Large $\{1}$-expansion $\{0}(\{1})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_large_p_bar(p_arr2),
                  "Small $\{1}$-expansion $\{0}(\{1})$".format("bar{\epsilon}", "bar{p}"): EoS_bar_small_p_bar(p_arr2)})
fig.axes.append(ax1)
fig.axes.append(ax2)
plt.tight_layout()
# plt.savefig("analytical_large_p_bar_small_p_bar_EoS_plot.svg", format="svg", dpi=600)
plt.show()
"""
