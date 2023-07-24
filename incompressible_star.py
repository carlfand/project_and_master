from plot_import_and_params import *


"""In this file, we plot the pressure profiles plot for the incompressible star.
The plot is found in chapter 3.3 in the thesis."""


def incomp_TOV_normalised_pressure(x, gamma):
    """Gamma = 2GM/Rc^2,
        x = r / R.
    Returns the solution of the TOV-equation for an incompressible star."""
    return (1 - 3 * np.sqrt(1 - gamma))/(np.sqrt(1 - gamma) - 1) * (np.sqrt(1 - gamma) - np.sqrt(1 - gamma * x**2)) / \
           (np.sqrt(1 - gamma * x**2) - 3 * np.sqrt(1 - gamma))


def incomp_Newton_normalised_pressure(x, gamma):
    """Gamma = 2GM/Rc^2,
        x = r / R.
    Returns the solution to the Newton hydrostatic equation."""
    return 1 - x**2


def incomp_TOV_pressure(x, R, M):
    """Returns p/(rho_0 * G)"""
    return (np.sqrt(1 - 2*M/R) - np.sqrt(1 - 2 * M / R * x**2)) / \
           (np.sqrt(1 - 2 * M/R * x**2) - 3 * np.sqrt(1 - 2*M/R))


def incomp_Newton_pressure(x, R, M):
    """Returns p/(rho_0 * G)"""
    return M/(2*R) * (1 - x**2)


def incomp_plot():
    R = 9
    numerator = [3, 35, 39, 399]
    denominator = [9, 90, 90, 900]
    N = 100
    x = np.linspace(0, 1, N)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    plt.title(r"Normalised pressure profiles, incompressible star", fontdict={"fontsize": 10})
    # Viridis coloring
    colornorm = colors.Normalize(vmin=((numerator[0] - 1) / denominator[0]) ** 2,
                                 vmax=(numerator[-1] / denominator[-1]) ** 2)
    colormap = plt.get_cmap("viridis")
    number_to_color = cm.ScalarMappable(norm=colornorm, cmap=colormap)
    for num, denom in zip(numerator, denominator):
        color_name = number_to_color.to_rgba((num / denom) ** 2)
        ax.plot(x, incomp_TOV_normalised_pressure(x, 2 * num / denom), color=color_name,
                label=r"TOV, $\{} = \{}$".format("frac{MG}{c^2R}",
                                                 "frac{}{}".format("{" + str(num) + "}", "{" + str(denom) + "}")))

    ax.plot(x, incomp_Newton_normalised_pressure(x, R), label=r"Newton", color="black")

    # Some plot formatting, for esthetics.
    x_ticks_large = np.arange(0, 1.2, 0.2)
    x_ticks_small = np.arange(0, 1.2, 0.1)

    ax.set_xticks(x_ticks_large)
    ax.set_xticks(x_ticks_small, minor=True)

    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.1), minor=True)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$r \, /  \, R$")
    ax.set_ylabel(r"$p \, / \, p_c$", rotation=90)
    ax.legend(fontsize="small", handlelength=1.5)
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.2)
    plt.tight_layout()
    # Uncomment to save figure:
    # plt.savefig("Normalised_pressure_profiles_incomp_star.svg", format="svg", dpi=600)
    plt.show()

"""Uncomment to generate plot and print the central pressure ratios for different values of gamma."""
"""
print("p_c_tov/p_c_newton, gamma = 1/9: {},\np_c_tov/p_c_newton,gamma = 2/9: {},\np_c_tov/p_c_newton, gamma = 3/9: {},\n"
      "p_c_tov/p_c_newton, gamma = 3.5/9: {},\np_c_tov/p_c_newton, gamma = 3.99/9: {}.".format(
    incomp_TOV_pressure(0, 9, 1)/incomp_Newton_pressure(0, 9, 1),
    incomp_TOV_pressure(0, 9, 2)/incomp_Newton_pressure(0, 9, 2),
    incomp_TOV_pressure(0, 9, 3)/incomp_Newton_pressure(0, 9, 3),
    incomp_TOV_pressure(0, 9, 3.5)/incomp_Newton_pressure(0, 9, 3.5),
    incomp_TOV_pressure(0, 9, 3.99)/incomp_Newton_pressure(0, 9, 3.99)))

incomp_plot()
"""
