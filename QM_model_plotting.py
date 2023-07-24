from plot_import_and_params import *
from matplotlib.lines import Line2D
from QM_model import f_pi, omega_0_incr, omega_0_diff_incr, omega_0_cr, omega_0_diff_cr, find_mus, \
    separate_mfs_and_mus, pressure, energy_density, number_densites_quarks, number_densities_electrons, \
    maxwell_construct

"""The goal in this file is to create the plots related to the QM model.
This is heavily dependent on the functions in QM_model.py."""


def plot_incr_mesonic_potential():
    """Creates a plot of the grand potential in vacuum as a function of mf, <sigma>. In the thesis, this is the
    first plot in chapter 11.4."""
    fig, ax = plt.subplots(figsize=(6, 3))
    m_sigmas = [400, 500, 533, 600]
    mfs = np.linspace(-50, 150, 800)
    n_f_pi = int(round((93 - min(mfs)) / (max(mfs) - min(mfs)) * len(mfs)) + 0.1)   # index of mf = f_pi
    colornorm = colors.Normalize(vmin=min(m_sigmas), vmax=max(m_sigmas))
    cmap = cm.get_cmap("viridis")
    number_to_color = cm.ScalarMappable(norm=colornorm, cmap=cmap)
    for n, m_sigma in enumerate(m_sigmas):
        grand_pot = omega_0_incr(mfs / f_pi, m_sigma / f_pi)   # Calculation grand potential in vacuum, Omega_0.
        ax.scatter(mfs[n_f_pi], grand_pot[n_f_pi], color="black", s=10, zorder=2)
        ax.plot(mfs, grand_pot, label=r"$m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"),
                color=number_to_color.to_rgba(m_sigma), zorder=1)
        ax.set_xlim(min(mfs), max(mfs))
    ax.set_ylim(-5, 2)  # Fitting for our selection of m_sigmas
    ax.set_xlabel(r"$\langle \sigma \rangle \, / \, \unit{\mega \eV}$")
    ax.set_ylabel(r"$\Omega_0 \, /  \, f_\pi^4$")
    ax.grid()
    ax.legend(loc="lower left")
    plt.title(r"Grand potential $\Omega$ at $\mu_u = \mu_d = \mu_e = 0$")
    plt.tight_layout()
    # Uncomment to save figure
    # plt.savefig(format="eps", fname="mesonic_potential.eps", dpi=600)
    plt.show()


def plot_mesonic_potential_both():
    """Creates a plot of the grand potential in vacuum for both cr and incr. We find this as the first plot in
    chapter 11.5."""
    fig, ax = plt.subplots(figsize=(5.5, 3))
    mfs = np.linspace(-0.5, 1.5, 1000)
    n_vacuum_min = 750
    m_sigmas_proper = [400, 500, 600]
    m_sigmas_incons = [600, 700, 800]
    norm_proper, norm_incons = colors.Normalize(vmin=min(m_sigmas_proper), vmax=max(m_sigmas_proper)), \
                   colors.Normalize(vmin=min(m_sigmas_incons), vmax=max(m_sigmas_incons))
    cmap = cm.get_cmap("viridis")
    num_to_col_proper, num_to_col_incons = cm.ScalarMappable(norm_proper, cmap), cm.ScalarMappable(norm_incons, cmap)
    handles_proper, handles_incons = [], []
    proper_line, incons_line = [Line2D([0], [0], color=cm.coolwarm(0.5), label="Consistent"),
                                Line2D([0], [0], color=cm.coolwarm(0.5), linestyle="dashed", label="Inconsistent")]
    handles_proper.append(proper_line)
    handles_incons.append(incons_line)
    for m_sigma_prop, m_sigma_incons in zip(m_sigmas_proper, m_sigmas_incons):
        omega_0_prop = omega_0_cr(mfs, m_sigma_prop / f_pi)
        omega_0_incons = omega_0_incr(mfs, m_sigma_incons / f_pi)
        line_proper, = ax.plot(mfs, omega_0_prop,
                               label=r"$m_\sigma = {} \, \unit{}$".format(round(m_sigma_prop), "{\mega \eV}"),
                               color=num_to_col_proper.to_rgba(m_sigma_prop))
        line_incons, = ax.plot(mfs, omega_0_incons,
                               label=r"$m_\sigma = {} \, \unit{}$".format(m_sigma_incons, "{\mega \eV}"),
                               color=num_to_col_incons.to_rgba(m_sigma_incons), linestyle="dashed")
        handles_proper.append(line_proper)
        handles_incons.append(line_incons)
        ax.scatter([n_vacuum_min], omega_0_prop[n_vacuum_min], s=10, zorder=2, color="black")
        ax.scatter([n_vacuum_min], omega_0_incons[n_vacuum_min], s=10, zorder=2, color="black")

    legend_proper = ax.legend(handles=handles_proper, loc="upper right", fontsize="small", handlelength=1.7)
    plt.gca().add_artist(legend_proper)  # Allows for a second legend
    ax.legend(handles=handles_incons, loc="lower left", fontsize="small", handlelength=1.7)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-9, 1)
    ax.set_xticks([-0.375, -0.125, 0.125, 0.375, 0.625, 0.875, 1.125, 1.375], minor=True)
    ax.set_yticks([-9, -7, -5, -3, -1, 1], minor=True)

    ax.set_xlabel(r"$\langle \sigma \rangle \, / \, f_\pi$")
    ax.set_ylabel(r"$\Omega_0 \, / \, f_\pi$")
    ax.set_title(r"$\Omega$ at $\mu_u = \mu_d = \mu_e = 0$, consistent and inconsistent renormalisation", fontdict={"fontsize": 10})

    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.2)
    plt.tight_layout(pad=1.05)
    # Uncomment to save figure
    # plt.savefig(fname="Vacuum_potential_comparison.eps", format="eps", dpi=600)
    plt.show()


def plot_mu_solution():
    """Having found mu_u and mu_d for each mf, we can plot them. We also plot mu_u and mu_d
    as a function of their average, mu = 1/2(mu_u + mu_d) In addition, we can plot the number densities for the
    quarks and electron as a function of mu."""
    mfs = np.linspace(0.001, 0.999, 500)
    m_sigmas = [600, 700, 800]
    cmap = cm.get_cmap("viridis")
    cmap2 = cm.get_cmap("plasma")
    color_norm = colors.Normalize(vmin=min(m_sigmas), vmax=max(m_sigmas))
    num_to_col = cm.ScalarMappable(norm=color_norm, cmap=cmap)
    num_to_col2 = cm.ScalarMappable(norm=color_norm, cmap=cmap2)
    fig = plt.figure(figsize=(6.5, 5), layout="constrained", )
    subfig1, subfig2 = fig.subfigures(nrows=2, height_ratios=[1.1, 1])
    subfig1.suptitle("Quark chemical potentials", fontsize=10)
    subfignest = subfig1.subfigures()
    subfig1.suptitle(" ")
    subfignest.suptitle(" ")
    ax, ax2 = subfignest.subplots(ncols=2)
    ax.set_title(r"Quark chemical potentials $\mu_q$", fontdict={"fontsize": 10})
    ax3 = subfig2.subplots()
    ax3.set_title(r"Quark number densities $n_q$", fontdict={"fontsize": 10})
    ax3.axes.set_aspect(aspect=1)
    handles_mu_u, handles_mu_d = [], []
    handles_n_u, handles_n_d = [], []

    # We must convert from natural units to ordinary units. We have mass in terms of energy.
    # Then we must regain ordinary number density 1/V. We do this by multiplying by 1/(hbar^3 c^3)
    # 1 unit of n / f_\pi^3 -> 1.047 * 10 ** 44 m^-3.
    # Finally, we scale to sensible units
    # 1 unit of n / f_\pi^3 -> 0.1047 / fm^3
    conversion_factor = 0.1047

    for m_sigma in m_sigmas:
        mfs_and_mus = np.flip(find_mus(mfs, m_sigma / f_pi, omega_0_diff_incr), axis=1)
        vnm, _ = separate_mfs_and_mus(mfs_and_mus)
        line_mu_u, = ax.plot(vnm[0, :], vnm[1, :], color=num_to_col.to_rgba(m_sigma), label=r"$\mu_u, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        line_mu_d, = ax.plot(vnm[0, :], vnm[2, :], color=num_to_col.to_rgba(m_sigma),
                label=r"$\mu_d, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"), linestyle=(0, (3, 3)))
        handles_mu_u.append(line_mu_u)
        handles_mu_d.append(line_mu_d)
        mus = (vnm[1, :] + vnm[2, :]) / 2
        mu_es = vnm[2, :] - vnm[1, :]
        n_us, n_ds = number_densites_quarks(vnm)
        n_es = number_densities_electrons(vnm)
        ax2.plot(mus, vnm[1, :], color=num_to_col.to_rgba(m_sigma))
        ax2.plot(mus, vnm[2, :], color=num_to_col.to_rgba(m_sigma), linestyle=(0, (3, 3)))
        ax2.plot((0, mus[-1]), (0, mus[-1]), color="black")
        line_mu_e, = ax2.plot(mus, mu_es, color=num_to_col.to_rgba(m_sigma), linestyle=(0, (1, 1)),
                              label=r"$\mu_e, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega\eV}"))
        print("The slope of mu_d(mu_u) for large mu: {}".format(np.gradient(vnm[2, -20:], vnm[1, -20:])[-2]))

        line_u, = ax3.plot(mus, n_us * conversion_factor, color=num_to_col2.to_rgba(m_sigma),
                           label=r"$n_u, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega\eV}"))
        line_d, = ax3.plot(mus, n_ds * conversion_factor, color=num_to_col2.to_rgba(m_sigma), linestyle=(0, (3, 3)),
                           label=r"$n_d, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega\eV}"))
        handles_n_u.append(line_u)
        handles_n_d.append(line_d)
        ax3.plot(mus, n_es * conversion_factor, color=num_to_col2.to_rgba(m_sigma), linestyle=(0, (1, 1)),
                 label=r"$n_e, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega\eV}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(2, 6.5)
    ax.set_xlabel(r"$\langle \sigma \rangle \, / \, f_\pi$")
    ax.set_ylabel(r"$\mu_q \,  / \,  f_\pi$")

    ax2.set_xlabel(r"$\mu \, / \, f_\pi$")
    ax2.set_ylabel(r"$\mu_q \, / \, f_\pi$")
    ax2.set_ylim(2, 6.5)
    ax2.set_xlim(2, 6.5)

    ax3.set_xlabel(r"$\mu \, / \, f_\pi$")
    ax3.set_ylabel(r"$n_i \, / \, \unit{\per\cubic\femto\metre}$")
    ax3.set_xlim(2.5, 5)
    ax3.set_ylim(0, 1.5)

    ax.grid()
    ax2.grid()
    ax3.grid()

    subfignest.legend(handles=handles_mu_u + handles_mu_d, fontsize="small",
                      bbox_to_anchor=(0, 1.075, 1.0, 0.01), ncol=2,
                      handlelength=1.5, fancybox=True)

    ax3.legend(handles=handles_n_u + handles_n_d, loc="upper left", fontsize="small", bbox_to_anchor=(1.05, 1),
               handlelength=1.5)
    # Uncomment to save figure
    # plt.savefig(fname="mu_of_vev_solution.eps", format="eps", dpi=600)
    plt.show()


def plot_pressure_eps_of_mu(m_sigmas):
    """Plotting incr pressure and energy density as a function of mu = 1/2(mu_u + mu_d). This is the third figure in
    chapter 11.4 in the thesis with m_sigmas = [600, 800]."""
    fig, (ax, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(8, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.05, 0.05]})
    epsilon = 0.01
    lin_space = np.linspace(0, 1, 400, endpoint=True)
    mfs = lin_space ** 3 * (1 - 2 * epsilon) + epsilon
    cmap = cm.get_cmap("viridis")
    cmap2 = cm.get_cmap("plasma")
    norm = colors.Normalize(vmin=0, vmax=1)
    num_to_col = cm.ScalarMappable(norm=norm, cmap=cmap)
    num_to_col2 = cm.ScalarMappable(norm=norm, cmap=cmap2)
    num_to_col_list = [num_to_col, num_to_col2]
    min_mu, max_mu = 3, 0
    min_press, min_eps = 0, 0
    for m, m_sigma in enumerate(m_sigmas):
        mfs_and_mus = find_mus(mfs, m_sigma / f_pi, omega_0_diff_incr)
        mfs_and_mus_sep, _ = separate_mfs_and_mus(mfs_and_mus)
        min_mu, max_mu = min(min_mu, min((mfs_and_mus_sep[1, :] + mfs_and_mus_sep[2, :])/2)), \
                         max(max_mu, max((mfs_and_mus_sep[1, :] + mfs_and_mus_sep[2, :])/2))
        pressures = pressure(mfs_and_mus_sep, m_sigma / f_pi, omega_0_incr)
        epss = energy_density(mfs_and_mus_sep, m_sigma / f_pi, omega_0_incr)
        min_press, min_eps = min(min_press, min(pressures)), min(min_eps, min(epss))
        # Favouring (mu_u + mu_d)/2 over using either mu_u or mu_d.
        # Coloring after mf-value
        # Forcefully adding when <vev> = 1, as the numerical solution does not work for that value
        ax.plot((300 / 93, (mfs_and_mus_sep[1, -1] + mfs_and_mus_sep[2, -1]) / 2), (0, pressures[-1]),
                color=num_to_col_list[m].to_rgba(mfs_and_mus_sep[0, -1]))
        ax2.plot((300 / 93, (mfs_and_mus_sep[1, -1] + mfs_and_mus_sep[2, -1]) / 2), (0, epss[-1]),
                 color=num_to_col_list[m].to_rgba(mfs_and_mus_sep[0, -1]))
        for n in range(1, len(mfs_and_mus_sep[0, :])):
            ax.plot((mfs_and_mus_sep[1, n-1:n+1] + mfs_and_mus_sep[2, n-1:n+1])/2, pressures[n-1:n+1],
                    color=num_to_col_list[m].to_rgba(mfs_and_mus_sep[0, n-1]))
            ax2.plot((mfs_and_mus_sep[1, n-1:n+1] + mfs_and_mus_sep[2, n-1:n+1])/2, epss[n-1:n+1],
                     color=num_to_col_list[m].to_rgba(mfs_and_mus_sep[0, n-1]))
    forced_x_max = 6
    forced_y_max_p, forced_y_max_eps = 40, 80
    ax.set_xlabel(r"$\mu \, / \, f_\pi$")
    ax.set_ylabel(r"$p \, / \, f_\pi^4$", labelpad=2.5)
    ax.set_xlim(2.5, min(forced_x_max, max_mu))
    ax.set_ylim(-0.5, forced_y_max_p)
    ax.margins(tight=True)
    ax.set_xticks([3, 4, 5, 6])
    ax.set_xticks([2.5, 3.5, 4.5, 5.5], minor=True)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticks([5, 15, 25, 35], minor=True)
    # ax.set_yticklabels([0, 10, 20, 30, 40], minor=False)
    m_sigma_string = "\{" + str(m_sigmas[0]) + ",  \, " + str(m_sigmas[1]) + "\}"
    ax.set_title(r"$p(\mu), \quad m_\sigma \in {} \, \unit{}$".format(m_sigma_string, "{\mega \eV}"),
                 fontdict={"fontsize": 10})
    ax2.set_xlabel(r"$\mu \, / \, f_\pi$")
    ax2.set_ylabel(r"$\epsilon \, / \, f_\pi^4 $", labelpad=2.5)
    ax2.set_xlim(2.5, min(forced_x_max, max_mu))
    ax2.set_ylim(min_eps, forced_y_max_eps)
    ax2.set_xticks([3, 4, 5, 6])
    ax2.set_xticks([2.5, 3.5, 4.5, 5.5], minor=True)
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.set_yticks([10, 30, 50, 70], minor=True)
    ax2.set_title(r"$\epsilon(\mu), \quad m_\sigma \in {} \, \unit{}$".format(m_sigma_string, "{\mega \eV}"),
                  fontdict={"fontsize": 10})

    cbar = fig.colorbar(num_to_col, cax=ax3, orientation="vertical", pad=0.01, aspect=10)
    cbar2 = fig.colorbar(num_to_col2, cax=ax4, orientation="vertical", aspect=10)
    cbar2.ax.set_ylabel(r"$\langle \sigma \rangle \, / \, f_\pi$", rotation=90)
    cbar.ax.set_title(r"$m_\sigma = {}\,  \unit{}$".format(m_sigmas[0], "{\mega \eV}"), fontdict={"fontsize": 10})
    cbar2.ax.set_xlabel(r"$m_\sigma = {}\,  \unit{}$".format(m_sigmas[1], "{\mega \eV}"), labelpad=9,
                        fontdict={"fontsize": 10})
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.2)
    ax2.grid(which="major", alpha=0.5)
    ax2.grid(which="minor", alpha=0.2)
    ax3.margins(0)

    plt.tight_layout(pad=0.5, w_pad=0.25)
    # Uncomment to save figure
    # plt.savefig(fname="pressure_and_energy_density_of_mu_incons_renorm.svg", format="svg", dpi=600)
    plt.show()


def plot_EoS(m_sigmas, omega_0, omega_0_diff):
    """Plotting the equation of state for incr or cr. Creates the fourth figure in chapter 11.4 and the second figure
    in 11.5."""
    fig, ax = plt.subplots(ncols=1, figsize=(5.5, 3.2))

    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=min(m_sigmas), vmax=max(m_sigmas))
    num_to_col = cm.ScalarMappable(norm=norm, cmap=cmap)

    # We must convert from natural units to ordinary units. We have mass in terms of energy.
    # First, we take this into account m^4 -> (m/c^2)^4 : multiply with factor 1/c^8.
    # Then we must regain ordinary pressure N/m^2. We do this my multiplying the mass with c^5 / hbar^3.
    # 1 unit of p / f_\pi^4 -> 1.56 * 10^33 pascal.
    # Finally, we scale to sensible units: GeV / fm^3 (multiply by 1/ (1.602 * 10^35)
    # 1 unit of p / f_\pi^4 -> 0.00974 GeV / fm^3
    conversion_factor = 0.00974
    mfs = np.linspace(0.03, 0.999, 2000)
    handles1, handles2 = [], []
    for m_sigma in m_sigmas:
        print("m_sigma: {} MeV.".format(m_sigma))
        mfs_and_mus = np.flip(find_mus(mfs, m_sigma / f_pi, omega_0_diff), axis=1)
        mf_mu_sep, _ = separate_mfs_and_mus(mfs_and_mus)    # short for mfs_and_mus_separated
        pressures = pressure(mf_mu_sep, m_sigma / f_pi, omega_0) * conversion_factor
        epss = energy_density(mf_mu_sep, m_sigma / f_pi, omega_0) * conversion_factor
        press_after_construct, eps_after_construct, indices = maxwell_construct(pressures, epss, return_indices=True)
        if len(indices) == 2:
            line2, = ax.plot(pressures[indices[0]:indices[1]], epss[indices[0]:indices[1]], linestyle="dotted",
                             color=num_to_col.to_rgba(m_sigma),
                             label=r"$m_\sigma = {} \, \unit{}$, no construction".format(m_sigma, "{\mega \eV}"))
            handles2.append(line2)
        elif len(indices) == 1:
            line2, = ax.plot(pressures[0:indices[0]], epss[0:indices[0]], linestyle="dotted",
                             color=num_to_col.to_rgba(m_sigma),
                             label=r"$m_\sigma = {} \, \unit{}$, no construction".format(m_sigma, "{\mega \eV}"))
            handles2.append(line2)
        construction_text = "after construction"
        if not indices:
            construction_text = "no construction"
        line1, = ax.plot(press_after_construct, eps_after_construct, color=num_to_col.to_rgba(m_sigma),
                         label=r"$m_\sigma = {} \, \unit{}$, {}".format(m_sigma, "{\mega \eV}", construction_text))
        handles1.append(line1)
    ax.set_xlim(-0.005, 0.04)
    ax.set_ylim(0, 0.45)
    ax.set_xticks([0.00, 0.01, 0.02, 0.03, 0.04])
    ax.set_xticks([-0.005, 0.005, 0.015, 0.025, 0.035], minor=True)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0.05, 0.15, 0.25, 0.35, 0.45], minor=True)
    femto_txt = "\{}".format("femto")
    ax.set_xlabel(r"$p \, / \, \unit{}$".format("{" + "\giga \eV \per{}\cubic\metre".format(femto_txt) + "}"))
    ax.set_ylabel(r"$\epsilon \, / \, \unit{}$".format("{" + "\giga \eV \per{}\cubic\metre".format(femto_txt) + "}"))
    ax.grid(which="minor", alpha=0.2)
    ax.grid(which="major", alpha=0.5)
    ax.legend()
    ax.set_title(r"$\text{QM model quation of state, consistent renormalising}$", fontdict={"fontsize": 10})

    first_legend = ax.legend(handles=handles1, loc="lower right", fontsize="small", handlelength=1.5)
    plt.gca().add_artist(first_legend)  # Allows for a second legend
    ax.legend(handles=handles2, loc="upper left", fontsize="small", handlelength=1.5)

    plt.tight_layout()
    # Uncomment to save figure
    # plt.savefig(fname="EoS_QM_consistent_renorm.svg", format="svg", dpi=600)
    plt.show()


"""Run to generate EoS plot for inconsistently renormalised QM-matter."""
# plot_EoS([600, 700, 800], omega_0_incr, omega_0_diff_incr)
# plot_EoS([400, 500, 550, 600], omega_0_cr, omega_0_diff_cr)


def find_bag_window_plot(mfs, m_sigma, omega_0, omega_0_diff):
    """Creating illustration of how we find the minimum bag constant for the incr case.
    Creates the first figure in chapter 11.4.1."""
    # Must find the bag window according to (eps_u + eps_d) / n_b > 931 MeV.
    energy_per_nucleon_bar = 931 / f_pi
    mfs_and_mus = np.flip(find_mus(mfs, m_sigma / f_pi, omega_0_diff), axis=1)
    mfs_and_mus_sep, _ = separate_mfs_and_mus(mfs_and_mus)
    # Need to find mu such that p(mu) = 0
    press_arr = pressure(mfs_and_mus_sep, m_sigma / f_pi, omega_0)
    eps_arr = energy_density(mfs_and_mus_sep, m_sigma / f_pi, omega_0)
    n_u, n_d = number_densites_quarks(mfs_and_mus_sep)
    press_after_construct, eps_after_construct, indices = maxwell_construct(press_arr, eps_arr, return_indices=True)
    # Now we need to slice n_u and n_d correspondingly to how we sliced press_arr and eps_arr in
    # the maxwell construction. No construction: empty list. non-zero energy density at p = 0: one element in the list
    # Proper construction: two indices in index-list.
    if len(indices) == 0:
        print("No index")
        n_u_after_construct, n_d_after_construct = n_u, n_d
    elif len(indices) == 1:
        print("One index")
        n_u_after_construct, n_d_after_construct = n_u[indices[0]:], n_d[indices[0]:]
        assert len(n_u_after_construct) == len(press_after_construct), "Length of n_u: {}, length of ps: {}".format(len(n_u_after_construct), len(press_after_construct))
    else:
        print("Two indices")
        n_u_after_construct, n_d_after_construct = np.concatenate((n_u[0:indices[0] + 1], n_u[indices[1]:])), \
                                                   np.concatenate((n_d[0:indices[0] + 1], n_d[indices[1]:]))
        assert len(n_u_after_construct) == len(press_after_construct), "Length of n_u: {}, length of ps: {}".format(len(n_u_after_construct), len(press_after_construct))
        # indices[...] + 1 to INCLUDE the index in the list
    n_B = (n_u_after_construct + n_d_after_construct) / 3   # baryon number for one quark is 1 / 3.
    # 1 is suitable for m_sigma = 800. 4 is suitable for m_sigma = 600.
    bag_max = 1     # Suitable range for bag-shifts
    n_bag = 20     # Suitable number of curves to be plotted
    bag_consts = np.linspace(0.0, bag_max, n_bag, endpoint=True)

    gridspec_kwargs = dict(width_ratios=[1, 0.02], height_ratios=[1, 1, 1])
    fig, ax_dict = plt.subplot_mosaic([["1", "0"],
                                       ["2", "0"],
                                       ["3", "0"]], figsize=(5, 6), layout="constrained", gridspec_kw=gridspec_kwargs)
    ax_cbar, ax1, ax2, ax3 = ax_dict["0"], ax_dict["1"], ax_dict["2"], ax_dict["3"]
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=0, vmax=max(bag_consts))
    num_to_col = cm.ScalarMappable(norm, cmap)

    def plot_EoS_with_shift(ax, bag):
        ax.plot(press_after_construct - bag, eps_after_construct + bag, color=num_to_col.to_rgba(bag))
        return ax

    def plot_number_density_with_shift(ax, bag):
        ax.plot(press_after_construct - bag, n_B, color=num_to_col.to_rgba(bag))

    def energy_per_baryon(bag_const):
        shifted_press = press_after_construct - bag_const
        zero_press_index = np.argmin(np.abs(shifted_press))
        shifted_eps = eps_after_construct + bag_const
        # ax1.scatter(shifted_press[zero_press_index], shifted_eps[zero_press_index], s=10, color="black", zorder=3)
        # ax2.scatter(shifted_press[zero_press_index], n_B[zero_press_index], s=10, color="black", zorder=3)
        return shifted_eps[zero_press_index] / n_B[zero_press_index], 1

    for bag in bag_consts:
        plot_EoS_with_shift(ax1, bag)
        plot_number_density_with_shift(ax2, bag)
    ax1.set_xlim(-bag_max, 1)
    ax1.set_ylim(0, 35)
    ax1.plot((0, 0), (0, 35), color="black", linewidth=1)
    ax1.set_xlabel(r"$(p - B) \, / \,  f_\pi^4$")
    ax1.set_ylabel(r"$(\epsilon + B) \, / \, f_\pi^4$")
    ax1.grid()
    ax1.set_title(r"EoS shifted by bag constant $B$", fontdict={"fontsize": 10})

    ax2.set_xlim(-bag_max, 1)
    ax2.set_ylim(0, 4.5)
    ax2.set_xlabel(r"$(p - B) \, / \, f_\pi^4 $")
    ax2.set_ylabel(r"$n_B \, / \, f_\pi^3$")
    ax2.grid()
    ax2.set_title(r"Baryonic number density $n_B$", fontdict={"fontsize": 10})

    coord_list = []
    ax3.scatter(bag_consts[0], energy_per_baryon(bag_consts[0])[0] - energy_per_nucleon_bar,
                color=num_to_col.to_rgba(bag_consts[0]))
    last_coord = (bag_consts[0], energy_per_baryon(bag_consts[0])[0] - energy_per_nucleon_bar)
    coord_list.append(last_coord)
    for bag in bag_consts[1:]:
        next_coord = (bag, energy_per_baryon(bag)[0] - energy_per_nucleon_bar)
        ax3.scatter(next_coord[0], next_coord[1], color=num_to_col.to_rgba(bag), zorder=2)
        ax3.plot((last_coord[0], next_coord[0]), (last_coord[1], next_coord[1]), color="black", linewidth=1, zorder=1)
        last_coord = (bag, energy_per_baryon(bag)[0] - energy_per_nucleon_bar)
        coord_list.append(last_coord)
    ax3.set_xlim(0, bag_max)
    # ax3.set_ylim(-10, 40)
    ax3.set_ylim(min([coord[1] for coord in coord_list]), max([coord[1] for coord in coord_list]))
    ax3.set_xlabel(r"$B \, / \, f_\pi^4$")
    ax3.set_ylabel(r"$\frac{\epsilon(p=0)}{n_B(p=0)} - e_{nuc} \, / \, f_\pi$")
    ax3.set_title(r"Energy per baryon shifted by $e_{nuc}$", fontdict={"fontsize": 10})
    ax3.grid(which="major", alpha=0.5)
    ax3.grid(which="minor", alpha=0.2)
    # For generating plot in thesis:
    # ax3.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax3.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    # ax3.set_yticks([0, 20, 40])            # NB: This was WRONG. The plotting is ok, however, this manual setting
    # ax3.set_yticks([-10, 10, 30], minor=True) # of the y-axis was faulty.
    cbar = fig.colorbar(num_to_col, cax=ax_cbar, orientation="vertical", pad=0.01, aspect=10)
    cbar.ax.set_ylabel(r"$B / f_\pi^4$", rotation=90)
    ax_cbar.yaxis.set_label_position("right")
    ax_cbar.yaxis.tick_right()
    ax_cbar.margins(0)
    # plt.tight_layout(pad=2, w_pad=2)
    # Uncomment to save figure
    # plt.savefig(format="eps", fname="bag_pressure_determination_600_linear_vev.eps", dpi=600)
    plt.show()


"""Uncomment to generate bag constant plot"""
# mfs, m_sigma = np.linspace(0.001, 0.9999, 3000), 800
# find_bag_window_plot(mfs, m_sigma, omega_0_incr, omega_0_diff_incr)


