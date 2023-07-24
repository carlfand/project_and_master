from plot_import_and_params import *
from QM_model import get_system_quantities_standard_cr
from hybrid_stars import EoS_hybrid_standard, EoS_hybrid, EoS_unified_standard_3_3, find_critical_mu_B, \
    find_best_polynomial_params, unific_3_3_free_midpoint, causality_condition
from APR_equation_of_state import nuclear_matter_shifted_neutron_mass, nuclear_matter_mu_B_n_B_p_eps, EoS_pure_APR

"""In this file, we handle the plotting of the APR, hybrid, and unified model."""


def nuclear_matter_plotting():
    """Generates plots which describes the properties of nuclear matter found in chapter 12.1.
    In this plot, we use m_n = 939.6 MeV."""
    # Getting data from APR EoS
    # mu_B, n_B, ps, epss = nuclear_matter_mu_B_n_B_p_eps()
    mu_B, n_B, ps, epss = nuclear_matter_shifted_neutron_mass()
    epss_diff_ps = np.gradient(epss, ps)
    n_gradient_one = np.where(np.diff(np.sign(epss_diff_ps - 1)))[0]    # seeking where d eps / dp < 1
    # As the growth is decreasing, the first element should be where the speed of sound goes above zero.
    crit_p = ps[n_gradient_one] + (epss_diff_ps[n_gradient_one] - 1) / (epss_diff_ps[n_gradient_one] - epss_diff_ps[n_gradient_one + 1]) * (ps[n_gradient_one + 1] - ps[n_gradient_one])
    crit_eps = np.interp(crit_p, ps, epss)
    crit_mu = np.interp(crit_p, ps, mu_B)
    crit_n = np.interp(crit_p, ps, n_B)
    print("APR matter has d eps / dp < 1 (non causality) at p = {} [Pa],"
          " eps = {} [GeV / fm^3], mu_B {} [MeV] and n {} [fm^(-3)]".format(crit_p * 1.56 * 10**33, crit_eps * 0.00974,
                                                                            crit_mu * 93, crit_n * 0.1047))
    fig = plt.figure(figsize=(5.5, 4.2), layout="constrained")
    # A lot of figure formatting follows.
    fig.suptitle(r"APR model of nuclear matter", fontsize=10)
    subfig1, subfig2 = fig.subfigures(2, 1, height_ratios=[1, 1])
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=0, vmax=3)
    num_to_col = cm.ScalarMappable(norm, cmap)
    ax1, ax2 = subfig1.subplots(ncols=2, nrows=1)
    ax3, ax4 = subfig2.subplots(ncols=2, nrows=1)
    ax1.plot(mu_B, n_B, label=r"$n_B(\mu_B)$", color=num_to_col.to_rgba(0))
    ax1.scatter(crit_mu, crit_n, color="black", s=15, marker="x", zorder=2)
    ax1.set_xlabel(r"$\mu_B \, / \, f_\pi$")
    ax1.set_ylabel(r"$n_B \, / \, f_\pi^3$")
    ax1.set_xticks([12.5, 17.5, 22.5, 27.5], minor=True)
    ax1.set_yticks([0, 4, 8, 12])
    ax1.set_yticks([2, 6, 10], minor=True)
    ax1.grid(which="both")
    ax1.set_xlim(10, 30)     # Chosen after plot had been inspected
    ax1.set_ylim(0, 13)
    ax1.legend(loc="lower right")
    ax1.set_title(r"Baryonic number density, $n_B$", fontdict={"fontsize": 10})
    ax2.plot(mu_B, ps, label=r"$p(\mu_B)$", color=num_to_col.to_rgba(1))
    ax2.scatter(crit_mu, crit_p, color="black", s=15, marker="x", zorder=2)
    ax2.plot(mu_B, epss, label=r"$\epsilon(\mu_B)$", color=num_to_col.to_rgba(2))
    ax2.scatter(crit_mu, crit_eps, color="black", s=15, marker="x", zorder=2)
    ax2.set_xlabel(r"$\mu_B \, / \, f_\pi$")
    ax2.set_ylabel(r"$f_\pi^{4}$")
    ax2.set_xticks([7.5, 12.5, 17.5, 22.5, 27.5], minor=True)
    ax2.set_yticks([25, 75, 125, 175], minor=True)
    ax2.grid(which="both")
    ax2.set_xlim(10, 30)
    ax2.set_ylim(0, 175)
    ax2.legend()
    ax2.set_title(r"Pressure, $p$, and energy density, $\epsilon$", fontdict={"fontsize": 10})
    ax3.plot(ps * 0.00974, epss * 0.00974, label=r"$\epsilon(p)$", color=num_to_col.to_rgba(3))
    ax3.scatter(crit_p * 0.00974, crit_eps * 0.00974, color="black", s=15, marker="x", zorder=2)
    # ^ Multiply by 0.00974 to convert to GeV / fm^3
    ax3.set_xlabel(r"$p \, / \, \unit{}$".format("{\giga \eV \per\cubic" + "\{}".format("femto") + "\metre}"))
    ax3.set_ylabel(r"$\epsilon \, / \, \unit{}$".format("{\giga \eV \per\cubic" + "\{}".format("femto") + "\metre}"))

    # Smaller window
    ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax3.set_xticks([0.125, 0.375, 0.625, 0.875], minor=True)
    ax3.set_yticks([0, 0.5, 1, 1.5])
    ax3.set_yticks([0.25, 0.75, 1.25], minor=True)
    ax3.set_xlim(-0.05, 1)
    ax3.set_ylim(0, 1.5)
    ax3.grid(which="both")
    ax3.legend(loc="lower right")
    ax3.set_title(r"APR equation of state, $\epsilon(p)$", fontdict={"fontsize": 10})
    # Plotting black box for zoom:
    ax3.plot((-0.01, 0.06, 0.06, -0.01, -0.01), (0, 0, 0.5, 0.5, 0), color="black", linewidth=0.7)
    ax4.plot(ps[0:130] * 0.00974, epss[0:130] * 0.00974, label=r"$\epsilon(p)$", color=num_to_col.to_rgba(3))
    # ^ Multiply by 0.00974 to convert to GeV / fm^3
    ax4.set_xlabel(r"$p \, / \, \unit{}$".format("{\giga \eV \per\cubic" + "\{}".format("femto") + "\metre}"))
    ax4.set_ylabel(r"$\epsilon \, / \, \unit{}$".format("{\giga \eV \per\cubic" + "\{}".format("femto") + "\metre}"))
    ax4.set_xticks([0, 0.02, 0.04, 0.06])
    ax4.set_xticks([-0.01, 0.01, 0.03, 0.05], minor=True)
    ax4.set_yticks([0, 0.2, 0.4])
    ax4.set_yticks([0.1, 0.3, 0.5], minor=True)
    ax4.grid(which="both")
    ax4.set_xlim(-0.01, 0.06)
    ax4.set_ylim(0, 0.5)
    ax4.legend(loc="lower right")
    ax4.set_title(r"Zoom APR EoS", fontdict={"fontsize": 10})
    # Uncomment to save figure
    # plt.savefig(fname="nuclear_matter_properties.eps", format="eps", dpi=600)
    plt.show()


"""Uncomment to generate plots for nuclear matter properties"""
# nuclear_matter_plotting()


def hybrid_star_equation_of_state_patching_plot(m_sigmas):
    """Best suited for having 3 m_sigmas. Creates the first figure in chapter 12.2."""
    mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_shifted_neutron_mass()
    handles = []
    fig, (axs1, axs2) = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 4))
    to_GeV_per_cubic_fm = 0.00974
    for n_col in range(len(axs1)):
        axs1[n_col].set_xlim(10, 16)
        axs1[n_col].set_ylim(0, 40)
        line_nuc, = axs1[n_col].plot(mu_Bs_nuc, ps_nuc, color="black", label=r"APR")
        axs1[n_col].set_xlabel(r"$\mu_B \, / \, f_\pi$")
        axs1[n_col].set_xticks([11, 13, 15], minor=True)
        axs1[n_col].grid()

        axs2[n_col].set_xlim(0, 0.4)
        axs2[n_col].set_ylim(0, 1.5)
        axs2[n_col].plot(ps_nuc * to_GeV_per_cubic_fm, epss_nuc * to_GeV_per_cubic_fm,
                         color="black", label=r"$\epsilon_\text{N}(p_\text{N})$")
        axs2[n_col].grid()
        axs2[n_col].set_xlabel(r"$p \, / \, \unit{\giga \eV \per\cubic\femto\metre}$")
    handles.append(line_nuc)

    cmap = cm.get_cmap("viridis")
    delta_m_sigma = max(m_sigmas) - min(m_sigmas)
    norm = colors.Normalize(vmin=min(m_sigmas) - delta_m_sigma * 0.3, vmax=max(m_sigmas))
    num_to_col = cm.ScalarMappable(norm=norm, cmap=cmap)

    for n, m_sigma in enumerate(m_sigmas):
        (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, n_mf=2000)
        mu_Bqs = 3 / 2 * (mu_us + mu_ds)
        crit_mu, crit_p, B = find_critical_mu_B(ps, mu_Bqs, ps_nuc, mu_Bs_nuc)
        crit_n_nuc, crit_n_q = np.interp(crit_mu, mu_Bs_nuc, n_Bs_nuc), np.interp(crit_mu, mu_Bqs, ns / 3)
        print("Critical baryonic number density APR: {} (dim.less, n / f_pi^3), {} (dim.ful fm^(-3))".format(crit_n_nuc, crit_n_nuc * 0.1047))
        print("Critical baryonic number density QM: {} (dim.less, n / f_pi^3), {} (dim.ful fm^(-3))".format(crit_n_q, crit_n_q * 0.1047))
        print("These number densities occur at p_crit: {} (dim.ful Pa)".format(crit_p * 1.56 * 10 ** 33))
        n_crit_p = np.argmin(np.abs(ps - crit_p))
        eps_crit_q = np.interp(crit_p, ps, epss)
        eps_crit_nuc = np.interp(crit_p, ps_nuc, epss_nuc)
        line_q, = axs1[n].plot(mu_Bqs, (ps - B),
                               color=num_to_col.to_rgba(m_sigma),
                               label=r"$m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        handles.append(line_q)
        axs2[n].plot((ps - B) * to_GeV_per_cubic_fm, (epss + B) * to_GeV_per_cubic_fm,
                     color=num_to_col.to_rgba(m_sigma), label=r"$\epsilon_q(p_q)$")
        axs1[n].scatter(crit_mu, crit_p, color="black", s=10, zorder=3)
        axs2[n].scatter((crit_p * to_GeV_per_cubic_fm, crit_p * to_GeV_per_cubic_fm),
                        (eps_crit_nuc * to_GeV_per_cubic_fm, eps_crit_q * to_GeV_per_cubic_fm),
                        color="black", s=10, zorder=3)
        axs1[n].plot((0, crit_mu), (crit_p, crit_p), linestyle="dotted", color="grey", linewidth=1.1)
        axs2[n].plot((crit_p * to_GeV_per_cubic_fm, crit_p * to_GeV_per_cubic_fm),
                     (0, epss[n_crit_p] * to_GeV_per_cubic_fm), linestyle="dotted", color="grey", linewidth=1.1)
        axs1[n].plot((crit_mu, crit_mu), (0, crit_p), linestyle="dotted", color="grey", linewidth=1.1)
        axs1[n].set_xticks([crit_mu], minor=True)
        axs1[n].set_xticklabels([r"$\mu_\text{c}$"], minor=True, fontdict={"fontsize": 8})
        axs1[n].set_yticks([crit_p], minor=True)
        axs1[n].set_yticklabels([r"$p_\text{c}$"], minor=True, fontdict={"fontsize": 8})
        axs1[n].tick_params(which="minor", axis="x", direction="in", pad=-8)
        axs1[n].tick_params(which="minor", axis="y", direction="in", pad=-10)
        axs2[n].set_xticks([crit_p * to_GeV_per_cubic_fm], minor=True)
        axs2[n].set_xticklabels([r"$p_\text{c}$"], minor=True, fontdict={"fontsize": 8})
        axs2[n].tick_params(which="minor", axis="x", direction="in", pad=-8)
        # Constructing hybrid EoS:
        # For a pretty plot, we need a suited p-array
        ps_hybrid = np.concatenate(([p for p in ps if min(ps_nuc) <= p < crit_p], [crit_p-0.00001, crit_p + 0.00001],
                                    [p for p in ps if crit_p < p]))
        eps_hybrid_arr = EoS_hybrid(ps - B, epss + B, ps_nuc, epss_nuc, crit_p)(ps_hybrid)
        line_hybrid, = axs2[n].plot((ps_hybrid - B) * to_GeV_per_cubic_fm, eps_hybrid_arr * to_GeV_per_cubic_fm,
                                    color="red", linestyle=(0, (5, 5)), linewidth=1.2,
                                    label="Hybrid")
    handles.append(line_hybrid)
    axs1[0].legend(handles=handles[0:2], bbox_to_anchor=(0., 1.13, 1., .07), loc='upper center', mode="expand",
                   borderaxespad=0., ncol=2, fancybox=True, fontsize="small")
    axs1[1].legend(handles=[handles[2]], bbox_to_anchor=(0., 1.13, 1., .07), loc='upper center', mode="expand",
                   borderaxespad=0., fancybox=True, fontsize="small")
    axs1[2].legend(handles=handles[3:], bbox_to_anchor=(0., 1.13, 1., .07), loc='upper center', mode="expand",
                   borderaxespad=0., ncol=2, fancybox=True, fontsize="small")

    axs2[1].tick_params(labelleft=False)
    axs2[2].tick_params(labelleft=False)
    axs1[1].tick_params(labelleft=False)
    axs1[2].tick_params(labelleft=False)
    axs1[0].set_ylabel(r"$p \, / \, f_\pi^4$")
    axs2[0].set_ylabel(r"$\epsilon \, / \, \unit{\giga \eV \per\cubic\femto\metre}$")
    plt.tight_layout()
    # Uncomment to save figure
    # plt.savefig(fname="Hybrid_equation_of_state.eps", format="eps", dpi=600)
    plt.show()


"""Uncomment to generate hybrid equation of state plots."""
# hybrid_star_equation_of_state_patching_plot([400, 500, 600])


def plot_polynomial_unification(lim_APR_n, lim_q_n):
    """Using polynomial unificator to unify APR and QM EoS.
    lim_APR_n is the limiting number density in units of nuclear saturation densities, n_0, where we stop trusting APR.
    lim_q_n is the limiting number density in units of nuclear saturation densities where we enter
    the pure quark phase.
    This creates the first figure in chapter 12.3."""
    mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_shifted_neutron_mass()
    # Need to choose where to start distrusting APR. Standard: 2 * n_0
    n_0 = 0.16  # units of fm^(-3). Remember: Conversion factor to f_pi^(-3): 1 / 0.1047
    distrust_APR_n = lim_APR_n * n_0 / 0.1047
    distrust_APR_mu = np.interp(distrust_APR_n, n_Bs_nuc, mu_Bs_nuc)
    distrust_APR_p = np.interp(distrust_APR_mu, mu_Bs_nuc, ps_nuc)
    distrust_APR_eps = np.interp(distrust_APR_mu, mu_Bs_nuc, epss_nuc)
    ps_nuc_trunc = np.array([p_nuc for p_nuc in ps_nuc if p_nuc < distrust_APR_p] + [distrust_APR_p])
    mus_nuc_trunc = np.array([mu_nuc for mu_nuc in mu_Bs_nuc if mu_nuc < distrust_APR_mu] + [distrust_APR_mu])
    ns_nuc_trunc = np.array([n_nuc for n_nuc in n_Bs_nuc if n_nuc < distrust_APR_n] + [distrust_APR_n])
    epss_nuc_trunc = np.array([eps_nuc for eps_nuc in epss_nuc if eps_nuc < distrust_APR_eps] + [distrust_APR_eps])

    print("distrust_APR_mu: {}, distrust_APR_p {} [Pa]".format(distrust_APR_mu, distrust_APR_p * 1.56*10**33))
    n_diff_mu = np.gradient(n_Bs_nuc, mu_Bs_nuc)
    n_APR_diff_mu_at_distrust = np.interp(distrust_APR_mu, mu_Bs_nuc, n_diff_mu)
    print("d n / d mu = {} at limiting APR_mu".format(n_APR_diff_mu_at_distrust))

    fig = plt.figure(figsize=(6, 5.5), layout="constrained")
    subfig1, subfig2 = fig.subfigures(2, 1, height_ratios=[1, 1])
    ax1, ax2 = subfig1.subplots(nrows=1, ncols=2)
    ax3 = subfig2.subplots(nrows=1, ncols=1)
    ax1.set_title(r"Unified $p(\mu)$", fontdict={"fontsize": 10})
    ax2.set_title(r"Unified $n(\mu)$", fontdict={"fontsize": 10})
    ax3.set_title(r"Unified $\epsilon(p)$", fontdict={"fontsize": 10})
    ax3.axes.set_aspect(aspect=0.2)
    ax2.plot(mus_nuc_trunc, ns_nuc_trunc * 0.1047, color="black", label=r"APR $n(\mu_B)$")
    ax2.scatter(distrust_APR_mu, distrust_APR_n * 0.1047, color="black", s=10, zorder=3)
    ax2.plot((0, distrust_APR_mu), (distrust_APR_n * 0.1047, distrust_APR_n * 0.1047),
             color="black", linestyle=(0, (2, 2)), linewidth=1.0)
    ax1.plot(mus_nuc_trunc, ps_nuc_trunc, color="black", label=r"APR $p(\mu_B)$")
    ax1.scatter(distrust_APR_mu, distrust_APR_p, color="black", s=10, zorder=3)

    to_SI = 0.00974     # To go from normalised pressure, energy density to GeV / fm^3

    legend_handles = []

    line, = ax3.plot(ps_nuc_trunc * to_SI, epss_nuc_trunc * to_SI, color="black", label="APR")
    legend_handles.append(line)
    ax3.scatter(distrust_APR_p * to_SI, distrust_APR_eps * to_SI, color="black", s=10, zorder=3)

    # When we search for parameters for the "smoothest" polynomial, we may tune the range of (mu, n) we look at.
    # See find_best_polynomial_params. These contain (i_mu, search_interval_mu_rel) and (i_n, search_interval_n_rel)
    m_sigmas_and_mu_search = {400: (30, [0.05, 0.05]), 500: (30, [0.05, 0.05]), 600: (30, [0.01, 0.7])}
    m_sigmas_and_n_search = {400: (30, [0.05, 0.05]), 500: (30, [0.05, 0.05]), 600: (30, [0.01, 0.7])}
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=min(m_sigmas_and_mu_search.keys()), vmax=max(m_sigmas_and_mu_search.keys()))
    num_to_col = cm.ScalarMappable(norm, cmap)

    for m_sigma in m_sigmas_and_mu_search.keys():
        print("m_sigma = {} [MeV]".format(m_sigma))
        # Need to choose where the disbelief in quark matter starts being significant. Standard: 4 * n_0
        (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002, n_mf=1000,
                                                                         bag_extra=0.0)
        distrust_quark_n = lim_q_n * n_0 / 0.1047
        n_Bs_q = 1 / 3 * ns  # Looking at baryonic number density, not total quark density
        mu_Bs_q = 3 / 2 * (mu_us + mu_ds)
        distrust_quark_mu = np.interp(distrust_quark_n, n_Bs_q, mu_Bs_q)
        distrust_quark_p = np.interp(distrust_quark_mu, mu_Bs_q, ps)
        distrust_quark_eps = np.interp(distrust_quark_mu, mu_Bs_q, epss)
        print("Limiting mu_q: {} [f_pi^(1)], limiting p_q: {} [f_pi^(4)] ({} Pa)"
              " and limiting n_q: {} [f_pi^(3)]".format(distrust_quark_mu, distrust_quark_p,
                                                        distrust_quark_p * 1.56 * 10**33, distrust_quark_n))
        ps_q_trunc = np.array([distrust_quark_p] + [p_q for p_q in ps if p_q > distrust_quark_p])
        mu_q_trunc = np.array([distrust_quark_mu] + [mu_q for mu_q in mu_Bs_q if mu_q > distrust_quark_mu])
        n_q_trunc = np.array([distrust_quark_n] + [n_q for n_q in n_Bs_q if n_q > distrust_quark_n])
        epss_q_trunc = np.array([distrust_quark_eps] + [eps_q for eps_q in epss if eps_q > distrust_quark_eps])
        n_q_diff_mu_at_distrust = np.interp(distrust_quark_mu, mu_Bs_q, np.gradient(n_Bs_q, mu_Bs_q))
        print("d n / d mu = {} at limiting mu_q".format(n_q_diff_mu_at_distrust))
        # Now we may search for parameters (mu, n) which gives the "smoothest" n(mu).
        i_mu, search_interval_mu = m_sigmas_and_mu_search[m_sigma]
        i_n, search_interval_n = m_sigmas_and_n_search[m_sigma]
        if m_sigma == 600:
            n_APR_diff_mu_at_distrust -= 0.8    # Need to not be as strict in order to find
            # physcially acceptable solution. This value was elected through trial and error
        APR_limit_values = [distrust_APR_mu, distrust_APR_p, distrust_APR_n, n_APR_diff_mu_at_distrust]
        q_limit_values = [distrust_quark_mu, distrust_quark_p, distrust_quark_n, n_q_diff_mu_at_distrust]
        (mu_best, n_best), (unificator, unificator_diff, unificator_diff_diff) = find_best_polynomial_params(unific_3_3_free_midpoint, APR_limit_values, q_limit_values, i_mu=i_mu, i_n=i_n,
                                                        search_interval_mu_rel=search_interval_mu,
                                                        search_interval_n_rel=search_interval_n)
        mu_intermittent = np.linspace(distrust_APR_mu, distrust_quark_mu, 300, endpoint=True)
        p_unified = unificator(mu_intermittent, mu_best, n_best, n_APR_diff_mu_at_distrust)
        n_unified = unificator_diff(mu_intermittent, mu_best, n_best, n_APR_diff_mu_at_distrust)
        epss_unified = - p_unified + n_unified * mu_intermittent      # Simplest energy density estimate

        ax2.plot(mu_intermittent, n_unified * 0.1047, color=num_to_col.to_rgba(m_sigma), linestyle=(0, (3, 3)))
        ax2.plot(mu_q_trunc, n_q_trunc * 0.1047, color=num_to_col.to_rgba(m_sigma),
                 label=r"QM $n, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        ax2.scatter(distrust_quark_mu, distrust_quark_n * 0.1047, color="black", s=10, zorder=3)
        ax2.scatter(mu_best, n_best * 0.1047, color=num_to_col.to_rgba(m_sigma), s=10, zorder=3)

        ax1.plot(mu_q_trunc, ps_q_trunc, color=num_to_col.to_rgba(m_sigma),
                 label=r"QM $p, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        ax1.scatter(distrust_quark_mu, distrust_quark_p, s=10, zorder=3, color="black")
        ax1.plot(mu_intermittent, p_unified, color=num_to_col.to_rgba(m_sigma), linestyle=(0, (3, 3)))
        line, = ax3.plot(ps_q_trunc * to_SI, epss_q_trunc * to_SI, color=num_to_col.to_rgba(m_sigma),
                         label=r"QM, $m_\sigma={} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        legend_handles.append(line)
        line, = ax3.plot(p_unified * to_SI, epss_unified * to_SI, color=num_to_col.to_rgba(m_sigma),
                         linestyle=(0, (3, 3)),
                         label=r"Interpolation, $m_\sigma={} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        legend_handles.append(line)
        ax3.scatter(distrust_quark_p * to_SI, distrust_quark_eps * to_SI, color="black", s=10, zorder=3)
        ax3.scatter(p_unified[-1] * to_SI, epss_unified[-1] * to_SI, color="black", s=10, zorder=3)

    mu_lower, mu_upper = min(mu_Bs_nuc), 14.5
    ax2.plot((0, distrust_quark_mu), (distrust_quark_n * 0.1047, distrust_quark_n * 0.1047),
             color="black", linestyle=(0, (2, 2)), linewidth=1.0)

    ax2.set_xlim(mu_lower, mu_upper)
    ax2.set_ylim(0, 5.5 * n_0)
    ax2.set_xlabel(r"$\mu_B \, / \, f_\pi$")
    ax2.set_ylabel(r"$n \, / \, \unit{\per\cubic\femto\metre}$")
    ax2.grid()
    ax2.set_yticks([lim_APR_n * n_0, lim_q_n * n_0], minor=True)
    ax2.set_yticklabels([r"${}n_0$".format(lim_APR_n), r"${}n_0$".format(lim_q_n)], minor=True,
                        fontdict={"fontsize": 8})
    ax2.tick_params(which="minor", axis="y", direction="out", pad=1, zorder=4, labelcolor="red")
    ax1.set_xlim(mu_lower, mu_upper)
    ax1.set_ylim(0, 25)
    ax1.set_ylabel(r"$p \, / \, f_\pi^4$")
    ax1.set_xlabel(r"$\mu_B \, / \, f_\pi$")
    ax1.grid()

    ax3.plot()
    ax3.set_xlim(0, 0.25)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel(r"$p \, / \, \unit{\giga \eV \per\cubic\femto\metre}$")
    ax3.set_ylabel(r"$\epsilon \, / \, \unit{\giga \eV \per\cubic\femto\metre}$")
    ax3.grid()
    ax3.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), fontsize="small")

    # Uncomment to save figure
    # plt.savefig("Unified_EoS_polynomial_3_3.eps", format="eps", dpi=600)
    plt.show()


"""Run to get polynomial unification p(mu), n(mu) and eps(p)"""
# plot_polynomial_unification(2, 4)


def plot_polynomial_unification_unshifted(lim_APR_n, lim_q_n):
    """Using polynomial unificator to unify APR and QM EoS.
        lim_APR_n is the limiting number density in units of nuclear saturation densities, n_0, where we stop trusting APR.
        lim_q_n is the limiting number density in units of nuclear saturation densities where we enter
        the pure quark phase. The same as above, but now we use the neutron mass m_n = 939.6 MeV.
        This creates the fourth figure in chapter 12.3"""
    mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_mu_B_n_B_p_eps()
    # Need to choose where to start distrusting APR. Standard: 2 * n_0
    n_0 = 0.16  # units of fm^(-3). Remember: Conversion factor to f_pi^(-3): 1 / 0.1047
    distrust_APR_n = lim_APR_n * n_0 / 0.1047
    distrust_APR_mu = np.interp(distrust_APR_n, n_Bs_nuc, mu_Bs_nuc)
    distrust_APR_p = np.interp(distrust_APR_mu, mu_Bs_nuc, ps_nuc)
    distrust_APR_eps = np.interp(distrust_APR_mu, mu_Bs_nuc, epss_nuc)
    ps_nuc_trunc = np.array([p_nuc for p_nuc in ps_nuc if p_nuc < distrust_APR_p] + [distrust_APR_p])
    mus_nuc_trunc = np.array([mu_nuc for mu_nuc in mu_Bs_nuc if mu_nuc < distrust_APR_mu] + [distrust_APR_mu])
    ns_nuc_trunc = np.array([n_nuc for n_nuc in n_Bs_nuc if n_nuc < distrust_APR_n] + [distrust_APR_n])
    epss_nuc_trunc = np.array([eps_nuc for eps_nuc in epss_nuc if eps_nuc < distrust_APR_eps] + [distrust_APR_eps])

    print("distrust_APR_mu: {}, distrust_APR_p {} [Pa]".format(distrust_APR_mu, distrust_APR_p * 1.56 * 10 ** 33))
    n_diff_mu = np.gradient(n_Bs_nuc, mu_Bs_nuc)
    n_APR_diff_mu_at_distrust = np.interp(distrust_APR_mu, mu_Bs_nuc, n_diff_mu)
    print("d n / d mu = {} at limiting APR_mu".format(n_APR_diff_mu_at_distrust))

    fig = plt.figure(figsize=(6, 5.5), layout="constrained")
    subfig1, subfig2 = fig.subfigures(2, 1, height_ratios=[1, 1])
    ax1, ax2 = subfig1.subplots(nrows=1, ncols=2)
    ax3 = subfig2.subplots(nrows=1, ncols=1)
    ax1.set_title(r"Unified $p(\mu)$", fontdict={"fontsize": 10})
    ax2.set_title(r"Unified $n(\mu)$", fontdict={"fontsize": 10})
    ax3.set_title(r"Unified $\epsilon(p)$", fontdict={"fontsize": 10})
    ax2.plot(mus_nuc_trunc, ns_nuc_trunc * 0.1047, color="black", label=r"APR $n(\mu_B)$")
    ax2.scatter(distrust_APR_mu, distrust_APR_n * 0.1047, color="black", s=10, zorder=3)
    ax2.plot((0, distrust_APR_mu), (distrust_APR_n * 0.1047, distrust_APR_n * 0.1047),
             color="black", linestyle=(0, (2, 2)), linewidth=1.0)
    ax1.plot(mus_nuc_trunc, ps_nuc_trunc, color="black", label=r"APR $p(\mu_B)$")
    ax1.scatter(distrust_APR_mu, distrust_APR_p, color="black", s=10, zorder=3)

    to_SI = 0.00974  # To go from normalised pressure, energy density to GeV / fm^3

    legend_handles = []

    line, = ax3.plot(ps_nuc_trunc * to_SI, epss_nuc_trunc * to_SI, color="black", label="APR")
    legend_handles.append(line)
    ax3.scatter(distrust_APR_p * to_SI, distrust_APR_eps * to_SI, color="black", s=10, zorder=3)

    m_sigmas = [400, 500, 600]
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=min(m_sigmas), vmax=max(m_sigmas))
    num_to_col = cm.ScalarMappable(norm, cmap)

    for m_sigma in m_sigmas:
        print("m_sigma = {} [MeV]".format(m_sigma))
        # Need to choose where the disbelief in quark matter starts being significant. Standard: 4 * n_0
        (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002, n_mf=1000,
                                                                         bag_extra=0.0)
        distrust_quark_n = lim_q_n * n_0 / 0.1047
        n_Bs_q = 1 / 3 * ns  # Looking at baryonic number density, not total quark density
        mu_Bs_q = 3 / 2 * (mu_us + mu_ds)
        distrust_quark_mu = np.interp(distrust_quark_n, n_Bs_q, mu_Bs_q)
        distrust_quark_p = np.interp(distrust_quark_mu, mu_Bs_q, ps)
        distrust_quark_eps = np.interp(distrust_quark_mu, mu_Bs_q, epss)
        print("Limiting mu_q: {} [f_pi^(-1)], limiting p_q: {} [f_pi^(-4)] ({} Pa)"
              " and limiting n_q: {} [f_pi^(-3)]".format(distrust_quark_mu, distrust_quark_p,
                                                         distrust_quark_p * 1.56 * 10 ** 33, distrust_quark_n))
        ps_q_trunc = np.array([distrust_quark_p] + [p_q for p_q in ps if p_q > distrust_quark_p])
        mu_q_trunc = np.array([distrust_quark_mu] + [mu_q for mu_q in mu_Bs_q if mu_q > distrust_quark_mu])
        n_q_trunc = np.array([distrust_quark_n] + [n_q for n_q in n_Bs_q if n_q > distrust_quark_n])
        epss_q_trunc = np.array([distrust_quark_eps] + [eps_q for eps_q in epss if eps_q > distrust_quark_eps])
        n_q_diff_mu_at_distrust = np.interp(distrust_quark_mu, mu_Bs_q, np.gradient(n_Bs_q, mu_Bs_q))
        print("d n / d mu = {} at limining mu_q".format(n_q_diff_mu_at_distrust))
        # Now we may search for paramters (mu, n) which gives the "smoothest" n(mu).
        APR_limit_values = [distrust_APR_mu, distrust_APR_p, distrust_APR_n, n_APR_diff_mu_at_distrust]
        q_limit_values = [distrust_quark_mu, distrust_quark_p, distrust_quark_n, n_q_diff_mu_at_distrust]
        i_mu, i_n = 30, 30
        search_interval_mu, search_interval_n = [0.1, 0.1], [0.1, 0.1]
        (mu_best, n_best), (unificator, unificator_diff, unificator_diff_diff) = find_best_polynomial_params(
            unific_3_3_free_midpoint, APR_limit_values, q_limit_values, i_mu=i_mu, i_n=i_n,
            search_interval_mu_rel=search_interval_mu,
            search_interval_n_rel=search_interval_n)
        mu_intermittent = np.linspace(distrust_APR_mu, distrust_quark_mu, 300, endpoint=True)
        p_unified = unificator(mu_intermittent, mu_best, n_best, n_APR_diff_mu_at_distrust)
        n_unified = unificator_diff(mu_intermittent, mu_best, n_best, n_APR_diff_mu_at_distrust)
        epss_unified = - p_unified + n_unified * mu_intermittent  # Simplest energy density estimate
        ax2.plot(mu_intermittent, n_unified * 0.1047, color=num_to_col.to_rgba(m_sigma), linestyle=(0, (3, 3)))
        ax2.plot(mu_q_trunc, n_q_trunc * 0.1047, color=num_to_col.to_rgba(m_sigma),
                 label=r"QM $n, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        ax2.scatter(distrust_quark_mu, distrust_quark_n * 0.1047, color="black", s=10, zorder=3)
        ax2.scatter(mu_best, n_best * 0.1047, color=num_to_col.to_rgba(m_sigma), s=10, zorder=3)

        ax1.plot(mu_q_trunc, ps_q_trunc, color=num_to_col.to_rgba(m_sigma),
                 label=r"QM $p, \, m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        ax1.scatter(distrust_quark_mu, distrust_quark_p, s=10, zorder=3, color="black")
        ax1.plot(mu_intermittent, p_unified, color=num_to_col.to_rgba(m_sigma), linestyle=(0, (3, 3)))
        line, = ax3.plot(ps_q_trunc * to_SI, epss_q_trunc * to_SI, color=num_to_col.to_rgba(m_sigma),
                         label=r"QM, $m_\sigma={} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        legend_handles.append(line)
        line, = ax3.plot(p_unified * to_SI, epss_unified * to_SI, color=num_to_col.to_rgba(m_sigma),
                         linestyle=(0, (3, 3)),
                         label=r"Interpolation, $m_\sigma={} \, \unit{}$".format(m_sigma, "{\mega \eV}"))
        legend_handles.append(line)
        ax3.scatter(distrust_quark_p * to_SI, distrust_quark_eps * to_SI, color="black", s=10, zorder=3)
        ax3.scatter(p_unified[-1] * to_SI, epss_unified[-1] * to_SI, color="black", s=10, zorder=3)

    x_ticks = [10, 11, 12, 13, 14, 15, 16]
    mu_lower, mu_upper = min(mu_Bs_nuc), 16
    ax2.plot((0, distrust_quark_mu), (distrust_quark_n * 0.1047, distrust_quark_n * 0.1047),
             color="black", linestyle=(0, (2, 2)), linewidth=1.0)

    ax2.set_xlim(mu_lower, mu_upper)
    ax2.set_ylim(0, 1.1)    # Fits well for lim_q_n = 6
    ax2.set_xlabel(r"$\mu_B \, / \, f_\pi$")
    ax2.set_ylabel(r"$n \, / \, \unit{\per\cubic\femto\metre}$")
    ax2.grid()
    ax2.set_yticks([lim_APR_n * n_0, lim_q_n * n_0], minor=True)
    ax2.set_yticklabels([r"${}n_0$".format(lim_APR_n), r"${}n_0$".format(lim_q_n)], minor=True,
                        fontdict={"fontsize": 8})
    ax2.tick_params(which="minor", axis="y", direction="out", pad=1, zorder=4, labelcolor="red")
    ax2.set_xticks(x_ticks)
    ax1.set_xlim(mu_lower, mu_upper)
    ax1.set_ylim(0, 35)
    ax1.set_ylabel(r"$p \, / \, f_\pi^4$")
    ax1.set_xlabel(r"$\mu_B \, / \, f_\pi$")
    ax1.set_xticks(x_ticks)
    ax1.grid()

    p_lower, p_upper = 0, 0.35
    eps_lower, eps_upper = 0, 1.5
    square_aspect = (p_upper - p_lower) / (eps_upper - eps_lower)
    ax3.set_xlim(p_lower, p_upper)
    ax3.set_ylim(eps_lower, eps_upper)
    ax3.set_xlabel(r"$p \, / \, \unit{\giga \eV \per\cubic\femto\metre}$")
    ax3.set_ylabel(r"$\epsilon \, / \, \unit{\giga \eV \per\cubic\femto\metre}$")
    ax3.grid()
    ax3.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.1, 1),
               fontsize="small")
    ax3.axes.set_aspect(aspect=square_aspect * 0.85)
    # Uncomment to save figure
    # plt.savefig("Unified_EoS_polynomial_3_3_unshifted_m_n.eps", format="eps", dpi=600)
    plt.show()


"""Uncomment to generate unified p(mu), n(mu) and eps(p) for unshifted neutron mass, m_n."""
# plot_polynomial_unification_unshifted(2, 6)


def unificator_show_procedure_plot(unifier, unifier_params, shifted=True):
    """Shows what the polynomials look like for different parameters.
    This creates the first figure in Appendix I.2."""
    m_sigmas = [400, 500, 600]
    fig, axs = plt.subplots(ncols=len(m_sigmas), figsize=(7, 3), layout="constrained")
    distrust_APR_n = 2 * 0.16 / 0.1047
    distrust_quark_n = 4 * 0.16 / 0.1047
    if shifted:
        mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_shifted_neutron_mass()
    else:
        mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_mu_B_n_B_p_eps()
    distrust_APR_mu = np.interp(distrust_APR_n, n_Bs_nuc, mu_Bs_nuc)
    distrust_APR_p = np.interp(distrust_APR_mu, mu_Bs_nuc, ps_nuc)
    mus_nuc_trunc = np.array([mu_nuc for mu_nuc in mu_Bs_nuc if mu_nuc < distrust_APR_mu] + [distrust_APR_mu])
    ns_nuc_trunc = np.array([n_nuc for n_nuc in n_Bs_nuc if n_nuc < distrust_APR_n] + [distrust_APR_n])

    # Colouring using first unifier_params value, namely mu.
    min_color_value, max_color_value = min(unifier_params[:, 0]), max(unifier_params[:, 0])
    cmap, norm = cm.get_cmap("viridis"), colors.Normalize(vmin=min_color_value, vmax=max_color_value)
    num_to_col = cm.ScalarMappable(norm, cmap)
    for n, m_sigma in enumerate(m_sigmas):
        # Find all the quark EoS-related quantities.
        (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002)
        mu_Bs, n_Bs = 3 / 2 * (mu_us + mu_ds), 1 / 3 * ns  # Transforming into baryonic
        distrust_quark_mu = np.interp(distrust_quark_n, n_Bs, mu_Bs)
        distrust_quark_p = np.interp(distrust_quark_mu, mu_Bs, ps)
        n_diff_mu_q_at_distrust = np.interp(distrust_quark_mu, mu_Bs, np.gradient(n_Bs, mu_Bs))
        mus_q_trunc = np.array([distrust_quark_mu] + [mu_q for mu_q in mu_Bs if mu_q > distrust_quark_mu])
        ns_q_trunc = np.array([distrust_quark_n] + [n_q for n_q in n_Bs if n_q > distrust_quark_n])
        axs[n].set_xlim(10.2, distrust_quark_mu + (distrust_quark_mu - distrust_APR_mu) * 0.1)
        axs[n].set_ylim(2.3, 7)
        axs[n].set_xlabel(r"$\mu \, / \, f_\pi$")
        axs[n].set_ylabel(r"$n \, / \, f_\pi^3$")
        axs[n].set_title(r"$m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"), fontdict={"fontsize": 10})
        axs[n].plot(mus_nuc_trunc, ns_nuc_trunc, color="black")
        axs[n].scatter(distrust_APR_mu, distrust_APR_n, color="black", s=10, zorder=4)
        axs[n].plot(mus_q_trunc, ns_q_trunc, color="black")
        axs[n].scatter(distrust_quark_mu, distrust_quark_n, color="black", s=10, zorder=4)
        axs[n].grid()
        mu_intermittent = np.linspace(distrust_APR_mu, distrust_quark_mu, 300, endpoint=True)

        min_kink = np.infty
        best_params = None
        for param_set in unifier_params:
            unified, unified_diff, unified_diff_diff = unifier(distrust_APR_mu, distrust_APR_p, distrust_APR_n,
                                                               distrust_quark_mu, distrust_quark_p, distrust_quark_n)
            arr_diff = unified_diff(mu_intermittent, *param_set)
            arr_diff_diff = unified_diff_diff(mu_intermittent, *param_set)
            causal_condition = min(causality_condition(mu_intermittent, arr_diff, arr_diff_diff))
            end_kink = abs(arr_diff_diff[-1] - n_diff_mu_q_at_distrust)
            meeting_point_kink = max(np.abs([arr_diff_diff[i] - arr_diff_diff[i - 1] for i in range(1, len(arr_diff_diff))]))
            if causal_condition < 0:
                axs[n].plot(mu_intermittent, arr_diff, color=num_to_col.to_rgba(param_set[0]), linestyle=(0, (3, 3)),
                            linewidth=1.1, alpha=0.4)
                axs[n].scatter(param_set[0], param_set[1], color=num_to_col.to_rgba(param_set[0]), s=10)
            else:
                if end_kink + meeting_point_kink < min_kink:
                    min_kink = np.sqrt(end_kink**2 + meeting_point_kink**2)
                    best_params = param_set
                axs[n].plot(mu_intermittent, arr_diff, color=num_to_col.to_rgba(param_set[0]), linewidth=1.1)
                axs[n].scatter(param_set[0], param_set[1], color=num_to_col.to_rgba(param_set[0]), s=10)
        if best_params is not None:
            arr_diff_best = unified_diff(mu_intermittent, *best_params)
            axs[n].plot(mu_intermittent, arr_diff_best, color="red", zorder=3)
            axs[n].scatter(best_params[0], best_params[1], color="red", s=10)
    # Uncomment to save figure
    # plt.savefig(fname="polynomial_fit_m_sigma_500_n_N_2_n_q_4.svg", format="svg", dpi=600)
    plt.show()


"""Uncomment to run unificator_show_procedure_plot()"""
# n = 3 * 0.16 / 0.1047           # Chosen to be in the middle of 2 n_0 and 4 n_0.
# n_diff_mu_at_apr_stop = 2.0249  # Chosen for continuous second derivative
# unifier_parameters = np.array([[mu, n, n_diff_mu_at_apr_stop] for mu in np.linspace(10.7, 13, 10)])
# unificator_show_procedure_plot(unific_3_3_free_midpoint, unifier_parameters, shifted=True)


def best_free_midpoint_plot(unific, mu_min_shift, mu_max_shift, n_min_shift, n_max_shift, n_res, lim_APR_n=2,
                            lim_quark_n=4, all_m=True, delta_n_diff=0.0, shifted_m_n=True):
    # Firstly, we get all the nucleon EoS-related quantites we need.
    distrust_APR_n = 0.16 * lim_APR_n / 0.1047  # n_0 = 0.16 fm^(-3), 0.1047 converts from n / f_pi^3 -> fm^(-3)
    distrust_quark_n = 0.16 * lim_quark_n / 0.1047
    if shifted_m_n:
        mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_shifted_neutron_mass()
    else:
        mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_mu_B_n_B_p_eps()
    distrust_APR_mu = np.interp(distrust_APR_n, n_Bs_nuc, mu_Bs_nuc)
    distrust_APR_p = np.interp(distrust_APR_mu, mu_Bs_nuc, ps_nuc)
    n_diff_mu_at_distrust = np.interp(distrust_APR_mu, mu_Bs_nuc, np.gradient(n_Bs_nuc, mu_Bs_nuc))
    print("Slope of n(mu) of nuclear phase at the point "
          "we no longer trust the nuclear phase: {}".format(n_diff_mu_at_distrust))

    if all_m:
        # For each of the different m_sigma masses, we need to fit one polynomial.
        m_sigmas = [400, 500, 600]
        fig, axs = plt.subplots(ncols=3, figsize=(7, 3), layout="constrained")
        side_pad = 0.02
        fig.suptitle(r"$(\mu, n), \quad n_\text{} = {}n_0, \quad n_q = {}n_0$".format("{APR}", lim_APR_n, lim_quark_n),
                     fontsize=10, y=0.98, x=0.52)
    else:
        # For one figure only
        fig = plt.figure(figsize=(4, 3))
        m_sigmas = [600]
        side_pad = 0.005
        n_diff = 2.0249     # n_diff at end of APR-phase
        fraction = round((n_diff - delta_n_diff) / n_diff, 3)
        fig.suptitle("")
        subfig, subfig2 = fig.subfigures(ncols=2, width_ratios=[3, 1.1])
        ax = subfig.subplots()
        axs = [ax]
        subfig.suptitle(r"$(\mu, n), \quad n_\text{} = {}n_0, \quad n_q = {}n_0$".format("{APR}", lim_APR_n, lim_quark_n),
                     fontsize=10, y=0.98, x=0.52)
        subfig2.suptitle(r"Shifted $n'(\mu_{0})$".format("{B, \, l}"),
                         fontsize=10)

    for i, m_sigma in enumerate(m_sigmas):
        # Find all the quark EoS-related quantities.
        (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002)
        mu_Bs, n_Bs = 3 / 2 * (mu_us + mu_ds), 1 / 3 * ns    # Transforming into baryonic
        distrust_quark_mu = np.interp(distrust_quark_n, n_Bs, mu_Bs)
        distrust_quark_p = np.interp(distrust_quark_mu, mu_Bs, ps)
        n_diff_mu_q_at_distrust = np.interp(distrust_quark_mu, mu_Bs, np.gradient(n_Bs, mu_Bs))
        print("Slope of n(mu) of quark phase at the point"
              " we no longer trust the quark phase: {}".format(n_diff_mu_q_at_distrust))
        axs[i].grid()
        axs[i].set_xlabel(r"$\mu \, / \, f_\pi$")
        axs[i].set_ylabel(r"$n \, / \, f_\pi^3$")
        axs[i].set_title(r"$m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"), fontdict={"fontsize": 10})
        delta_mu, delta_n = distrust_quark_mu - distrust_APR_mu, distrust_quark_n - distrust_APR_n
        axs[i].set_xlim(distrust_APR_mu + (mu_min_shift - side_pad) * delta_mu,
                        distrust_quark_mu - (mu_max_shift - side_pad) * delta_mu)
        axs[i].set_ylim(distrust_APR_n + (n_min_shift - side_pad) * delta_n,
                        distrust_quark_n - (n_max_shift - side_pad) * delta_n)
        # Generating arrays for values of mu and n which lies between mu_nuc and mu_q, n_nuc and n_q, respectively:
        mu_coords = np.linspace(distrust_APR_mu + mu_min_shift * delta_mu, distrust_quark_mu - mu_max_shift * delta_mu,
                                n_res)
        n_coords = np.linspace(distrust_APR_n + n_min_shift * delta_n, distrust_quark_n - n_max_shift * delta_n,
                                n_res)
        unifier, unifier_diff, unifier_diff_diff = unific(distrust_APR_mu, distrust_APR_p, distrust_APR_n,
                                                          distrust_quark_mu, distrust_quark_p, distrust_quark_n)
        mu_intermittent = np.linspace(distrust_APR_mu, distrust_quark_mu, 300)
        # axs[i].plot((distrust_APR_mu, distrust_quark_mu), (3 * 0.16 / 0.1047, 3 * 0.16 / 0.1047), color="black",
        #             linestyle="dashed", alpha=0.8, linewidth=1.4)     # Adding line for where
        #             unificator_show_procedure has been applied
        color_lim_kink = 5     # By testing, this seems like a sensible value
        cmap = cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=0, vmax=color_lim_kink)
        norm2 = colors.Normalize(vmin=-1, vmax=0)
        num_to_col = cm.ScalarMappable(norm, cmap)
        num_to_col2 = cm.ScalarMappable(norm2, cmap)
        for mu in mu_coords:
            for n in n_coords:
                arr_diff = unifier_diff(mu_intermittent, mu, n, n_diff_mu_at_distrust - delta_n_diff)  # This is n(mu)
                arr_diff_diff = unifier_diff_diff(mu_intermittent, mu, n, n_diff_mu_at_distrust - delta_n_diff)    # This is n'(mu)
                causal = min(causality_condition(mu_intermittent, arr_diff, arr_diff_diff))
                kink_end = abs(arr_diff_diff[-1] - n_diff_mu_q_at_distrust)
                start_kink = abs(arr_diff_diff[0] - n_diff_mu_at_distrust)
                # The next may sound like a lot, so let's break it down. It is an array containing how the
                # double derivative acting upon p(mu) changes. A large change happens at a kink of n(mu). We wish to
                # choose an interpolating polynomial with little to no kinks in n.
                arr_diff_diff_differences = np.array([arr_diff_diff[i] - arr_diff_diff[i - 1] for i in range(1, len(arr_diff_diff))])
                kink = max(np.abs(arr_diff_diff_differences))   # If there is no kink, this number is small anyway.
                if causal < 0:
                    dot_faded = axs[i].scatter(mu, n, marker="x", color=num_to_col2.to_rgba(causal), alpha=0.3, s=10, label="Non-physical")
                else:
                    dot_coloured = axs[i].scatter(mu, n, s=10,
                                                  color=num_to_col.to_rgba(np.sqrt(start_kink**2 + kink_end**2
                                                                                   + kink**2)), label="Acceptable")
    # For only one panel, tight layout is best.
    if all_m:
        axs[-1].legend(handles=[dot_faded, dot_coloured], loc="lower center", bbox_to_anchor=(0., 1.09, 1., .07),
                       fontsize="small", ncol=2, handlelength=1)
        # Uncomment to save figure
        # plt.savefig(fname="m_n_coords_for_n_apr_{}_n_q_{}_m_n_APR.svg".format(lim_APR_n, lim_quark_n), format="svg",
        #             dpi=600)
    else:
        subfig2.legend(handles=[dot_faded, dot_coloured], loc="center",
                       fontsize="small", handlelength=1)
        plt.tight_layout()
        # Uncomment to save figure
        # plt.savefig(fname="m_n_coords_for_n_apr_{}_n_q_{}_m_sigma_600.svg".format(lim_APR_n, lim_quark_n),
        #             format="svg", dpi=600)
    plt.show()


"""Plot m_n-coordinates for all three:"""
# Uncomment for the last figure of Appendix I.2.
# best_free_midpoint_plot(unific_3_3_free_midpoint, 0.1, 0.1, 0.1, 0.1, 25, lim_APR_n=2, lim_quark_n=6, all_m=True,
#                         delta_n_diff=0.0, shifted_m_n=False)
# Uncomment for the second figure of Appendix I.2.
# best_free_midpoint_plot(unific_3_3_free_midpoint, 0.1, 0.1, 0.1, 0.1, 25, lim_APR_n=2, lim_quark_n=4, all_m=True,
#                         delta_n_diff=0.0, shifted_m_n=True)

"""Plot m_n-coordinates for only m_simga = 600 MeV"""
# Uncomment for the third figure
# best_free_midpoint_plot(unific_3_3_free_midpoint, 0.02, 0.7, 0.02, 0.7, 30, lim_APR_n=2, lim_quark_n=4, all_m=False,
#                         delta_n_diff=0.80)


def compare_equations_of_state_plot():
    """Compares the hybrid, pure APR and unified equation of state. Creates the second figure in chapter 12.3."""
    m_sigmas = [400, 500, 600]
    fig = plt.figure(layout="constrained", figsize=(6, 3))
    subfig, subfig2 = fig.subfigures(nrows=2, height_ratios=[1, 0.23])
    axs = subfig.subplots(ncols=3)
    subfig.suptitle("Comparison between hybrid and unified EoS", fontsize=10)
    cmap = cm.get_cmap("viridis")
    color_offset = m_sigmas[1] - m_sigmas[0]    # Would like to star colouring from lighter colours
    norm = colors.Normalize(vmin=min(m_sigmas) - color_offset, vmax=max(m_sigmas))
    num_to_col = cm.ScalarMappable(norm, cmap)
    num_to_col2 = cm.ScalarMappable(norm, cm.get_cmap("plasma"))
    conv_factor = 0.00974  # Going from f_pi^(-4) to GeV / fm^3
    handles = []
    for n, m_sigma in enumerate(m_sigmas):
        EoS_hybrid_first_order = EoS_hybrid_standard(m_sigma, conversion_factor=conv_factor)
        EoS_unified = EoS_unified_standard_3_3(m_sigma, conversion_factor=conv_factor)
        EoS_APR = EoS_pure_APR(conv_factor)
        ps = np.linspace(0, 0.5, 500)
        APR_line, = axs[n].plot(ps, EoS_APR(ps), color="black", label="APR")
        if n == 0:
            handles.append(APR_line)
        line, = axs[n].plot(ps, EoS_hybrid_first_order(ps), color=num_to_col2.to_rgba(m_sigma),
                            label=r"Hybrid, $m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"),
                            linestyle=(0, (3, 3)), zorder=3)
        handles.append(line)
        line, = axs[n].plot(ps, EoS_unified(ps), color=num_to_col.to_rgba(m_sigma),
                            label=r"Unified, $m_\sigma = {} \, \unit{}$".format(m_sigma, "{\mega \eV}"), zorder=2)
        handles.append(line)
        axs[n].set_xlim(0, 0.4)
        axs[n].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        axs[n].set_ylim(0, 1.5)
        axs[n].set_xlabel(r"$p \, / \, \unit{\giga\eV\per\cubic\femto\metre}$")
        axs[n].grid()

    subfig2.legend(handles=handles, handlelength=1.5, fontsize="small", ncol=3, loc="center",
                   fancybox=True)
    axs[1].tick_params(labelleft=False)
    axs[2].tick_params(labelleft=False)
    axs[0].set_ylabel(r"$\epsilon \, / \, \unit{\giga\eV\per\cubic\femto\metre}$")
    # Uncomment to save figure
    # plt.savefig(fname="Hybrid_unified_EoS_compare.eps", format="eps", dpi=600)
    plt.show()


"""Uncomment to generate plot comparing APR, hybrid and unified EoS."""
compare_equations_of_state_plot()
