import numpy as np
from scipy.optimize import root, root_scalar
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import interp1d
np.seterr(all="raise")

"""This file includes all the functions we need for the quark-meson model (QM model).

In the thesis, we mention what we call the inconsistently renormalised QM model and the consistently renormalised 
QM model. These will in short be referred to as incr and cr, respectively.

One of the tasks we must perform, is to determine the chemical potentials of the up and down quarks, mu_u and mu_d, 
as a function of the sigma mean field value, <sigma>. The mean field value will be abbreviated mf.
Adding an s at the end, e.g. mu_us or mfs, simply implies we are dealing with an array of several values.
Thus: mfs = (sigma) mean field values.

For brevity, we also write maxwell_construct() instead of construction. This is despite the fact that it is certainly 
a constuctuion, NOT a construct.

An addition of bar, e.g. mf_bar, indicates that we are dealing with the dimensionless form. In fact, we are 
dealing in dimensionless quantities almost everywhere, despite not writing _bar. We are only explicit about it
sometimes."""


# At first, we introduce the constants we use.
f_pi = 93                   # Pion decay constant
m_e_bar = 0.511 / f_pi      # Dimensionless electron mass
g = 300 / f_pi              # Coupling constant g, dimensionless
mu_scale_bar = 182 / f_pi   # Incr dim.reg. scale in dimensionless form
h_bar = (139 / f_pi) ** 2   # h parameter in dimensionless form. NB: Not to be confused with planck's reduced constant
m_pi_bar = 139 / f_pi       # Pion mass, dimensionless form
N_c, N_q = 3, 2             # Number of colours and flavours in our theory. Note that in the thesis, we use N_f instead
# of N_q.


def lambda_coef(m_sigma_bar):
    """Lambda in terms of m_sigma and m_pi at tree-level."""
    return 1/2 * (m_sigma_bar ** 2 - m_pi_bar**2)


def v_square_bar_coef(m_sigma_bar):
    """v square in terms of m_sigma at tree-level."""
    return (m_sigma_bar**2 - 3 * (m_pi_bar**2))/(m_sigma_bar**2 - m_pi_bar**2)


def omega_0_incr(mfs, m_sigma_bar):
    """Return the dim.less Omega_0 for the inconsistently renormalised QM model."""
    lambd, v_square_bar = lambda_coef(m_sigma_bar), v_square_bar_coef(m_sigma_bar)
    return lambd / 4 * mfs ** 4 - lambd * v_square_bar / 2 * mfs ** 2 - h_bar * mfs \
        + N_c * N_q * g ** 4 * mfs**4 / (16 * np.pi ** 2) \
        * (3 / 2 + np.log(mu_scale_bar ** 2 / (g ** 2 * mfs ** 2)))


def omega_0_diff_incr(mfs, m_sigma_bar):
    """Returns the derivative of the above incr Omega_0 with respect to mfs (<sigma>)."""
    lambd, v_square_bar = lambda_coef(m_sigma_bar), v_square_bar_coef(m_sigma_bar)
    return lambd * mfs ** 3 - lambd * v_square_bar * mfs - h_bar \
        + (N_c * N_q * g ** 4 * mfs ** 3) / (4 * np.pi ** 2) \
        * (1 + np.log(mu_scale_bar ** 2 / (g ** 2 * mfs ** 2)))


def omega_0_cr(mfs, m_sigma_bar):
    """Dim.less Omega_0, proper renormalising. Everything dim.less:
      - m_q = 300 MeV / f_pi,
      - m_pi = 139 MeV / f_pi
      - m_sigma_bar, passed as argument.
      Otherwise, follows notation from the thesis."""
    r_pi = np.sqrt(4 * g**2 / m_pi_bar**2 - 1)
    r_sigma = np.sqrt(4 * g**2 / m_sigma_bar**2 - 1)
    if not r_sigma:
        r_sigma = 0.000001  # To avoid divide by zero-error
    # We evaluate F(m_pi) and F(m_sigma)
    F_pi = 2 - 2 * r_pi * np.arctan(1 / r_pi)
    F_sigma = 2 - 2 * r_sigma * np.arctan(1 / r_sigma)
    # G is only evaluated on m_pi, G(m_pi)
    G_pi = 4 * g**2 / (m_pi_bar ** 2 * r_pi) * np.arctan(1 / r_pi) - 1
    g_N_c_frac = g**2 * N_c / (4 * np.pi ** 2)  # Commonly occurring factor
    return 3 / 4 * m_pi_bar**2 * (1 - g_N_c_frac * G_pi) * mfs**2 \
        - (m_sigma_bar**2) / 4 * (1 + g_N_c_frac * ((1 - 4 * g**2 / (m_sigma_bar**2)) * F_sigma
                                                    + 4 * g ** 2 / (m_sigma_bar**2) - F_pi - G_pi)) * mfs**2 \
        + (m_sigma_bar**2) / 8 * (1 - g_N_c_frac * (4 * g**2 / (m_sigma_bar**2) * (np.log(mfs ** 2))
                                                    - (1 - 4 * g**2 / (m_sigma_bar**2)) * F_sigma
                                                    + F_pi + G_pi)) * mfs**4 \
        - (m_pi_bar**2) / 8 * (1 - g_N_c_frac * G_pi) * mfs**4 + 3 / 4 * g**2 * g_N_c_frac * mfs**4 \
        - m_pi_bar**2 * (1 - g_N_c_frac * G_pi) * mfs


def omega_0_diff_cr(mfs, m_sigma_bar):
    """The derivative of omega_0_cr with respect to mfs."""
    r_pi = np.sqrt(4 * g ** 2 / m_pi_bar ** 2 - 1)
    r_sigma = np.sqrt(4 * g ** 2 / m_sigma_bar ** 2 - 1)
    if not r_sigma:
        r_sigma = 0.000001  # To avoid divide by zero-error
    # We evaluate F(m_pi) and F(m_sigma)
    F_pi = 2 - 2 * r_pi * np.arctan(1 / r_pi)
    F_sigma = 2 - 2 * r_sigma * np.arctan(1 / r_sigma)
    # G is only evaluated on m_pi, G(m_pi)
    G_pi = 4 * g ** 2 / (m_pi_bar ** 2 * r_pi) * np.arctan(1 / r_pi) - 1
    g_N_c_frac = g ** 2 * N_c / (4 * np.pi ** 2)  # Commonly occurring factor
    return 3 / 2 * m_pi_bar**2 * (1 - g_N_c_frac * G_pi) * mfs \
        - (m_sigma_bar**2) / 2 * (1 + g_N_c_frac * ((1 - 4 * g**2 / (m_sigma_bar**2)) * F_sigma
                                                    + 4 * g ** 2 / (m_sigma_bar**2) - F_pi - G_pi)) * mfs \
        + (m_sigma_bar**2) / 2 * (1 - g_N_c_frac * (4 * g**2 / (m_sigma_bar**2) * (np.log(mfs ** 2))
                                                    - (1 - 4 * g**2 / (m_sigma_bar**2)) * F_sigma
                                                    + F_pi + G_pi)) * mfs**3 \
        - (m_pi_bar**2) / 2 * (1 - g_N_c_frac * G_pi) * mfs**3 + 2 * g**2 * g_N_c_frac * mfs**3 \
        - m_pi_bar**2 * (1 - g_N_c_frac * G_pi)


def g_1(mus, mf, m_sigma_bar, omega_0_diff_func=omega_0_diff_incr):
    omega_0_diff = omega_0_diff_func(mf, m_sigma_bar)
    try:
        mu_u_dependent = (g ** 2 * mf * N_c) / (2 * np.pi ** 2) * np.sqrt(mus[0] ** 2 - g ** 2 * mf ** 2) * mus[0] \
                         - (g ** 4 * mf ** 3 * N_c) / (2 * np.pi ** 2) * (np.log(np.sqrt(mus[0] ** 2 / (g ** 2 * mf ** 2) - 1) + mus[0] / (g * mf)))
    except FloatingPointError:
        mu_u_dependent = 0
    try:
        mu_d_dependent = (g ** 2 * mf * N_c) / (2 * np.pi ** 2) * np.sqrt(mus[1] ** 2 - g ** 2 * mf ** 2) * mus[1] \
                         - (g ** 4 * mf ** 3 * N_c) / (2 * np.pi ** 2) * np.log(np.sqrt(mus[1] ** 2 / (g ** 2 * mf ** 2) - 1) + mus[1] / (g * mf))
    except FloatingPointError:
        mu_d_dependent = 0
    return omega_0_diff + mu_u_dependent + mu_d_dependent


def g_2(mus, mf):
    try:
        mu_u_term = 2 / 3 * (mus[0] ** 2 - g ** 2 * mf ** 2) ** (3 / 2)
    except FloatingPointError:
        mu_u_term = 0
    try:
        mu_d_term = - 1 / 3 * (mus[1] ** 2 - g ** 2 * mf ** 2) ** (3 / 2)
    except FloatingPointError:
        mu_d_term = 0
    try:
        mu_e_term = - 1 / 3 * ((mus[1] - mus[0])**2 - m_e_bar**2)**(3/2)
    except FloatingPointError:
        mu_e_term = 0
    return mu_u_term + mu_d_term + mu_e_term


def g_func(mus, mf, m_sigma_bar, omega_0_diff_func=omega_0_diff_incr):
    """For a given """
    return np.array([g_1(mus, mf, m_sigma_bar, omega_0_diff_func), g_2(mus, mf)])


def find_mus(mfs, m_sigma_bar, omega_0_diff):
    """Find mu_u and mu_d as a function of mfs."""
    mfs_and_mus = np.zeros((3, len(mfs)))   # Dropping _bar. Whole array is dim.less.
    mfs_and_mus[0, :] = mfs
    for n, mf in enumerate(mfs):
        root_sol = root(g_func, g * np.array([1.5 * mf, 2 * mf]),
                        args=(mf, m_sigma_bar, omega_0_diff), method="lm")
        assert root_sol.success
        vec = root_sol.x
        mfs_and_mus[1:, n] = vec
    return mfs_and_mus


def get_tags(mfs_and_mus):
    """Some solutions of the system of equations correspond to cases when some of the particles have
    a chemical potential less than the mass of the particle. We wish to distinguish these.
    Each set of mf, mu_u and mu_d generates a set of [ , , ]
    First digit = 1: mu_u < g mf
    Second didit = 1: mu_d < g mf
    Third digit = 1: mu_d - mu_u < m_e_bar
    """
    tags = np.zeros_like(mfs_and_mus)
    for n, (mf, mu_u, mu_d) in enumerate(zip(mfs_and_mus[0, :], mfs_and_mus[1, :], mfs_and_mus[2, :])):
        taglist = np.array([1, 1, 1])
        if mu_u < g * mf:
            taglist[0] = 0
        if mu_d < g * mf:
            taglist[1] = 0
        if mu_d - mu_u < m_e_bar:
            taglist[2] = 0
        tags[:, n] = taglist
    return tags


def separate_mfs_and_mus(mfs_and_mus):
    """Accept only mfs_and_mus where either tags are [1, 1, 1] (all particles present)
    or [1, 1, 0] (no electron present)."""
    tags = get_tags(mfs_and_mus)
    mfs_and_mus1 = np.array([[vev, mu_u, mu_d] for n, (vev, mu_u, mu_d) in enumerate(zip(mfs_and_mus[0, :], mfs_and_mus[1, :], mfs_and_mus[2, :])) if sum(tags[:, n]) == 3]).transpose()
    mfs_and_mus2 = np.array([[vev, mu_u, mu_d] for n, (vev, mu_u, mu_d) in enumerate(zip(mfs_and_mus[0, :], mfs_and_mus[1, :], mfs_and_mus[2, :])) if np.all([tags[:, n], np.array([1, 1, 0])])]).transpose()
    return mfs_and_mus1, mfs_and_mus2


def chem_pot_term_pres(mus, masses):
    """Chemical potentials and corresponding masses as input. This returns the brackets of Eq. (11.42). Note: Does NOT
    include the numerical prefactor."""
    return 1 / 3 * (mus ** 2 - masses ** 2)**(3 / 2) * mus + (masses ** 4) / 2 * np.log(np.sqrt(mus ** 2 / masses**2 - 1) + mus / masses) - (masses ** 2) / 2 * np.sqrt(mus ** 2 - masses ** 2) * mus


def chem_pot_term_eps(mus, masses):
    """Chemical potentials and corresponding masses as input. This returns the brackets of Eq. (11.44). Note: Does NOT
    include the numerical prefactor."""
    return (mus ** 2 - masses ** 2)**(3 / 2) * mus - (masses ** 4) / 2 * np.log(np.sqrt(mus ** 2 / masses**2 - 1) + mus / masses) + (masses ** 2) / 2 * np.sqrt(mus ** 2 - masses ** 2) * mus


def pressure(mfs_and_mus, m_sigma_bar, omega_0):
    """Omega_0 is a function which takes mfs and m_sigma_bar as arguments, and returns the mesonic potential."""
    mfs, mu_us, mu_ds = mfs_and_mus[0, :], mfs_and_mus[1, :], mfs_and_mus[2, :]
    omega_0s = omega_0(mfs, m_sigma_bar)
    # omega_vacuum = omega_0s[0] # Investigate the difference between these choices?
    omega_vacuum = omega_0(1, m_sigma_bar)
    return omega_vacuum - omega_0s + N_c / (4 * np.pi ** 2) * chem_pot_term_pres(mu_us, g * mfs) \
        + N_c / (4 * np.pi ** 2) * chem_pot_term_pres(mu_ds, g * mfs) \
        + 1 / (4 * np.pi ** 2) * chem_pot_term_pres(mu_ds - mu_us, m_e_bar)


def energy_density(mfs_and_mus, m_sigma_bar, omega_0):
    """Returns the energy densities. omega_0 is the purely mesonic contribution (passed as either incr or cr)."""
    mfs, mu_us, mu_ds = mfs_and_mus[0, :], mfs_and_mus[1, :], mfs_and_mus[2, :]
    omega_0s = omega_0(mfs, m_sigma_bar)
    omega_vacuum = omega_0(1, m_sigma_bar)
    # Note that the omega_vacuum shift corresponds to B
    return - omega_vacuum + omega_0s + N_c / (4 * np.pi ** 2) * chem_pot_term_eps(mu_us, g * mfs) \
           + N_c / (4 * np.pi ** 2) * chem_pot_term_eps(mu_ds, g * mfs) \
           + 1 / (4 * np.pi ** 2) * chem_pot_term_eps(mu_ds - mu_us, m_e_bar)


def energy_density_no_electron(vevs_and_mus, m_sigma_bar, omega_0):
    """The energy density without the electron contribution."""
    vevs, mu_us, mu_ds = vevs_and_mus[0, :], vevs_and_mus[1, :], vevs_and_mus[2, :]
    omega_0s = omega_0(vevs, m_sigma_bar)
    omega_vacuum = omega_0(1, m_sigma_bar)
    return - omega_vacuum + omega_0s + N_c / (4 * np.pi ** 2) * chem_pot_term_eps(mu_us, g * vevs) \
           + N_c / (4 * np.pi ** 2) * chem_pot_term_eps(mu_ds, g * vevs)


def number_densites_quarks(mfs_and_mus):
    """Dimensionless number densities for the up and down quarks."""
    return np.array([N_c / (3 * np.pi ** 2) * (mfs_and_mus[1, :] ** 2 - g ** 2 * mfs_and_mus[0, :] ** 2) ** (3 / 2),
                     N_c / (3 * np.pi ** 2) * (mfs_and_mus[2, :] ** 2 - g ** 2 * mfs_and_mus[0, :] ** 2) ** (3 / 2)])


def number_densities_electrons(mfs_and_mus):
    """Dimensionless number density for the electrons."""
    mu_e = mfs_and_mus[2, :] - mfs_and_mus[1, :]
    return 1 / (3 * np.pi**2) * (mu_e - m_e_bar)**(3/2)


def maxwell_construct(ps, epss, debug_mode=False, return_indices=False):
    """Perform a maxwell construction.
    Note: ps and epss are sorted similarly, such that the far end of ps and epss are monotonically increasing:
    A maxwell construction is only needed whenever ps are increasing in the beginning.
    In practice, this is done with a np.flip() on mfs_and_mus."""
    diff_ps = np.gradient(ps)
    diff_ps_crosses_zero = []
    for n in range(1, len(diff_ps)):
        if diff_ps[n-1] * diff_ps[n] < 0:
            diff_ps_crosses_zero.append(n)
    # know that p_crit lies within [0, ps[n]]
    if debug_mode:
        print("dp = 0 at indices: {}".format(diff_ps_crosses_zero))
    if len(diff_ps_crosses_zero) == 0:
        # The pressure is solely increasing
        if return_indices:
            return ps, epss, []
        return ps, epss
    elif len(diff_ps_crosses_zero) == 1:
        # The pressure is decreasing and then increasing
        n = 1
        while ps[n] < 0:
            n += 1
        interpolate_zero = np.interp(0, ps[diff_ps_crosses_zero[0]:], epss[diff_ps_crosses_zero[0]:])
        # print(ps[n-1], interpolate_zero, ps[n])
        if return_indices:
            return np.concatenate((np.array([0.0]), ps[n:])), np.concatenate((np.array([interpolate_zero]), epss[n:])), [n - 1]
        return np.concatenate((np.array([0.0]), ps[n:])), np.concatenate((np.array([interpolate_zero]), epss[n:]))
    else:
        def gibbs_area(p_crit):
            assert p_crit <= ps[diff_ps_crosses_zero[0]]
            p_crit_index1, p_crit_index2 = np.argmin(np.abs(p_crit - ps[0: diff_ps_crosses_zero[0]])), np.argmin(np.abs(p_crit - ps[diff_ps_crosses_zero[1]:])) + diff_ps_crosses_zero[1]
            area1 = np.trapz(1/epss[p_crit_index1:diff_ps_crosses_zero[0]], ps[p_crit_index1:diff_ps_crosses_zero[0]])
            area2 = np.trapz(1/epss[diff_ps_crosses_zero[0]:diff_ps_crosses_zero[1]], ps[diff_ps_crosses_zero[0]:diff_ps_crosses_zero[1]])
            area3 = np.trapz(1/epss[diff_ps_crosses_zero[1]:p_crit_index2 + 1], ps[diff_ps_crosses_zero[1]:p_crit_index2 + 1])
            if debug_mode:
                print("pressures at p_crit: p_crit1: {}, p_crit2 {}".format(ps[p_crit_index1], ps[p_crit_index2]))
                print("area1: {}, area2: {}, area3: {}".format(area1, area2, area3))
            return area1 + area2 + area3, (p_crit_index1, p_crit_index2)

    # Finding p_crit is minimising gibbs area
    if debug_mode:
        print("Gibbs area p_crit = 0: {}".format(gibbs_area(0)))
    if gibbs_area(0)[0] < 0:
        # Just barely crossing zero in the beginning, cannot make proper numerical Maxwell construction
        # This corresponds to one index.
        n = diff_ps_crosses_zero[1]
        while ps[n] < 0:
            n += 1
        interpolate_zero = np.interp(0, ps[diff_ps_crosses_zero[1]:], epss[diff_ps_crosses_zero[1]:])
        if return_indices:
            return np.concatenate((np.array([0.0]), ps[n:])), np.concatenate((np.array([interpolate_zero]), epss[n:])), [n - 1]
        return np.concatenate((np.array([0.0]), ps[n:])), np.concatenate((np.array([interpolate_zero]), epss[n:]))

    solution = root_scalar(lambda p_crit: gibbs_area(p_crit)[0], method="bisect", bracket=(0, ps[diff_ps_crosses_zero[0]]))
    if debug_mode:
        print("Convergence: {}, solution: {}".format(solution.converged, solution.root))
        print("Lentght diff_ps: {}, indices for crossing zero: {}".format(len(diff_ps), diff_ps_crosses_zero))
        p_crit = solution.root
        gibbs_a, p_crit_indices = gibbs_area(p_crit)
        print("for p_crit = {} (indices: {})), the gibbs area is {}".format(p_crit, p_crit_indices, gibbs_a))
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(7, 3))
        ax.plot((p_crit, p_crit), (epss[p_crit_indices[0]], epss[p_crit_indices[1]]), color="black", linewidth=1)
        for index in diff_ps_crosses_zero:
            ax.scatter(ps[index], epss[index], color="black", s=10, zorder=2)
        cmap = cm.get_cmap("viridis")
        norm = colors.Normalize(vmin=0.5, vmax=4)
        num_to_col = cm.ScalarMappable(norm, cmap)
        ax.plot(ps[0: p_crit_indices[0] + 1], epss[0: p_crit_indices[0] + 1], color=num_to_col.to_rgba(1))
        ax.plot(ps[p_crit_indices[1]:], epss[p_crit_indices[1]:], color=num_to_col.to_rgba(1))
        ax.plot(ps[p_crit_indices[0]:diff_ps_crosses_zero[0] + 1], epss[p_crit_indices[0]:diff_ps_crosses_zero[0] + 1],
                color=num_to_col.to_rgba(2))
        ax.plot(ps[diff_ps_crosses_zero[0]:diff_ps_crosses_zero[1] + 1],
                epss[diff_ps_crosses_zero[0]: diff_ps_crosses_zero[1] + 1], color=num_to_col.to_rgba(3))
        ax.plot(ps[diff_ps_crosses_zero[1]:p_crit_indices[1] + 1],
                epss[diff_ps_crosses_zero[1]:p_crit_indices[1] + 1], color=num_to_col.to_rgba(4))
        ax.scatter((ps[p_crit_indices[0]], ps[p_crit_indices[1]]),
                   (epss[p_crit_indices[0]], epss[p_crit_indices[1]]), s=10, color="black")
        ax.set_xlim(min(ps)*1.05, max(2 * ps[diff_ps_crosses_zero[0]], 0.05))
        ax.set_ylim(0, epss[diff_ps_crosses_zero[-1]] * 1.7)
        ax.set_xlabel(r"$p \, / \, f_\pi^4$")
        ax.set_ylabel(r"$\epsilon \, / \, f_\pi^4$")
        ax.grid()
        ax2.grid()
        ax2.fill_between(ps[p_crit_indices[0]:diff_ps_crosses_zero[0] + 1], 0,
                         1/epss[p_crit_indices[0]:diff_ps_crosses_zero[0] + 1], color=num_to_col.to_rgba(2), zorder=1, alpha=0.5, label="Gibbs free energy, area 1")
        ax2.fill_between(ps[diff_ps_crosses_zero[0]:diff_ps_crosses_zero[1] + 1], 0,
                         1/epss[diff_ps_crosses_zero[0]:diff_ps_crosses_zero[1] + 1], color=num_to_col.to_rgba(3), zorder=2, alpha=0.5, label="Gibbs free energy, area 2")
        ax2.fill_between(ps[diff_ps_crosses_zero[1]:p_crit_indices[1] + 1], 0,
                         1/epss[diff_ps_crosses_zero[1]: p_crit_indices[1] + 1], color=num_to_col.to_rgba(4), zorder=3, alpha=0.5)
        ax2.set_ylim(0, 1/epss[int(diff_ps_crosses_zero[0] / 2)])
        ax2.plot((p_crit, p_crit), (0, 1/epss[p_crit_indices[0]]), color="black")
        ax2.set_xlabel(r"$p \, / \, f_\pi^4$")
        ax2.set_ylabel(r"$\frac{1}{\epsilon} \, / \, f_\pi^{-4}$")
        ax2.legend()
        plt.tight_layout(pad=0.5, w_pad=1.2)
        plt.show()
    assert solution.converged
    indices = gibbs_area(solution.root)[1]
    if return_indices:
        # This corresponds to two indices.
        if ps[indices[1]] < ps[indices[0]]:
            return np.concatenate((ps[0:indices[0] + 1], ps[indices[1] + 1:])), np.concatenate((epss[0:indices[0] + 1], epss[indices[1] + 1:])), [indices[0], indices[1] + 1]
        return np.concatenate((ps[0:indices[0] + 1], ps[indices[1]:])), np.concatenate((epss[0:indices[0] + 1], epss[indices[1]:])), indices
    if ps[indices[1]] < ps[indices[0]]:
        return np.concatenate((ps[0:indices[0] + 1], ps[indices[1] + 1:])), np.concatenate((epss[0:indices[0] + 1], epss[indices[1] + 1:]))
    return np.concatenate((ps[0:indices[0] + 1], ps[indices[1]:])), np.concatenate((epss[0:indices[0] + 1], epss[indices[1]:]))


def find_bag_window(mfs, m_sigma, bag_max, omega_0, omega_0_diff):
    """As discussed in the thesis, chapter 11.4.1, the strange matter hypothesis may force us to add a bag constant.
    This function finds the smallest bag constant we must add in order for the hypothesis to be fulfilled."""
    # Must find the bag window according to (eps_u + eps_d) / n_b < 931 MeV.
    energy_per_nucleon_bar = 931 / f_pi
    mfs_and_mus = np.flip(find_mus(mfs, m_sigma / f_pi, omega_0_diff), axis=1)
    mfs_and_mus_sep, mfs_and_mus_no_e = separate_mfs_and_mus(mfs_and_mus)
    # Need to find mu such that p(mu) = 0
    press_arr, eps_arr = pressure(mfs_and_mus_sep, m_sigma / f_pi, omega_0), energy_density_no_electron(mfs_and_mus_sep, m_sigma / f_pi, omega_0)
    n_u, n_d = number_densites_quarks(mfs_and_mus_sep)
    press_after_construct, eps_after_construct, indices = maxwell_construct(press_arr, eps_arr, return_indices=True)
    # Now we need to slice n_u and n_d correspondingly to how we sliced press_arr and eps_arr in
    # the maxwell construction. No construction: empty list. non-zero energy density at p = 0: one element in the list
    # Proper construction: two indices in index-list.
    if len(indices) == 0:
        print("No index (no Maxwell construction)")
        n_u_after_construct, n_d_after_construct = n_u, n_d
    elif len(indices) == 1:
        print("One index from Maxwell construction")
        n_u_after_construct, n_d_after_construct = n_u[indices[0]:], n_d[indices[0]:]
        assert len(n_u_after_construct) == len(press_after_construct), "Length of n_u: {}, length of ps: {}".format(len(n_u_after_construct), len(press_after_construct))
    else:
        print("Two indices from Maxwell construction")
        n_u_after_construct, n_d_after_construct = np.concatenate((n_u[0:indices[0] + 1], n_u[indices[1]:])), \
                                                   np.concatenate((n_d[0:indices[0] + 1], n_d[indices[1]:]))
        assert len(n_u_after_construct) == len(press_after_construct), "Length of n_u: {}, length of ps: {}".format(len(n_u_after_construct), len(press_after_construct))
        # indices[...] + 1 to INCLUDE the index in the list
    n_B = (n_u_after_construct + n_d_after_construct) / 3   # baryon number for one quark is 1 / 3.
    # 1 is suitable for m_sigma = 800. 4 is suitable for m_sigma = 600.
    n_bag = 20000     # Gives accuracy at least bag_max / n_bag
    bag_consts = np.linspace(0.0, bag_max, n_bag, endpoint=True)

    def energy_per_baryon(bag_const):
        shifted_press = press_after_construct - bag_const
        zero_press_index = np.argmin(np.abs(shifted_press))
        shifted_eps = eps_after_construct + bag_const
        return shifted_eps[zero_press_index] / n_B[zero_press_index], 1

    coord_list = []
    last_coord = (bag_consts[0], energy_per_baryon(bag_consts[0])[0] - energy_per_nucleon_bar)
    coord_list.append(last_coord)
    for bag in bag_consts[1:]:
        last_coord = (bag, energy_per_baryon(bag)[0] - energy_per_nucleon_bar)
        coord_list.append(last_coord)
    bags = [bag_const for n, bag_const in enumerate(bag_consts) if coord_list[n][1] > 0]
    return bags[0]


"""Now we may piece everything together to create an equation of state. We must choose m_sigma. 
We may also increase the bag constant."""


def EoS_QM_maxwell_construct(mfs, m_sigma, omega_0, omega_0_diff, bag_const=0, conversion_factor=1):
    """Return the QM EoS. Standard dimensionless units of p / f_pi^4"""
    # Conversion factor is added if we want to change units
    # Choose a sensible vev for a proper equation of state, i.e. low enough to reach high pressures, dense enough to be
    # properly smooth.
    mfs_and_mus = np.flip(find_mus(mfs, m_sigma / f_pi, omega_0_diff), axis=1)    # Flipping gives low mus first.
    mfs_and_mus_sep, vevs_and_mus_not_for_use = separate_mfs_and_mus(mfs_and_mus)    # Removing part where electrons
    # disappear i.e. where the solutions are unstable
    ps, epss = pressure(mfs_and_mus_sep, m_sigma / f_pi, omega_0), energy_density(mfs_and_mus_sep, m_sigma / f_pi,
                                                                                   omega_0)
    ps_maxwell, epss_maxwell = maxwell_construct(ps, epss, debug_mode=False)
    press_final, epss_final = (ps_maxwell - bag_const) * conversion_factor, \
                              (epss_maxwell + bag_const) * conversion_factor
    return interp1d(press_final, epss_final, kind="linear", bounds_error=True, assume_sorted=True)


def EoS_bar_min_bag_standard(m_sigma, cr=True, verbose=True):
    """Returns the EoS for the QM model. Standardised inputs.
    - Uses the minimum bag constant
    - Unless we change cr, it returns the consistently renormalised model."""
    vev_accurate = np.linspace(0.001, 0.9999, 5000)  # For finding bag const
    vevs = np.linspace(0.001, 0.9999, 1000)     # For EoS
    if not cr:
        omega_0, omega_0_diff = omega_0_incr, omega_0_diff_incr
    else:
        omega_0, omega_0_diff = omega_0_cr, omega_0_diff_cr
    bag_const = find_bag_window(vev_accurate, m_sigma, 3, omega_0, omega_0_diff)
    if verbose:
        print("Minimal bag constant {} for m_sigma {}".format(bag_const, m_sigma))
    conversion_factor = 1.56 * 10**33 / (1.646776 * 10 ** 36)  # 1.56 * 10**33 transforms to SI,
    # 1 / (1.646776 * 10 ** 36) transforms to dimensionless units used in TOV.
    return EoS_QM_maxwell_construct(vevs, m_sigma, omega_0, omega_0_diff, bag_const, conversion_factor)


def get_system_quantities_standard_cr(m_sigma, epsilon_left=0.01, epsilon_right=0.01, n_mf=1000, bag_extra=0.0):
    """Calculates mfs and mus, finds pressures and energy densities, performs maxwell construction and slices the other
    quantities accordingly.
    Optional parameters tune the initialisation of the mfs.
    returns equal length arrays of (mu_us, mu_ds), ns, ps, epss."""
    mfs = np.linspace(epsilon_left, 1 - epsilon_right, n_mf)
    m_sigma_bar = m_sigma / f_pi
    vevs_and_mus = np.flip(find_mus(mfs, m_sigma_bar, omega_0_diff_cr), axis=1)
    vnm, _ = separate_mfs_and_mus(vevs_and_mus)
    n_us_pre_construct, n_ds_pre_construct = number_densites_quarks(vnm)
    ps_pre_maxwell, epss_pre_maxwell = pressure(vnm, m_sigma_bar, omega_0_cr), energy_density(vnm, m_sigma_bar,
                                                                                              omega_0_cr)
    ps, epss, indices = maxwell_construct(ps_pre_maxwell, epss_pre_maxwell, return_indices=True)
    n_us, n_ds = arr_slice(n_us_pre_construct, indices), arr_slice(n_ds_pre_construct, indices)
    mu_us, mu_ds = arr_slice(vnm[1, :], indices), arr_slice(vnm[2, :], indices)
    ns = n_us + n_ds
    B = find_bag_window(np.linspace(0.02, 0.9999, 5000), m_sigma, 3, omega_0_cr, omega_0_diff_cr)
    print("shape of ps: {}, shape of epss: {}, indices: {}".format(ps.shape, epss.shape, indices))
    return (mu_us, mu_ds), ns, ps - B, epss + B


def arr_slice(arr, index_list):
    """Utility function. After a Maxwell construction, certain states are "unreachable".
    Want to slice away those states. Information about which elements should be removed, is given in a list indices."""
    if not index_list:
        return arr
    elif len(index_list) == 1:
        return arr[index_list[0]:]
    else:
        return np.concatenate((arr[0: index_list[0] + 1], arr[index_list[1]:]))


"""Calls to find the bag constant lower limit."""
# mfs = np.linspace(0.05, 0.9999, 5000)
# bag_600, bag_700, bag_800 = find_bag_window(mfs, 600, 4, omega_0_incr, omega_0_diff_incr), find_bag_window(mfs, 700, 1, omega_0_incr, omega_0_diff_incr), find_bag_window(mfs, 800, 0.1, omega_0_incr, omega_0_diff_incr)
# print("(incr) Lowest allowed bag value for m_sigma = 600: {} (dimesionless), {}^4 (dimensionful)".format(bag_600, (bag_600 * 93 ** 4) ** (1/4)))
# print("(incr) Lowest allowed bag value for m_sigma = 700: {} (dimesionless), {}^4 (dimensionful)".format(bag_700, (bag_700 * 93 ** 4) ** (1/4)))
# print("(incr) Lowest allowed bag value for m_sigma = 800: {} (dimesionless), {}^4 (dimensionful)".format(bag_800, (bag_800 * 93 ** 4) ** (1/4)))

# bag_400, bag_500, bag_600_cr = find_bag_window(mfs, 400, 4, omega_0_cr, omega_0_diff_cr), find_bag_window(mfs, 500, 1, omega_0_cr, omega_0_diff_cr), find_bag_window(mfs, 600, 0.1, omega_0_cr, omega_0_diff_cr)
# bag_550_cr = find_bag_window(mfs, 550, 1, omega_0_cr, omega_0_diff_cr)
# print("(cr) Lowest allowed bag value for m_sigma = 400: {} (dimesionless), {}^4 (dimensionful)".format(bag_400, (bag_400 * 93 ** 4) ** (1/4)))
# print("(cr) Lowest allowed bag value for m_sigma = 500: {} (dimesionless), {}^4 (dimensionful)".format(bag_500, (bag_500 * 93 ** 4) ** (1/4)))
# print("(cr) Lowest allowed bag value for m_sigma = 550: {} (dimesionless), {}^4 (dimensionful)".format(bag_550_cr, (bag_550_cr * 93 ** 4) ** (1/4)))
# print("(cr) Lowest allowed bag value for m_sigma = 600: {} (dimesionless), {}^4 (dimensionful)".format(bag_600_cr, (bag_600_cr * 93 ** 4) ** (1/4)))

