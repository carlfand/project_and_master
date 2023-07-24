import numpy as np
import numba
from TOV import integrate, p_analytic, epsilon_analytic

"""Here, we investigate the stability of neutron stars numerically. The theory behind this file is found in the thesis
in chapter 6.2. Here, we consider the ideal neutron stars, meaning that everything is expressed in terms of x_F.

The calculations we perform here are necessary for the plots in chapter 6.3."""


@numba.njit
def alpha(r_hat, p_bar_diff_hat, p_bar, eps_bar, M_bar, Kappa_2_hat):
    """Returns alpha, or equvalently exp(2 nu)."""
    nu_arr = np.zeros_like(r_hat)
    stepsize = r_hat[1] - r_hat[0]
    for n in range(1, len(r_hat)):
        nu_arr[n] = nu_arr[n - 1] + stepsize / 2 * \
                    (-p_bar_diff_hat[n] / (eps_bar[n] + p_bar[n]) -
                     p_bar_diff_hat[n - 1] / (eps_bar[n - 1] +
                                              p_bar[n - 1]))
    nu_arr = nu_arr - nu_arr[-1] + 1 / 2 * np.log(1 - 2 * M_bar[-1]
                                                  * Kappa_2_hat/r_hat[-1])
    return np.exp(2 * nu_arr)


@numba.njit
def beta(r_hat, M_bar, kappa_2_hat):
    """Finds beta or equivalently exp(2 lambda_0)"""
    beta_arr = np.zeros_like(r_hat)
    # we set beta(r=0) = 1 manually to avoid "division by zero"-problems
    beta_arr[0] = 1
    beta_arr[1:] = 1 / (1 - 2 * M_bar[1:] * kappa_2_hat / r_hat[1:])
    return beta_arr


@numba.njit
def gamma(x_F):
    """Returns the 0th order adiabatic index for the ideal neutron star.
    d p/d epsilon = x_F**2 /(3(1 + x_F^2))"""
    return (epsilon_analytic(x_F) + p_analytic(x_F))/p_analytic(x_F) * \
           1/3 * x_F ** 2 / (1 + x_F**2)


@numba.njit
def u_coefficients(r, x_F, p_bar, p_bar_diff, eps_bar, alphas, betas,
                   omega_bar, kappa_bar):
    # u_coeffs[0, :] are for u'.  u_coeffs[1, :] are for u.
    u_coeffs = np.zeros((2, len(r)))
    # Cutting off the first point to avoid division by zero error
    u_coeffs[0, 1:] = (-p_bar_diff[1:] / (eps_bar[1:] + p_bar[1:]) *
                       (4 + 5/x_F[1:]**2) + 3 / r[1:] - betas[1:]/r[1:]
                       - 4 * np.pi * kappa_bar * r[1:] *
                       (3 * p_bar[1:] + eps_bar[1:]) * betas[1:])

    u_coeffs[1, 1:] = (- 3 * (1 + 1/x_F[1:]**2) *
                       (p_bar_diff[1:]**2/((eps_bar[1:]+p_bar[1:])**2) -
                        4 * p_bar_diff[1:]/(r[1:] * (eps_bar[1:]
                                                     + p_bar[1:])) -
                        8 * np.pi * kappa_bar * betas[1:] * p_bar[1:] +
                        omega_bar * betas[1:] / alphas[1:]))
    return u_coeffs


@numba.njit
def find_nodes(omega_trial, r_hat, kappa_hat, kappa_2_hat, x_F,
               p_bar_diff_hat, M_bar):
    # Hard coded parameters:
    # When u has become larger than this, we assume it has diverged
    max_u = 50000
    pad1, pad2 = 0.001, 0.001
    p_bar, eps_bar = p_analytic(x_F), epsilon_analytic(x_F)
    alphas = alpha(r_hat, p_bar_diff_hat, p_bar, eps_bar,
                   M_bar, kappa_2_hat)
    betas = beta(r_hat, M_bar, kappa_2_hat)
    # Getting coefficients
    u_coeffs = u_coefficients(r_hat, x_F, p_bar, p_bar_diff_hat,
                              eps_bar, alphas, betas, omega_trial,
                              kappa_hat)

    # Initialising arrays with u
    u = np.zeros_like(r_hat)        # u
    u_diff = np.zeros_like(r_hat)   # u'
    u_curve = np.zeros_like(r_hat)  # u''

    step = r_hat[1] - r_hat[0]  # Equidistant r

    # Shooting once
    # At first, u = r**3
    n = 0
    while n * step < pad1 * r_hat[-1]:
        u[n] = (n * step) ** 3
        u_diff[n] = 3 * (n * step) ** 2
        u_curve[n] = 6 * n * step
        n += 1

    # Entering area when we integrate normally
    for i in [j for j in range(n, len(r_hat))
              if j * step < (1 - pad2) * r_hat[-1]]:
        u[i] = u[i - 1] + u_diff[i - 1] * step + \
               (step ** 2) / 2 * u_curve[i - 1]
        u_diff[i] = u_diff[i - 1] + step * u_curve[i - 1]
        u_curve[i] = u_coeffs[0, i - 1] * u_diff[i - 1] + \
                     u_coeffs[1, i - 1] * u[i - 1]
        if u[i] > max_u:
            break

    # At last, counting nodes.
    nodes = 0
    for k in range(1, len(u)):
        if u[k] * u[k - 1] < 0:
            nodes += 1

    return nodes


def eigenfreq(n, x_F, p_bar_diff, M_bar, step_SI, accuracy):
    """Find the n-th eigenfrequency squared for a given
    star-configuration."""
    # The usual constants:
    eps_g, c, G = 1.646776 * 10 ** 36, 3 * 10 ** 8, 6.674 * 10 ** (-11)
    # Setting a new natural length scale r_0
    r_0 = 10 ** 4
    # Defining constants for use in
    kappa_hat, kappa_2_hat = G * eps_g / c ** 4 * r_0 ** 2, \
                             G * eps_g / (c ** 2 * r_0)

    p_bar_diff_hat = p_bar_diff * r_0

    r = np.array([n * step_SI for n in range(0, len(x_F))])
    r_hat = r / r_0

    # At first, we must find two limiting omegas.
    omega_upper = 1
    omega_lower = -1
    while find_nodes(omega_upper, r_hat, kappa_hat,
                     kappa_2_hat, x_F, p_bar_diff_hat, M_bar) <= n:
        omega_upper *= 2
    while find_nodes(omega_lower, r_hat, kappa_hat,
                     kappa_2_hat, x_F, p_bar_diff_hat, M_bar) > n:
        omega_lower *= 2

    # Run until the error is less than accuracy
    while omega_upper - omega_lower > accuracy:
        omega_trial = (omega_upper + omega_lower) / 2
        nodes_trial = find_nodes(omega_trial, r_hat, kappa_hat,
                                 kappa_2_hat, x_F, p_bar_diff_hat, M_bar)
        if nodes_trial <= n:
            omega_lower = omega_trial
        else:
            omega_upper = omega_trial

    # Choose the middle value as our best guess for omega^2
    omega_guess = (omega_upper + omega_lower) / 2
    return omega_guess


def find_modes(n_modes, n_p_c, p_c_min, p_c_max, stepsize=0.5):
    c, G = 3.0 * 10 ** 8, 6.674 * 10 ** (-11)
    eps_g = 1.646776 * 10 ** 36
    solar_mass = 1.989 * 10 ** 30
    bisect_err = 10**(-8)
    omega_err = 10 ** (-4)
    r_0 = 10000

    # Making logspace with n_p_c elements in [p_c_min, p_c_max]
    p_cs = np.array([10 ** pwr for pwr in
                     np.linspace(np.log10(float(p_c_min)),
                                 np.log10(float(p_c_max)), n_p_c)])
    # Initialising an array of tuples (p_c, omega_1, ... omega_n, M, R)
    return_arr = np.zeros((n_p_c, n_modes + 3))
    for i, p_c in enumerate(p_cs):
        return_arr[i, 0] = p_c
        x_F, x_F_diff, M_bar, R = integrate(p_c / eps_g, eps_g, c, G,
                                            stepsize, bisect_err)
        p_bar_diff = x_F ** 4 / (3 * (np.sqrt(1 + x_F**2))) * x_F_diff
        for n in range(n_modes):
            return_arr[i, 1 + n] = eigenfreq(n, x_F, p_bar_diff, M_bar,
                                             stepsize, omega_err)
        return_arr[i, n_modes + 1] = M_bar[-1] * eps_g / solar_mass
        return_arr[i, n_modes + 2] = R / r_0
    return return_arr


def critical_p_c_for_nth_freq(n, p_c_min, p_c_max, stepsize, log_p_c_rel_error):
    """The goal is to find the critical pressure where the n-th mode becomes unstable. This is done via the bisection
    method of the interval p_c_min and p_c_max."""
    # The usual constants:
    lightspeed, grav_const = 3.0 * 10 ** 8, 6.674 * 10 ** (-11)
    eps_g_neutron = 1.646776 * 10 ** 36
    bisect_err = 10 ** (-8)
    omega_err = 10 ** (-4)

    p_c_small, p_c_large = p_c_min, p_c_max

    x_F1, x_F_diff1, M_bar1, R = integrate(p_c_small / eps_g_neutron, eps_g_neutron, lightspeed, grav_const, stepsize,
                                           bisect_err)
    omega_large = eigenfreq(n, x_F1, x_F1 ** 4 / (3 * (np.sqrt(1 + x_F1**2))) * x_F_diff1, M_bar1, stepsize, omega_err)
    x_F2, x_F_diff2, M_bar2, R2 = integrate(p_c_large / eps_g_neutron, eps_g_neutron, lightspeed, grav_const, stepsize,
                                            bisect_err)
    omega_small = eigenfreq(n, x_F2, x_F2 ** 4 / (3 * (np.sqrt(1 + x_F2**2))) * x_F_diff2, M_bar2, stepsize, omega_err)
    if omega_large * omega_small > 0:
        raise ValueError("Omega(p_c_min) * omega(p_c_max) > 0. Choose different p_c_min and/or p_c_max")
    while (np.log10(p_c_large) - np.log10(p_c_small))/np.log10(p_c_small) > log_p_c_rel_error and \
            omega_large > 2 * omega_err:
        # bisecting [log(p_c_small), log(p_c_large)]
        p_c_log_avg = 10 ** ((np.log10(p_c_small) + np.log10(p_c_large))/2)
        x_F, x_F_diff, M_bar, R = integrate(p_c_log_avg / eps_g_neutron, eps_g_neutron, lightspeed, grav_const, stepsize,
                                            bisect_err)
        omega_middle = eigenfreq(n, x_F, x_F ** 4 / (3 * (np.sqrt(1 + x_F**2))) * x_F_diff, M_bar, stepsize,
                                 omega_err)
        if omega_middle * omega_large < 0:
            p_c_large = p_c_log_avg
        else:
            omega_large = omega_middle
            p_c_small = p_c_log_avg

    p_c_return = 10 ** ((np.log10(p_c_small) + np.log10(p_c_large))/2)
    print(omega_large, p_c_return)
    return p_c_return
