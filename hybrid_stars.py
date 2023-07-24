import numpy as np
from scipy.interpolate import interp1d
from QM_model import get_system_quantities_standard_cr
from APR_equation_of_state import nuclear_matter_shifted_neutron_mass, nuclear_matter_mu_B_n_B_p_eps

"""In this file, we include the functions we need to perform calculations upon hybrid and unified hybrid 
star models.

We begin with the hybrid model."""


def find_mu_B_and_min_B_for_crossing(ps_q, mu_Bq, ps_nuc, mu_B_nuc):
    """Takes in p_q(mu_B) and p_nuc(mu_B) to return mu_B where the curves cross. Also return the change in bag shift B
    that is required for such a crossing to take place.
    ps_q and mu_Bq are arrays where each element corresponds to each other, and similarly for ps_nuc and mu_B_nuc.
    Note that ps_q and ps_nuc need not be of equal length.
    Assumes that ps_q will grow larger than ps_nuc for large values of mu_B.
    NOTE: We do not want the crossing to happen at mu_B = m_n. Chopping away mu_B / f_pi < 10.2 yields the desired
    result."""
    assert ps_q[-1] > ps_nuc[-1], "quark phase must grow largest for large mu_B"
    lower_mu_B_cutoff = 10.2
    ps_nuc_truncated = np.array([p_nuc for (p_nuc, mu_B) in zip(ps_nuc, mu_B_nuc) if mu_B >= lower_mu_B_cutoff])
    mu_B_nuc_trunc = np.array([mu_B for mu_B in mu_B_nuc if mu_B >= lower_mu_B_cutoff])
    # Make a function of ps_nuc, such that we can find its values on the same mu_Bs as the ones in mu_Bq
    ps_nuc_func = interp1d(mu_B_nuc_trunc, ps_nuc_truncated, kind="linear", bounds_error=True, assume_sorted=True)
    min_mu_B_nuc, max_mu_B_nuc = min(mu_B_nuc_trunc), max(mu_B_nuc_trunc)
    mu_B_trunc = np.array([mu_B for mu_B in mu_Bq if min_mu_B_nuc <= mu_B <= max_mu_B_nuc])
    ps_q_trunc = np.array([p_q for (p_q, mu_B) in zip(ps_q, mu_Bq) if min_mu_B_nuc <= mu_B <= max_mu_B_nuc])
    ps_nuc_new = ps_nuc_func(mu_B_trunc)
    min_difference, n_min_difference = min(ps_q_trunc - ps_nuc_new), np.argmin((ps_q_trunc - ps_nuc_new))
    # For which mu_B does this occur?
    n_max = len(mu_B_trunc)
    if min_difference < 0:
        for i in range(1, n_max + 1):
            if ps_q_trunc[n_max - i] < ps_nuc_new[n_max - i]:
                # Interpolate to get better accuracy
                p1q, p2q = ps_q_trunc[n_max - i], ps_q_trunc[n_max - i + 1]
                p1nuc, p2nuc = ps_nuc_new[n_max - i], ps_nuc_new[n_max - i + 1]
                mu1, mu2 = mu_B_trunc[n_max - i], mu_B_trunc[n_max - i + 1]
                mu_B_crit = crossing_interpolation(mu1, mu2, p1nuc, p2nuc, p1q, p2q)
                p_crit = np.interp(mu_B_crit, mu_B_trunc, ps_nuc_new)
                return mu_B_crit, p_crit, 0
    else:
        # Not useful for m_n = 900
        return mu_B_trunc[n_min_difference], ps_nuc_new[n_min_difference], min_difference


def find_critical_mu_B(ps_q, mu_Bq, ps_nuc, mu_B_nuc):
    """Finds the mu_B where the transition from nuclear to quark equation of state happens."""
    mu_B_crit, p_crit, min_B_for_crossing = find_mu_B_and_min_B_for_crossing(ps_q, mu_Bq, ps_nuc, mu_B_nuc)
    print("Addition to bag constant needed to get crossing: {}".format(min_B_for_crossing))
    print("The critical crossing happens at mu_B = {} (dimensionless)".format(mu_B_crit))
    return mu_B_crit, p_crit, min_B_for_crossing


def EoS_hybrid(ps_q, eps_q, ps_nuc, eps_nuc, p_crit, verbose=False):
    """Takes arrays of pressures and energy densities as input. Returns a function which patches together the two."""
    avg_stepsize_q = sum(ps_q[n] - ps_q[n-1] for n in range(1, len(ps_q)))/(len(ps_q) - 1)
    delta = 0.00000001 * avg_stepsize_q  # Introduce a very small number to help make a well-defined function
    eps_crit_nuc = np.interp(p_crit, ps_nuc, eps_nuc)
    eps_crit_q = np.interp(p_crit, ps_q, eps_q)
    ps_hybrid = np.concatenate(([p_nuc for p_nuc in ps_nuc if p_nuc <= p_crit - delta],
                                [p_crit - delta, p_crit + delta],
                                [p_q for p_q in ps_q if p_q >= p_crit + delta]))
    epss_hybrid = np.concatenate(([eps_nuc for (p_nuc, eps_nuc) in zip(ps_nuc, eps_nuc) if p_nuc <= p_crit - delta],
                                 [eps_crit_nuc, eps_crit_q],
                                 [eps_q for (p_q, eps_q) in zip(ps_q, eps_q) if p_q >= p_crit + delta]))
    # In the numerical integration, we will reach pressures barely below zero. To avoid interpolation bounds error,
    # we must extend the ps and eps slightly. With a large stepsize, one might reach "more negative" values.
    # NOTE, IMPORTANT: Interpolation error (ValueError) might stem from the fact that the negative extension here is
    # too small.
    ps_extend, eps_extend = np.array([-ps_hybrid[100]]), np.array([np.interp(-ps_hybrid[100], ps_hybrid, epss_hybrid)])
    # Interpolation range problems stopped by choosing as above for stepsize = 1 (corresponds to metre)
    if verbose:
        print("EoS for p in [{}, {}]".format(ps_extend[0], max(ps_hybrid)))
    return interp1d(np.concatenate((ps_extend, ps_hybrid)), np.concatenate((eps_extend, epss_hybrid)), kind="linear",
                    bounds_error=True, assume_sorted=True)


def crossing_interpolation(x1, x2, y1, y2, y1_mark, y2_mark):
    """Utitlity funtion to improve accuracy of finding the critical mu_B and pressure.
    Let f be a linear function running through (x1, y1) and (x2, y2), and similarly for f_mark with (x1, y1_mark) and
    (x2, y2_mark).
    We write f(u) = y1 + (y2 - y1)/(x2 - x1)u = y1 + delta_y / delta_x * u, u in [0, x2 - x1].
    Assuming that f and f_mark cross, we may write the solution from f(u_c) = f_mark(u_c), yielding
    u_c = delta_x * (y1_mark - y1) / (delta_y - delta_y_mark). Returns x1 + u_c"""
    delta_x = x2 - x1
    delta_y = y2 - y1
    delta_y_mark = y2_mark - y1_mark
    x_c = x1 + delta_x * (y1_mark - y1) / (delta_y - delta_y_mark)
    assert x1 <= x_c <= x2, "No solution within interval x1 = {}, x2 = {}".format(x1, x2)
    return x_c


def EoS_hybrid_standard(m_sigma, bag_const_add=0.0, conversion_factor=1, n_vev=1000):
    """Returns the hybrid equation of state with few inputs. Conversion factor is added to allow for
    a free choice of units."""
    (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002, n_mf=n_vev,
                                                                     bag_extra=bag_const_add)
    mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_shifted_neutron_mass()
    mu_B_crit, p_crit, min_bag_add = find_critical_mu_B(ps, 3 / 2 * (mu_us + mu_ds), ps_nuc, mu_Bs_nuc)
    # added factor 3 to convert mu -> mu_B
    if min_bag_add:
        print("An addition to the bag constant has been made in order to have a crossing. Addition: {}".format(min_bag_add))
    ps -= min_bag_add
    epss += min_bag_add
    return EoS_hybrid(ps * conversion_factor, epss * conversion_factor, ps_nuc * conversion_factor,
                      epss_nuc * conversion_factor, p_crit * conversion_factor, verbose=False)


"""The methods for creating a unified equation of state follows. Several polynomial unificators were attempted, however,
due to their limited success in comparison to unific_3_3_free_midpoint(), we do not include them here."""


def causality_condition(xs, f_diffs, f_double_diffs):
    """Returns an array of the causality condition function.
    Takes arrays and x_1, a single value."""
    return xs / f_diffs * f_double_diffs - 1


def unific_3_3_free_midpoint(x_1, f_1, f_1_diff, x_2, f_2, f_2_diff):
    """One way of making a physically reasonable interpolating function.
        We use two 3rd degree polynomials.
        The meeting point for the polynomials is chosen as an
        external paramter, m (x-value).
        In addition, we may tune the second derivatives at the
        endpoints to match the external ones.
        p(x) = a + b x + c x^2 + d x^3, x in [0, (x_2 - x_1)/2],
        q(u) = e + f u + g u^2 + h u^3, u in [m - (x_2 - x_1), 0]
        x = 0 corresponds to x_1. x and u are chosen a linear shifts of x
        to make the coefficient determination easier.
        delta_x (dx) is the length of the interval over which
        we use the interpolating function.
        Continuous derivative. This yields the conditions
        p(0) = f_1, p'(0) = f_1_diff, p''(0) = f_1_double_diff
        q(0) = f_2, q'(0) = f_2_diff.
        p(m) = q(m) and p'(m) = q'(m) = n.
        These restrains fully determine the polynomials after we choose m and n."""
    dx = x_2 - x_1
    a, b = f_1, f_1_diff
    e, f = f_2, f_2_diff

    def c_d_g_h(m_unshifted, n, f_1_double_diff):
        m = m_unshifted - x_1
        c = f_1_double_diff / 2
        d = 1 / (3 * m**2) * (n - b - 2 * c * m)
        h = 2 / (dx - m)**3 * (a + b * m + c * m**2 + d * m**3
                               - e - 1 / 2 * f * (m - dx) - 1 / 2 * n * (m - dx))
        g = 1 / (2 * (m - dx)) * (n - f - 3 * h * (m - dx)**2)
        return c, d, g, h

    def interpolate(x, m, n, f_1_double_diff=1.0):
        c, d, g, h = c_d_g_h(m, n, f_1_double_diff)
        return np.where(x - x_1 < m - x_1, a + b * (x - x_1) + c * (x - x_1)**2 + d * (x - x_1)**3,
                        e + f * (x - x_2) + g * (x - x_2)**2 + h * (x - x_2)**3)

    def interpolate_diff(x, m, n, f_1_double_diff=1.0):
        c, d, g, h = c_d_g_h(m, n, f_1_double_diff)
        return np.where(x - x_1 < m - x_1, b + 2 * c * (x - x_1) + 3 * d * (x - x_1) ** 2,
                        f + 2 * g * (x - x_2) + 3 * h * (x - x_2) ** 2)

    def interpolate_double_diff(x, m, n, f_1_double_diff=1.0):
        c, d, g, h = c_d_g_h(m, n, f_1_double_diff)
        return np.where(x - x_1 < m - x_1, 2 * c + 6 * d * (x - x_1), 2 * g + 6 * h * (x - x_2))
    return interpolate, interpolate_diff, interpolate_double_diff


def find_best_polynomial_params(unific, distrust_APR_mu_p_n_n_diff, distrust_quark_mu_p_n_n_diff, i_mu, i_n,
                                search_interval_mu_rel=(0, 1), search_interval_n_rel=(0, 1)):
    """distrust_APR_mu_p_n_n_diff is a length 4-list [mu, p, n, n_diff] of the parameters we need to fit and judge how
    well a polynomial does the trick at interpolating. The same goes for distrust_quark_mu_p_n_n_diff.
    This function searches through i_mu * i_n discrete sets of (mu, n) and determines which one yields
    the best polynomial. The search_interval-parameters delimits the ranges we search in. (0, 1) corresponds
    to the full search area.
    For m_sigma = 600 MeV, the only solutions lie in the lower left corner of the (mu, n)-rectangle."""
    APR_mu, APR_p, APR_n, APR_n_diff = distrust_APR_mu_p_n_n_diff   # Unpacking. APR_n_diff might be shifted if we do
    # not require continuous second derivative.
    q_mu, q_p, q_n, q_n_diff = distrust_quark_mu_p_n_n_diff
    delta_mu, delta_n = q_mu - APR_mu, q_n - APR_n
    search_mus = np.linspace(APR_mu + search_interval_mu_rel[0] * delta_mu, q_mu - search_interval_mu_rel[1] * delta_mu,
                             i_mu, endpoint=True)
    search_ns = np.linspace(APR_n + search_interval_n_rel[0] * delta_n, q_n - search_interval_n_rel[1] * delta_n, i_n,
                            endpoint=True)
    unifier, unifier_diff, unifier_diff_diff = unific(APR_mu, APR_p, APR_n, q_mu, q_p, q_n)
    mu_intermittent = np.linspace(APR_mu, q_mu, 300, endpoint=True)    # x-axis
    kinkedness = np.infty   # Initialise comparing number
    best_mu_n = None        # If no suitable (mu, n) is found, return None
    for mu in search_mus:
        for n in search_ns:
            # Generate array of n(mu)
            arr_diff = unifier_diff(mu_intermittent, mu, n, APR_n_diff)
            arr_diff_diff = unifier_diff_diff(mu_intermittent, mu, n, APR_n_diff)
            causality_cond = causality_condition(mu_intermittent, arr_diff, arr_diff_diff)
            if min(causality_cond) > 0:
                # The next may sound like a lot, so let's break it down. It is an array containing how the
                # double derivative acting upon p(mu) changes. A large change happens at a kink of n(mu). We wish to
                # choose an interpolating polynomial with little to no kinks in n.
                arr_diff_diff_differences = np.array(
                    [arr_diff_diff[i] - arr_diff_diff[i - 1] for i in range(1, len(arr_diff_diff))])
                kink = max(np.abs(arr_diff_diff_differences))   # finding maximal internal "kink". If there are no kinks
                # We find where the curvature of n is the steepest.
                end_kink = abs(arr_diff_diff[-1] - q_n_diff)  # Measure kink of graph at the end
                start_kink = abs(arr_diff_diff[0] - APR_n_diff)     # Measure kink of graph at the beginning
                # Another possibility: end_kink + kink + start_kink
                if np.sqrt(end_kink**2 + kink**2 + start_kink**2) < kinkedness:
                    kinkedness = np.sqrt(end_kink**2 + kink**2 + start_kink**2)
                    best_mu_n = (mu, n)
    print("For search interval of mu: {} MeV with n_mu steps: {} and interval of n: {} fm^(-3) with n_n steps: {},\n"
          "the best (mu [MeV], n [fm^(-3)]) was found to be "
          "{}".format([(APR_mu + delta_mu * search_interval_mu_rel[0]) * 93,
                       (q_mu - delta_mu * search_interval_mu_rel[1]) * 93],
                      i_mu, [(APR_n + delta_n * search_interval_n_rel[0]) * 0.1047,
                             (q_n - delta_n * search_interval_n_rel[1]) * 0.1047],
                      i_n, (best_mu_n[0] * 93, best_mu_n[1] * 0.1047)))
    return best_mu_n, [unifier, unifier_diff, unifier_diff_diff]


def EoS_unified_standard_3_3(m_sigma, bag_const_add=0.0, conversion_factor=1, n_vev=1000):
    """Returns equation of state using the unific_3_3_free_midpoint. We take the neutron mass m_n = 900 MeV."""
    n_0 = 0.16  # units of fm^(-3). Remember: Conversion factor to f_pi^(-3): 1 / 0.1047
    # Need to choose where to start distrusting APR and quark model. Standard: 2 * n_0 and 4 * n_0
    limit_APR_n, limit_quark_n = 2, 4   # These are the standard delimiting n in units of n_0
    # When we fix the unifying polynomial, we search for the best paramter set (mu, n)
    m_sigmas_and_mu_search = {0: (30, [0.05, 0.05]), 600: (30, [0.02, 0.7], )}    # If m_sigma = 600,
    m_sigmas_and_n_search = {0: (30, [0.05, 0.05]), 600: (30, [0.02, 0.7])}     # narrow the search area for (mu, n)
    m_sigmas_and_n_diff_APR_shift = {0: 0, 600: 0.8}

    # Handling the nuclear matter
    mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_shifted_neutron_mass()
    distrust_APR_n = limit_APR_n * n_0 / 0.1047
    distrust_APR_mu = np.interp(distrust_APR_n, n_Bs_nuc, mu_Bs_nuc)
    distrust_APR_p = np.interp(distrust_APR_mu, mu_Bs_nuc, ps_nuc)
    distrust_APR_eps = np.interp(distrust_APR_mu, mu_Bs_nuc, epss_nuc)
    n_diff_mu = np.gradient(n_Bs_nuc, mu_Bs_nuc)
    n_APR_diff_mu_at_distrust = np.interp(distrust_APR_mu, mu_Bs_nuc, n_diff_mu)
    if m_sigma in m_sigmas_and_n_diff_APR_shift.keys():
        shift = m_sigmas_and_n_diff_APR_shift[m_sigma]
    else:
        shift = m_sigmas_and_n_diff_APR_shift[0]
    APR_limit_params = [distrust_APR_mu, distrust_APR_p, distrust_APR_n, n_APR_diff_mu_at_distrust - shift]

    ps_nuc_trunc = np.array([p_nuc for p_nuc in ps_nuc if p_nuc < distrust_APR_p] + [distrust_APR_p])
    epss_nuc_trunc = np.array([eps_nuc for eps_nuc in epss_nuc if eps_nuc < distrust_APR_eps] + [distrust_APR_eps])

    # Quark matter handling:
    (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002, n_mf=n_vev,
                                                                     bag_extra=bag_const_add)
    distrust_quark_n = limit_quark_n * n_0 / 0.1047
    n_Bs_q = 1 / 3 * ns  # Looking at baryonic number density, not total quark density
    mu_Bs_q = 3 / 2 * (mu_us + mu_ds)   # Baryonic chemical potential
    distrust_quark_mu = np.interp(distrust_quark_n, n_Bs_q, mu_Bs_q)
    distrust_quark_p = np.interp(distrust_quark_mu, mu_Bs_q, ps)
    distrust_quark_eps = np.interp(distrust_quark_mu, mu_Bs_q, epss)
    n_q_diff_mu_at_distrust = np.interp(distrust_quark_mu, mu_Bs_q, np.gradient(n_Bs_q, mu_Bs_q))
    quark_limit_params = [distrust_quark_mu, distrust_quark_p, distrust_quark_n, n_q_diff_mu_at_distrust]

    ps_q_trunc = np.array([distrust_quark_p] + [p_q for p_q in ps if p_q > distrust_quark_p])
    epss_q_trunc = np.array([distrust_quark_eps] + [eps_q for eps_q in epss if eps_q > distrust_quark_eps])

    if m_sigma in m_sigmas_and_mu_search.keys():
        i_mu, mu_search = m_sigmas_and_mu_search[m_sigma]
        i_n, n_search = m_sigmas_and_n_search[m_sigma]
    else:
        i_mu, mu_search = m_sigmas_and_mu_search[0]
        i_n, n_search = m_sigmas_and_n_search[0]
    (m, n), (unific, unific_diff, unific_diff_diff) = find_best_polynomial_params(unific_3_3_free_midpoint, APR_limit_params, quark_limit_params,
                                                                                  i_mu, i_n, mu_search, n_search)
    epsilon = 0.0002  # Adding this so there is no overlap in mu-arrays, and thus not in p-arrays
    mu_intermittent = np.linspace(distrust_APR_mu + epsilon, distrust_quark_mu + epsilon, 100, endpoint=True)
    ps_unific = unific(mu_intermittent, m, n, n_APR_diff_mu_at_distrust)
    ns_unific = unific_diff(mu_intermittent, m, n, n_APR_diff_mu_at_distrust)
    epss_unific = - ps_unific + mu_intermittent * ns_unific
    n_negative = 15
    ps_total = np.concatenate((np.array([-ps_nuc_trunc[n_negative]]), ps_nuc_trunc, ps_unific, ps_q_trunc))
    epss_total = np.concatenate((np.array([-epss_nuc_trunc[n_negative]]), epss_nuc_trunc, epss_unific, epss_q_trunc))
    # print("EoS unified allows p={} (dimensionless converted units)".format(-ps_nuc_trunc[n_negative]))
    return interp1d(ps_total * conversion_factor, epss_total * conversion_factor, kind="linear", bounds_error=True,
                    assume_sorted=True)


def EoS_unified_3_3_standard_unshifted(m_sigma, limit_APR_n=2, limit_quark_n=6, bag_const_add=0.0, conversion_factor=1,
                                       n_mf=1000):
    """Returns equation of state using the unific_3_3_free_midpoint. We take the neutron mass m_n = 939.6 MeV,
    the value from the APR files. """
    n_0 = 0.16  # units of fm^(-3). Remember: Conversion factor to f_pi^(-3): 1 / 0.1047
    # Need to choose where to start distrusting APR and quark model. Standard for unshifted m_n:
    # 2 * n_0 and 5 * n_0
    # Now, specifying search parameters (mu, n) is unnecessary

    # Handling the nuclear matter
    mu_Bs_nuc, n_Bs_nuc, ps_nuc, epss_nuc = nuclear_matter_mu_B_n_B_p_eps()
    distrust_APR_n = limit_APR_n * n_0 / 0.1047
    distrust_APR_mu = np.interp(distrust_APR_n, n_Bs_nuc, mu_Bs_nuc)
    distrust_APR_p = np.interp(distrust_APR_mu, mu_Bs_nuc, ps_nuc)
    distrust_APR_eps = np.interp(distrust_APR_mu, mu_Bs_nuc, epss_nuc)
    n_diff_mu = np.gradient(n_Bs_nuc, mu_Bs_nuc)
    n_APR_diff_mu_at_distrust = np.interp(distrust_APR_mu, mu_Bs_nuc, n_diff_mu)

    APR_limit_params = [distrust_APR_mu, distrust_APR_p, distrust_APR_n, n_APR_diff_mu_at_distrust]

    ps_nuc_trunc = np.array([p_nuc for p_nuc in ps_nuc if p_nuc < distrust_APR_p] + [distrust_APR_p])
    epss_nuc_trunc = np.array([eps_nuc for eps_nuc in epss_nuc if eps_nuc < distrust_APR_eps] + [distrust_APR_eps])

    # Quark matter handling:
    (mu_us, mu_ds), ns, ps, epss = get_system_quantities_standard_cr(m_sigma, epsilon_left=0.002, n_mf=n_mf,
                                                                     bag_extra=bag_const_add)
    distrust_quark_n = limit_quark_n * n_0 / 0.1047
    n_Bs_q = 1 / 3 * ns  # Looking at baryonic number density, not total quark density
    mu_Bs_q = 3 / 2 * (mu_us + mu_ds)   # Baryonic chemical potential
    distrust_quark_mu = np.interp(distrust_quark_n, n_Bs_q, mu_Bs_q)
    distrust_quark_p = np.interp(distrust_quark_mu, mu_Bs_q, ps)
    distrust_quark_eps = np.interp(distrust_quark_mu, mu_Bs_q, epss)
    n_q_diff_mu_at_distrust = np.interp(distrust_quark_mu, mu_Bs_q, np.gradient(n_Bs_q, mu_Bs_q))
    quark_limit_params = [distrust_quark_mu, distrust_quark_p, distrust_quark_n, n_q_diff_mu_at_distrust]

    ps_q_trunc = np.array([distrust_quark_p] + [p_q for p_q in ps if p_q > distrust_quark_p])
    epss_q_trunc = np.array([distrust_quark_eps] + [eps_q for eps_q in epss if eps_q > distrust_quark_eps])

    i_mu, i_n = 30, 30
    mu_search, n_search = [0.1, 0.1], [0.1, 0.1]
    (m, n), (unific, unific_diff, unific_diff_diff) = find_best_polynomial_params(unific_3_3_free_midpoint, APR_limit_params, quark_limit_params,
                                                                                  i_mu, i_n, mu_search, n_search)
    epsilon = 0.0002  # Adding this so there is no overlap in mu-arrays, and thus not in p-arrays
    mu_intermittent = np.linspace(distrust_APR_mu + epsilon, distrust_quark_mu + epsilon, 100, endpoint=True)
    ps_unific = unific(mu_intermittent, m, n, n_APR_diff_mu_at_distrust)
    ns_unific = unific_diff(mu_intermittent, m, n, n_APR_diff_mu_at_distrust)
    epss_unific = - ps_unific + mu_intermittent * ns_unific
    n_negative = 15
    ps_total = np.concatenate((np.array([-ps_nuc_trunc[n_negative]]), ps_nuc_trunc, ps_unific, ps_q_trunc))
    epss_total = np.concatenate((np.array([-epss_nuc_trunc[n_negative]]), epss_nuc_trunc, epss_unific, epss_q_trunc))
    # print("EoS unified allows p={} (dimensionless converted units)".format(-ps_nuc_trunc[n_negative] * conversion_factor))
    return interp1d(ps_total * conversion_factor, epss_total * conversion_factor, kind="linear", bounds_error=True,
                    assume_sorted=True)
