import numpy as np
import numba
from scipy.optimize import bisect
np.seterr(all="raise")

"""In this file we have written down all the functions we need to integrate the TOV-equation with RK4.
Note that we have a special integrator for ideal neutrons stars, where we know eps(x_F) and p(x_F).

We start by choosing a central pressure, and integrate until we reach p=0. We now know the total mass M, the radius R,
 and the internal pressure p_c. which we save in tuples of (M, R, p_c). These will be refered to as mrp-triplets."""
solar_mass = 1.9891 * 10 ** 30  # In order to express masses in terms of solar masses.


def runge_kutta4(func, y, x, stepsize, *params):
    """
    Standard RK4. y is a vector, x is an evolving parameter.
    """
    f1 = stepsize * func(y, x, *params)
    f2 = stepsize * func(y + f1/2, x + stepsize/2, *params)
    f3 = stepsize * func(y + f2/2, x + stepsize/2, *params)
    f4 = stepsize * func(y + f3, x + stepsize, *params)
    return y + f1/6 + f2/3 + f3/3 + f4/6


@numba.njit
def TOV_dimless(p_bar, r, M_bar, eps_bar, eps_g, c, G):
    """Evaluates the TOV-equation in its dimensionless form."""
    if r == 0:
        # Special case giving division by zero error.
        return 0
    elif eps_bar == 0:
        # Another special case giving division by zero error.
        return 0    # For the NR-case, TOV diverges for smaller and smaller p/eps
    elif M_bar == 0:
        # Last special case giving division by zero error.
        return 0
    return - (G * eps_g * M_bar * eps_bar) / (r ** 2 * c ** 2) * (1 + p_bar / eps_bar) * (
                1 + (4 * np.pi * r ** 3 * p_bar) / (M_bar * c ** 2)) / (1 - (2 * G * eps_g * M_bar) / (r * c ** 2))


def coupled_derivative(vec, r, EoS, eps_g, c, G, *params):
    """M and p evolves simultaneously. This vector is passed to RK4."""
    p_bar, M_bar = vec
    return np.array([TOV_dimless(p_bar, r, M_bar, EoS(p_bar, *params), eps_g, c, G),
                     np.pi * 4 * r ** 2 * EoS(p_bar, *params) / c ** 2])


def mass_radius_bar(p_c_bar, EoS, eps_g, c, G, step):
    """Creates an (M, r, p_c)-tuple for a given central pressure and equation of state."""
    # Initialising values:
    p_bar = [p_c_bar]
    M_bar = 0
    r = 0
    dp = 0
    assert 1 > (8 * G * eps_g * step ** 2 * np.pi * EoS(p_c_bar)) / (3 * c ** 4), "Stepsize too large!"
    n, n_max = 0, 80000     # Setting a maximum number of steps to stop integration.
    while p_bar[-1] > dp and n < n_max:
        p_next, M_bar = runge_kutta4(coupled_derivative, np.array([p_bar[-1], M_bar]), r, step, EoS, eps_g, c, G)
        p_bar.append(p_next)
        r += step
        n += 1
        if not n % 10000:
            print(n)
    if n == n_max:
        # If we cut of the integration before p=0 is reached, we inform about this.
        print("Ingration did not reach p=0 after n = {} iterations.".format(n))
    return np.array(p_bar), M_bar


def ps_p_diffs_Ms(p_c_bar, EoS, eps_g, c, G, step):
    """Similar to mass_radius_bar. This function returns arrays with all the intermediate values of
    p(r), p'(r) and M(r). This is useful for stability analysis."""
    p_bar = [p_c_bar]
    M_bar = [0]
    p_bar_diff = [0]
    r = 0
    dp = 0
    assert 1 > (8 * G * eps_g * step ** 2 * np.pi * EoS(p_c_bar)) / (3 * c ** 4), "Stepsize too large!"
    n, n_max = 0, 100000
    while p_bar[-1] > dp and n < n_max:
        p_next, M_next = runge_kutta4(coupled_derivative, np.array([p_bar[-1], M_bar[-1]]), r, step, EoS, eps_g, c, G)
        M_bar.append(M_next)
        p_bar.append(p_next)
        p_bar_diff.append(coupled_derivative([p_bar[-1], M_bar[-1]], r, EoS, eps_g, c, G)[0])
        r += step
        n += 1
        if not n % 10000:
            print(n)
    if n == n_max:
        print("Did not converge after n = {} iterations.".format(n))
    return np.array(p_bar), np.array(p_bar_diff), np.array(M_bar), step*(len(p_bar) - 1)


def get_mass_radius_p_c_triplet(p_min, p_max, EoS, energy_density_scale, step, n):
    """Creates an array of (M, R, p_c)-tuples.
    Energy density scale is the dimensionful scale we use. This means: p_bar = p / energy_density scale."""
    # Initialising log-space array of central pressures
    p_cs = np.array([10 ** pwr for pwr in np.linspace(np.log10(float(p_min)), np.log10(float(p_max)), n)])
    mass_radius_p_c_triplets = np.zeros((n, 3))
    for i, p_c in enumerate(p_cs):
        p_bar, M_bar = mass_radius_bar(p_c / energy_density_scale, EoS, energy_density_scale, c=3*10**8,
                                       G=6.674*10**(-11), step=step)
        # Converting into sensible units: M in terms of solar masses. R is in metres, p_c is in Pa.
        mass_radius_p_c_triplets[i, :] = np.array([M_bar * energy_density_scale / solar_mass, step * len(p_bar), p_c])
    return mass_radius_p_c_triplets


def write_results_to_file(p_min, p_max, EoS, energy_density_scale, n, step=1/4, fname="test_mrp_QM.txt"):
    """Writing results directly to a file."""
    print("Finding mrp-triplets.")
    mrp_triplet = get_mass_radius_p_c_triplet(p_min, p_max, EoS, energy_density_scale, step, n)
    arr_to_txt(mrp_triplet, fname)


@numba.njit
def epsilon_analytic(x_F):
    # Dimensionless, eps / eps_g.
    return 1 / 8 * (2 * x_F**3 * np.sqrt(1 + x_F**2)
                    + x_F * np.sqrt(1 + x_F**2) - np.arcsinh(x_F))


@numba.njit
def p_analytic(x_F):
    # Dimensionless, p / eps_g.
    return x_F**3 * np.sqrt(1 + x_F**2)/12 \
        + 1/8 * (np.arcsinh(x_F) - x_F * np.sqrt(1 + x_F**2))


@numba.njit
def TOV_x_F(x_F, r, M_bar, eps_g, c, G):
    """Rewritten the TOV-equation in terms of x_F for ideal neutron stars. This runs quicker, as we do not need to
    use a root finder to find eps for every p."""
    eps_bar, p_bar = epsilon_analytic(x_F), p_analytic(x_F)
    if x_F < 0:
        return 0
    elif r == 0:
        return 0
    elif M_bar == 0:
        return 0
    else:
        return - 3 * np.sqrt(1 + x_F ** 2) / x_F ** 4 * (G * eps_g *
                                                         M_bar*eps_bar) /\
               (r ** 2 * c ** 2) * (1 + p_bar / eps_bar) *  \
               (1 + (4 * np.pi * r ** 3 * p_bar) / (M_bar * c ** 2)) / \
               (1 - (2 * G * eps_g * M_bar) / (r * c ** 2))


def coupled_derivative_x_F(vec, r, eps_g, c, G):
    """Special coupled derivative for ideal neutron stars, expressed in terms of x_F."""
    x_F, M_bar = vec
    return np.array([TOV_x_F(x_F, r, M_bar, eps_g, c, G),
                     4 * np.pi * r ** 2 / c**2 * epsilon_analytic(x_F)])


def integrate(p_c_bar, eps_g, c, G, step, bisect_rel_error):
    """Integrates the TOV-equation for an ideal neutron star.
    Similar to ps_p_diffs_Ms, but specialised for ideal neutron stars.
    This means that this function returns the intermediate values x_F(r), x_F'(r), M(r).
    This is useful for stability analysis."""
    # Find first a relevant interval of x_F
    x_max = 1.0
    x_min = 1/2
    while p_c_bar < p_analytic(x_min):
        # If the guessed x_max and x_min were too large
        x_min *= 1/2
        x_max *= 1/2
    while p_c_bar > p_analytic(x_max):
        # If the guessed x_max and x_min were too small
        x_max *= 2
        x_min *= 2
    M_bar, r = 0, 0
    # Numerically find correct value for x_F, given the central pressure
    x_F = bisect(f=lambda x: p_analytic(x) - p_c_bar, a=x_min, b=x_max,
                 xtol=x_min * bisect_rel_error)
    x_Fs, x_F_diffs, M_bars = [], [], []
    # Introducing cut off, to avoid being stuck in the while loop
    n, n_max = 0, 150000
    while x_F > 0 and n < n_max:
        x_Fs.append(x_F), M_bars.append(M_bar)
        x_F, M_bar = runge_kutta4(coupled_derivative_x_F,
                                  [x_Fs[-1], M_bars[-1]],
                                  r, step, eps_g, c, G)
        x_F_diffs.append(coupled_derivative_x_F([x_Fs[-1], M_bars[-1]],
                                                r, eps_g, c, G)[0])
        r += step
        n += 1
        if not n % 10000:
            print(n)
    if n >= n_max:
        print("Did not converge after {} steps.".format(n_max))
    return np.array(x_Fs), np.array(x_F_diffs), np.array(M_bars), r - step


def triplet(p_min, p_max, step, n, rel_acc):
    """Generate an array of (M_bar, R, p_c)-tuples for the analytic EoS
    for the ideal neutron star. Similar to get_mass_radius_p_c_triplet,
    but specialised for ideal neutron stars."""
    # Constants:
    eps_g, c, G = 1.646776 * 10 ** 36, 3 * 10 ** 8, 6.674 * 10 ** (-11)
    # Constructing logspace
    p_cs = [10 ** pwr for pwr in np.linspace(np.log10(float(p_min)),
                                             np.log10(float(p_max)), n)]
    # Initialising array for mrp-triplets
    mass_radius_p_c_triplets = np.zeros((n, 3))
    for i, p_c in enumerate(p_cs):
        # Performing integration
        x_Fs, x_F_diffs, M_bars, R = integrate(p_c/eps_g, eps_g, c,
                                               G, step=step,
                                               bisect_rel_error=rel_acc)
        # Saving only relevant data
        mass_radius_p_c_triplets[i, :] = np.array([M_bars[-1] * eps_g / solar_mass,
                                                   step*len(x_Fs), p_c])
    return mass_radius_p_c_triplets


"""Finally, we include some utility functions to save the tuples we calculate."""


def arr_to_txt(array, filename):
    arr_file = open(filename, "w+")
    try:
        for data_line in array:
            arr_file.write("{}\n".format(",".join([str(num) for num in data_line.flatten()])))
    finally:
        arr_file.close()


def txt_to_arr(filename):
    """Reads a .txt-file into a 2D array."""
    read_file = open(filename, "r")
    try:
        # First, finding shape of array:
        line1 = [float(entry) for entry in read_file.readline().split(",")]
        line_len = len(line1)
        n = sum(1 for line in read_file if line.rstrip()) + 1
    finally:
        read_file.close()
    read_file = open(filename, "r")
    try:
        # Filling correctly shaped array with data
        array = np.zeros((n, line_len))
        for i, line in enumerate(read_file):
            array[i, :] = np.array([float(entry) for entry in line.split(",")])
    finally:
        read_file.close()
    return array


def merge_mrp_files(destination_filename, *filenames):
    """Spesifically for merging files of three columns of data, sorted by the third column."""
    dest_file = open(destination_filename, "w")
    data_dictionary = {}
    try:
        for filename in filenames:
            data_file = open(filename, "r")
            try:
                data_dictionary = dict(**data_dictionary, **{line.split(",")[2]: (line.split(",")[0],
                                                                                  line.split(",")[1]) for
                                                             line in data_file.readlines() if line.split(",")[2] not in data_dictionary.keys()})
            finally:
                data_file.close()
        for key, (value1, value2) in sorted(data_dictionary.items(), key=lambda triplet: float(triplet[0])):
            dest_file.write("{},{},{}".format(value1, value2, key))
    finally:
        dest_file.close()

