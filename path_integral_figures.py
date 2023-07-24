from plot_import_and_params import *
from scipy.interpolate import lagrange
from TOV import txt_to_arr, arr_to_txt

"""In relation to our development of the path integral in chapter 9, we have used two figures. 
This file containts the code which create the two. These figures are for illustrative purposes only."""

"""Starting with the first figure from chapter 9.2."""


def second_degree_polynomial(a, b, c=0):
    return lambda x: a * x**2 + b * x + c


def discrete_to_continuum():
    smooth_curve = np.zeros(1100)
    smooth_curve[0:200] = second_degree_polynomial(1, 0, 0)(np.linspace(0, 1, 200))
    smooth_curve[200:700] = second_degree_polynomial(-1, 2, 1)(np.linspace(0, 2.5, 500))
    smooth_curve[700:] = (lambda x: -1 / 4 - 3 * x + 51 / 16 * x ** 2 - 13 / 16 * x ** 3)(np.linspace(0, 2, 400))

    # Adding 20 equidistant points to a list
    pointlength = 10
    points = [smooth_curve[i * 110] for i in range(pointlength)]
    points.append(smooth_curve[-1])
    x_axis_p = np.linspace(0, 1, len(points), endpoint=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    ax1.set_title(r"(a) Discrete coordinates $q_n$ at $x_n$", fontdict={"fontsize": 10})
    ax2.set_title(r"(b) Continuous field $\phi (x)$", fontdict={"fontsize": 10})
    ax1.scatter(x_axis_p, points, color="blue", s=10)
    for n in range(1, len(points)):
        ax1.plot((x_axis_p[n - 1], x_axis_p[n]), (points[n - 1], points[n]), color="blue", linestyle="--", alpha=0.3)
    ax2.plot(np.linspace(0, 1, len(smooth_curve)), smooth_curve, color="blue")

    y_pad = 0.2
    ax1.set_ylim(-1 - y_pad, 2 + 2 * y_pad)
    ax2.set_ylim(-1 - y_pad, 2 + 2 * y_pad)
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)

    x_tick_labels = [r"$x_{}$".format("{" + str(n) + "}") for n in range(1, len(points) + 1)]
    x_ticks = np.linspace(0, 1, 11, endpoint=True)
    y_ticks = np.linspace(-1 - y_pad, 2 + 2 * y_pad, 7, endpoint=True)
    y_tick_labels = ["", "", 0, "", "", "", ""]

    ax1.plot((0, 1), (y_ticks[2], y_ticks[2]), color="black", alpha=0.5, linewidth=1.5)
    ax2.plot((0, 1), (y_ticks[2], y_ticks[2]), color="black", alpha=0.5, linewidth=1.5)

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([])
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_tick_labels)

    ax1.set_xlabel(r"$n\Delta x$")
    ax2.set_xlabel(r"$x$")
    ax1.set_ylabel(r"Displacement $q_n$, arb. units")
    ax2.set_ylabel(r"Displacement $\phi$, arb. units")

    ax1.grid()
    ax2.grid()

    fig.tight_layout()
    # Padding between subplots, as a fraction of avg. subplotwidth
    plt.subplots_adjust(wspace=0.3)

    # Uncomment to save figure
    # plt.savefig(fname="Discrete_and_continuous_string.svg", format="svg", dpi=600)
    plt.show()


"""Continuing with the figure in chapter 9.2.1.
Metropolis algorithm to generate fields distributed exponentially:
Move a coordinate one step dx. Calculate new energy of the configuration. Accept the step with probability 
P(accept) = min(1, exp(Delta E/(k_B T))). Run for a while.
Despite the figures only being for illustrative purposes, we use a specific algorithm. 
We have to generate some configurations anyway, so why not use a proper way?"""


def free_particle(q_n_arr, a=1, b=1):
    """Create a discrete string of points whose amplitude and nearest neighbour attraction contribute to the
    total energy of the string. Endpoints fixed. a is the coefficient for the harmonic potential around 0,
    b is the strength of nearest neighbour interaction."""
    energy = 0
    energy += sum(a * q_n_arr ** 2)
    for n in range(len(q_n_arr) - 1):
        energy += b * (q_n_arr[n] - q_n_arr[n + 1]) ** 2
    return energy


def free_energy_difference(q_n_arr, q_new, n, a=1, b=1):
    potential_difference = a * (q_new ** 2 - q_n_arr[n] ** 2)   # From harmonic potential term
    kinetic_difference = b * ((q_new - q_n_arr[n-1])**2 + (q_new - q_n_arr[n+1])**2 - (q_n_arr[n] - q_n_arr[n-1])**2 -
                              (q_n_arr[n] - q_n_arr[n+1])**2 )
    return potential_difference + kinetic_difference


def initial_state_rand_gauss(n, sigma):
    """Creating a string. Endpoints are fixed at 0, and the rest of the points are drawn from
    a Gaussian distribution."""
    random = np.random.normal(loc=0.0, scale=sigma, size=n)
    random[0], random[-1] = 0, 0
    return random


def metropolis(init_stat, energy_fnc, n_steps, beta, a=1, b=1):
    """Run the Metropolis-Hastings algorithm for n steps."""
    state = init_stat
    for n in range(n_steps):
        # Finding random index for which particle we would like to displace
        i = np.random.randint(1, len(state) - 1)    # randint is not inclusive [low, high)
        displacement = np.random.normal(0, 1)
        delta_E = energy_fnc(state, state[i] + displacement, i, a, b)
        if np.random.uniform(0, 1) < np.exp(-beta * delta_E):
            state[i] += displacement
    return state


# test_state = initial_state_rand_gauss(5, 2)
# metropolised_test_state = metropolis(test_state, free_energy_difference, 25, 100)


def plot_states_free_field(n_states, string_length, n_metropolis, beta, a=1, b=1, gauss_sigma=2, saved_file=""):
    if not saved_file:
        states = np.zeros((n_states, string_length))
        energies = np.zeros(n_states)
        old_energies = np.zeros(n_states)
        for i in range(n_states):
            init_state = initial_state_rand_gauss(string_length, gauss_sigma)
            old_energies[i] = free_particle(init_state, a, b)
            states[i, :] = metropolis(init_state, free_energy_difference, n_metropolis, beta, a, b)
            energies[i] = free_particle(states[i, :], a, b)
        print("Old energies: {}".format(old_energies))
        print("New energies: {}".format(energies))

    else:
        states = txt_to_arr(saved_file)
        print(states.shape)
        n_states = states.shape[0]
        energies = np.zeros(n_states)
        for i in range(n_states):
            energies[i] = free_particle(states[i, :], a, b)

    # Making colour palette:
    c_norm = colors.Normalize(vmin=np.sqrt(min(energies)), vmax=np.sqrt(max(energies)))
    c_map = plt.get_cmap("viridis")
    num_to_color = cm.ScalarMappable(cmap=c_map, norm=c_norm)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    x_axis = np.linspace(0, 1, string_length)
    x_axis_fine = np.linspace(0, 1, 100)
    ax.plot()
    ax.set_title(r"(a) Discrete partitioning $\Delta \beta$, $\phi_n$", fontdict={"fontsize": 10})
    ax2.set_title(r"(b) Continuum limit, $\phi_\beta$", fontdict={"fontsize": 10})

    # Limiting the plotting area
    ax.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    minimal, maximal = np.min(states), np.max(states)
    y_pad = 0.05 * max(abs(minimal), abs(maximal))
    y_max = max(abs(minimal - y_pad), abs(maximal + y_pad))
    ax.set_ylim(-y_max, y_max)
    ax2.set_ylim(-y_max, y_max)

    # Adding contour of the potential
    y = np.linspace(-y_max, y_max, 200)
    x = np.linspace(0, 1, 200)
    z = np.array([[q ** 2 for i in x] for q in y])
    ax.contourf(x, y, z, 200, cmap=plt.get_cmap("Reds"), vmax=np.max(z) * 1.5)
    ax2.contourf(x, y, z, 200, cmap=plt.get_cmap("Reds"), vmax=np.max(z) * 1.5)

    for i in range(n_states):
        ax.plot(x_axis, states[i, :], label="State {}".format(i), color=num_to_color.to_rgba(np.sqrt(energies[i])))
        ax.scatter(x_axis, states[i, :], color=num_to_color.to_rgba(np.sqrt(energies[i])), s=10)
        # To avoid rapid oscillation around the edges, introduce more interpolation points outside the plotting area
        # Adding two extra points outside
        dx = 1/(len(x_axis) - 1)
        n_extend = 25
        x_axis_extended = np.linspace(-n_extend * dx, 1 + n_extend * dx, string_length + 2 * n_extend)
        poly1 = lagrange(x_axis_extended, [0 for i in range(n_extend)] + list(states[i, :]) + [0 for i in range(n_extend)])
        ax2.plot(x_axis_fine, poly1(x_axis_fine), label="State {}".format(i),
                 color=num_to_color.to_rgba(np.sqrt(energies[i])))

    # Labelling axes
    ax.set_xlabel(r"$n\Delta\beta$")
    ax2.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"Displacement $\phi_n$, arb. units")
    ax2.set_ylabel(r"Displacement $\phi_\beta$, arb. units")

    # Must fix the tick labelling
    zero_index = int(np.floor(len(ax.get_yticklabels(which="both"))/2))
    y_ticklabels = [(lambda x: "" if x != zero_index else "0")(n) for n in range(len(ax.get_yticklabels(which="both")))]
    ax.set_yticklabels(y_ticklabels)
    ax2.set_yticklabels(y_ticklabels)
    ax.set_xticks([i/(string_length - 1) for i in range(string_length)])
    ax2.set_xticks([i/(string_length - 1) for i in range(string_length)])
    ax.set_xticklabels([r"$\beta_{}$".format("{" + str(n) + "}") for n in range(string_length)])
    ax2.set_xticklabels([])

    # Hiding ticks
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax2.tick_params(axis="y", which="both", left=False, right=False)
    ax2.tick_params(axis="x", which="both", bottom=False)

    # Plotting horisontal lines at particle positions
    for j in x_axis:
        ax.plot((j, j), (-y_max, y_max), color="black", alpha=0.5, linestyle=(0, (5, 5)),
                linewidth=0.5)
        ax2.plot((j, j), (-y_max, y_max), color="black", alpha=0.5, linestyle=(0, (5, 5)),
                 linewidth=0.5)

    plt.tight_layout()

    # For the plot added to the thesis, I wish to save the created array, as the process is stochastic.
    # arr_to_txt(states, "path_integral_discrete_points.txt")
    # plt.savefig(fname="path_integral_figure.svg", format="svg", dpi=600)

    plt.show()


"""Uncomment to generate the plots."""
# discrete_to_continuum()

# n_states, string_length = 4, 11
# beta = 4
# New configurations:
# plot_states_free_field(n_states, string_length, 3 * string_length ** 2, beta=beta, a=1, b=3, gauss_sigma=1)
# Fetching saved configuration:
# plot_states_free_field(n_states, string_length, 3 * string_length, beta=beta,
#                        saved_file="path_integral_discrete_points.txt")
