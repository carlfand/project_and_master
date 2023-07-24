from plot_import_and_params import *
plt.rc("text.latex", preamble=r"\usepackage{siunitx} \usepackage[T1]{fontenc} \usepackage{bm}")


"""The goal in this file is simply to plot he mexican hat potential. This is seen in chapter 10.2."""


def hat_potential(r, theta, lambd, v_square, h):
    """Magnitude is the value of sqrt(<sigma>^2 + <pi>^2).
    Theta = 0 is the direction of the sigma axis."""
    R, Theta = np.meshgrid(r, theta)
    linear_term = h * R * np.cos(Theta)
    radial_term = (R**2 - v_square)**2
    return radial_term - linear_term


def polar_plot(r_magnitude, lambd, v_square, h):
    """Polar plot.
    https://matplotlib.org/stable/gallery/mplot3d/surface3d_radial.html
    """
    rs = np.linspace(0, r_magnitude, 50)
    thetas = np.linspace(0, 2 * np.pi, 50)
    hat_flat = hat_potential(rs, thetas, lambd, v_square, 0)
    hat_tilted = hat_potential(rs, thetas, lambd, v_square, h)
    index = np.argmin(hat_tilted)

    vacuum_ring = hat_potential(v_square ** (1/2), thetas, lambd, v_square, 0)
    x_ring, y_ring = v_square ** (1/2) * np.cos(thetas), v_square ** (1/2) * np.sin(thetas)
    Rs, Thetas = np.meshgrid(rs, thetas)
    r_min, theta_min = Rs.flatten()[index], Thetas.flatten()[index]
    X, Y = Rs * np.cos(Thetas), Rs * np.sin(Thetas)

    fig = plt.figure(figsize=(6, 3))
    subfig1, subfig2 = fig.subfigures(ncols=2, width_ratios=[1, 1])
    ax1 = subfig1.add_subplot(projection="3d")
    ax2 = subfig2.add_subplot(projection="3d")
    ax1.set_title(r"Vacuum ring, $h = 0$", fontdict={"fontsize": 10})
    ax2.set_title(r"Unique vacuum, $h \neq 0$", fontdict={"fontsize": 10})

    ax1.plot_surface(X, Y, hat_flat, cmap=cm.get_cmap("viridis"), zorder=1)
    ax2.plot_surface(X, Y, hat_tilted, cmap=cm.get_cmap("viridis"), zorder=1)
    ax1.scatter(0, 0, v_square**2, s=15, color="black", zorder=5)
    ax2.scatter(0, 0, v_square**2, s=15, color="black", zorder=5)
    ax1.plot(x_ring, y_ring, vacuum_ring[:, 0], linewidth=1.8, zorder=2, color="red")
    ax1.scatter(x_ring[0], y_ring[0], vacuum_ring[0], s=15, color="red", zorder=3)
    ax2.scatter(r_min * np.cos(theta_min), r_min * np.sin(theta_min), hat_tilted.flatten()[index],
                color="red", zorder=3, s=15)

    ax1.set_xlabel(r"$\langle \sigma \rangle$", labelpad=0)
    ax2.set_xlabel(r"$\langle \sigma \rangle$", labelpad=0)
    ax1.set_ylabel(r"$\langle \boldsymbol{\pi} \rangle$", labelpad=0)
    ax2.set_ylabel(r"$\langle \boldsymbol{\pi} \rangle$", labelpad=0)
    ax1.set_zlabel(r"$\mathcal{V}\big(\langle \sigma \rangle, \, \langle \boldsymbol{\pi} \rangle\big)$", labelpad=0)
    ax2.set_zlabel(r"$\mathcal{V}\big(\langle \sigma \rangle, \, \langle \boldsymbol{\pi} \rangle\big)$", labelpad=0)

    ax1.spines["bottom"].set_position("zero")
    ax1.spines["left"].set_position("zero")

    ax1.tick_params(axis="x", labelbottom=False)
    ax1.tick_params(axis="y", labelleft=False)
    ax1.tick_params(axis="z", labelleft=False)

    ax2.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.tick_params(axis="z", labelleft=False)

    ax1.view_init(elev=15, azim=-70)
    ax1.view_init(elev=15, azim=-70)

    ax2.view_init(elev=15, azim=-70)
    ax2.view_init(elev=15, azim=-70)

    # Uncomment to save figure
    # plt.savefig(fname="Mexican_hats.svg", format="svg", dpi=600)
    plt.show()


# Uncomment to run.
polar_plot(5/4, 1/2, 1, 1/4)
