from plot_import_and_params import *
plt.rc("text.latex", preamble=r"""\usepackage{amssymb} \usepackage{siunitx} \usepackage[T1]{fontenc} \usepackage{xcolor} \DeclareMathSymbol{\shortminus}{\mathbin}{AMSa}{"39}""")
plt.rcParams.update({"font.size": 11})

"""Here, we wish to make an illustration of the matsubara frequency summation.
In the end, it is just an illustration, but we include it for completion."""


def add_box_to_plot(ax, point, sidelength, color="blue"):
    points = np.array([[point[0] + sidelength/2, point[1] - sidelength/2], [point[0] + sidelength/2, point[1] + sidelength/2],
                       [point[0] - sidelength/2, point[1] + sidelength/2], [point[0] - sidelength/2, point[1] - sidelength/2]])
    ax.scatter(point[0], point[1], color="black")
    for i in range(0, 4):
        ax.annotate("", xytext=(points[i-1, 0], points[i-1, 1]), xy=(points[i, 0], points[i, 1]), arrowprops=dict(arrowstyle="->",color=color))
    return ax


def add_box_to_plot_2(ax, point, sidelength, arrowlength=None, color="blue", n_curve=""):
    points = np.array(
        [[point[0] + sidelength / 2, point[1] - sidelength / 2], [point[0] + sidelength / 2, point[1] + sidelength / 2],
         [point[0] - sidelength / 2, point[1] + sidelength / 2],
         [point[0] - sidelength / 2, point[1] - sidelength / 2]])
    ax.scatter(point[0], point[1], s=10, color="black")
    if not arrowlength:
        arr_len = sidelength/10
    else:
        arr_len = arrowlength
    for i in range(0, 4):
        ax.plot((points[i - 1, 0], points[i, 0]), (points[i - 1, 1], points[i, 1]), color=color)
        offset_coord = ((points[i, 0] + points[i - 1, 0])/2 - point[0], (points[i, 1] + points[i - 1, 1])/2 - point[1])
        if abs(offset_coord[0]) < sidelength/4:
            sign = -offset_coord[1]/abs(offset_coord[1])
            arrow_direction = (sign * arr_len/2, 0)   # Arrow shall point along x-direction
        else:
            sign = offset_coord[0] / abs(offset_coord[0])
            arrow_direction = (0, sign * arr_len/2)   # Arrow shall point along y-direction
        arrow_start = (point[0] + offset_coord[0] - arrow_direction[0]/2, point[1] + offset_coord[1] - arrow_direction[1]/2)
        ax.arrow(x=arrow_start[0], y=arrow_start[1], dx=arrow_direction[0], dy=arrow_direction[1], color=color,
                 length_includes_head=True, head_width=arr_len*0.5, width=arr_len*0.1)
        zero_offset = 0
        if int(n_curve) < 0:
            sign = r"\shortminus"
        else:
            sign = ""
        ax.text(s=r"$\mathcal{}$".format("{C}_{" + sign + str(abs(int(n_curve))) + "}"), x=(point[0] + sidelength * 0.6),
                y=point[1] - sidelength/10 + zero_offset)
    return ax


def add_imaginary_pole_label(ax, point, sidelength, n_pole):
    zero_offset = 0
    if int(n_pole) < 0:
        sign = r"\shortminus"
    else:
        sign = ""
    ax.text(s=r"$p_{}$".format("{" + sign + str(abs(int(n_pole))) + "}"), x=point[0] - sidelength * 1/2.4, y=point[1] - sidelength/20 + zero_offset)
    return ax


def make_matsuraba_freq_plot(y_pos, n_points, sidelength):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9.6, 4), gridspec_kw={'width_ratios': [1, 1, 1.6]})
    ax, ax2, ax3 = axs
    epsilon = sidelength/10
    if not n_points % 2:
        y_values = np.linspace(y_pos - sidelength * (n_points//2 - 1) + 1/2 * sidelength, y_pos + sidelength * n_points//2 + 1/2 * sidelength, n_points)
        enumerate_offset = n_points // 2 - 1
    else:
        y_values = np.linspace(y_pos - sidelength * n_points//2 + 1/2 * sidelength, y_pos + sidelength * n_points//2 + 1/2 * sidelength, n_points)
        enumerate_offset = n_points // 2
    ax2.scatter([0 for i in range(n_points)], y_values, s=10, color="black")
    ax3.scatter([0 for i in range(n_points)], y_values, s=10, color="black")
    for n, y in enumerate(y_values):
        add_box_to_plot_2(ax, (0, y), sidelength - epsilon, arrowlength=None, color="red",
                          n_curve=str(n - enumerate_offset))
        add_imaginary_pole_label(ax, (0, y), sidelength, str(n - enumerate_offset))
        add_imaginary_pole_label(ax2, (0, y), sidelength, str(n - enumerate_offset))
        add_imaginary_pole_label(ax3, (0, y), sidelength, str(n - enumerate_offset))
        for ax_n in axs[1:]:
            ax_n.arrow(x=-sidelength / 2, y=(y + sidelength / 20), dx=0, dy=-sidelength / 10, color="red",
                       length_includes_head=True, head_width=sidelength / 15, width=sidelength / 100)
            ax_n.arrow(x=sidelength / 2, y=(y - sidelength / 20), dx=0, dy=sidelength / 10, color="red",
                       length_includes_head=True, head_width=sidelength / 15, width=sidelength / 100)
    x_pole = 3/4

    for ax_n in axs:
        # Set bottom and left spines as x and y axes of coordinate system
        ax_n.spines["bottom"].set_position("zero")
        ax_n.spines["left"].set_position("zero")

        # Remove top and right spines
        ax_n.spines["top"].set_visible(False)
        ax_n.spines["right"].set_visible(False)

        # Create "Re z" and "Im z" labels placed at the end of the axes
        ax_n.set_xlabel(r"$\mathrm{Re} \, \omega$", size=10, labelpad=5, x=1.1)
        ax_n.set_ylabel(r"$\mathrm{Im} \, \omega$", size=10, labelpad=-5, y=1.02, rotation=0)

        # Draw arrows
        arrow_fmt = dict(markersize=4, color='black', clip_on=False)
        ax_n.plot((1), (0), marker='>', transform=ax_n.get_yaxis_transform(), **arrow_fmt)
        ax_n.plot((0), (1), marker='^', transform=ax_n.get_xaxis_transform(), **arrow_fmt)

        ax_n.set_xticklabels("")
        ax_n.set_yticklabels("")

        ax_n.scatter((-x_pole, x_pole), (0, 0), s=10, color="black")
        ax_n.text(s=r"$r_\shortminus$", x=-x_pole * 1.05, y=sidelength / 5)
        ax_n.text(s=r"$r_+$", x=x_pole * 0.95, y=sidelength / 5)

        y_lim_max = y_values[-1] + sidelength * 0.8
        # Set identical scales for both axes
        ax_n.set(xlim=(-x_pole - sidelength / 4, x_pole + sidelength / 4),
               ylim=(y_values[0] - sidelength * 0.8, y_lim_max), aspect='equal')

        ax_n.set_xticks([])
        ax_n.set_yticks([])

    axs[2].set_xlabel(r"$\mathrm{Re} \, \omega$", size=10, labelpad=5, x=1.03)  # Resetting real label for third plot.
    # ax.grid()

    # Plotting for ax2 and ax3:
    n = 1
    for ax_n in axs[1:]:
        if n == 2:
            offset = 1/2 * sidelength # Plotting over a slightly smaller area for the half-circular contours
        else:
            offset = 0
        ax_n.plot((-sidelength/2, -sidelength/2),
                  (y_values[0] - sidelength * 0.8 + epsilon - offset, y_values[-1] + sidelength * 0.8 - epsilon - offset), color="red")
        ax_n.plot((sidelength/2, sidelength/2),
                  (y_values[0] - sidelength * 0.8 + epsilon - offset, y_values[-1] + sidelength * 0.8 - epsilon - offset), color="red")
        n += 1

    ax2.plot((-sidelength/2, sidelength/2),
             (y_values[-1] + sidelength * 0.8 - epsilon, y_values[-1] + sidelength * 0.8 - epsilon),
             color="red", linestyle="--")
    ax2.plot((-sidelength/2, sidelength/2),
             (y_values[0] - sidelength * 0.8 + epsilon, y_values[0] - sidelength * 0.8 + epsilon),
             color="red", linestyle="--")
    ax2.text(s=r"$\mathcal{C}$", x=sidelength * 0.65, y=sidelength)

    # Plotting curved part of contour in ax3:
    r = y_lim_max - epsilon - 1/2 * sidelength
    x_axis = r * np.cos(np.linspace(-np.pi/2, np.pi/2))
    y_values = r * np.sin(np.linspace(-np.pi/2, np.pi/2))
    ax3.plot((-x_axis - sidelength/2), y_values, color="red")
    ax3.plot((x_axis + sidelength / 2), y_values, color="red")
    max_x = max(x_axis + sidelength/2 + epsilon)
    ax3.text(s=r"$\mathcal{C}'_2$", x=y_lim_max - sidelength * 2 / 3, y=sidelength)
    ax3.text(s=r"$\mathcal{C}'_1$", x=-y_lim_max + sidelength * 4 / 9, y=sidelength)
    ax3.set_xlim(-max_x, max_x)
    ax3.set_ylim(-(r + epsilon), r + epsilon)

    ax.set_title(r"$\sum_n \oint_{\mathcal{C}_n} d\omega \, g(\omega)$", pad=30, fontdict={"fontsize": 11})
    ax2.set_title(r"$\oint_{\mathcal{C}} d\omega \, g(\omega)$", pad=30, fontdict={"fontsize": 11})
    ax3.set_title(r"$\oint_{\mathcal{C}'_1} d\omega \,  g(\omega) + \oint_{\mathcal{C}'_2} d\omega \,  g(\omega)$",
                  pad=30, fontdict={"fontsize": 11})
    if n_points // 2:
        extra_pole = (0, y_pos - sidelength * (n_points // 2 + 1) + 1 / 2 * sidelength)
        ax3.scatter(extra_pole[0], extra_pole[1], color="black", s=10)
        add_imaginary_pole_label(ax3, extra_pole, sidelength, str(-3))
    plt.tight_layout()
    # Uncomment to save figure.
    # plt.savefig(fname="Matsubara_freq_plot.svg", format="svg", dpi=600)
    plt.show()

# Create plot with 5 poles.
make_matsuraba_freq_plot(0, 5, 1/2)
