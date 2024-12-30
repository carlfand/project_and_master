import numpy as np
from TOV_QM_model import txt_to_arr, write_results_to_file, mass_radius_bar, merge_mrp_files
from QM_EoS import EoS_QM_maxwell_construct, find_bag_window, pressure, find_mus, omega_0_diff_cr, omega_0_cr, \
    omega_0_diff_incr, omega_0_incr
from Hybrid_stars import EoS_hybrid_standard, EoS_unified_standard_3_3, EoS_unified_3_3_standard_unshifted
from Nuclear_EoS import EoS_pure_APR
from math import log10
"""Copied plotting function from TOV_dimless.py in Prosjektoppgave-project and modified."""
# In order to follow along the lines of the project thesis, we use the same dimensionfull scale as we did for the ideal
# neutron star
eps_g_neutron = 1.646776 * 10 ** 36
solar_mass = 1.9891 * 10 ** 30
eps_scale = eps_g_neutron

"""In this file, we use the functions from TOV, QM_EoS, and Hybrid_stars to calculate mass-radius relations. We save
the results to .txt-files"""


def write_to_file_maxwell_min_bag_shift(m_sigma, p_min, p_max, n, omega_0, omega_0_diff, fname="", bag_const_add=0.0, cr=False, step=0.5):
    # Getting EoS
    # m_sigma = 700
    vev_bar_min, vev_bar_max = 0.002, 0.9999
    # p_min, p_max = 10**35, 10**36.5
    accurate_vevs = np.linspace(vev_bar_min, vev_bar_max, 5000)
    vevs = np.linspace(vev_bar_min, vev_bar_max, 1000)
    bag_lowest = find_bag_window(accurate_vevs, m_sigma, 3, omega_0, omega_0_diff)
    print("Minimal bag constant: {}".format(bag_lowest))
    if not fname:
        # filename = "m_sigma_{}_bag_const_{}_32_36_5.txt".format(m_sigma, "_".join(str(round(bag_lowest + bag_const_add, 4)).split()))
        filename = fname_creator(cr, m_sigma, p_min, p_max, bag_lowest + bag_const_add)
    else:
        filename = fname
    to_SI_conversion, dimful_scale = 1.56 * 10**33, eps_scale
    vev_and_mu = find_mus(np.array([vev_bar_min]), m_sigma / 93, omega_0_diff)
    highest_pressure = pressure(vev_and_mu, m_sigma / 93, omega_0)
    print("Highest possible pressure in transformed units: {}\nHighest internal parametrising pressure in transformed units: {}".format(highest_pressure * to_SI_conversion / dimful_scale, p_max / eps_g_neutron))
    EoS = EoS_QM_maxwell_construct(vevs, m_sigma, omega_0, omega_0_diff, bag_const=bag_lowest + bag_const_add,
                                   conversion_factor=to_SI_conversion / eps_scale)
    print("Write results to file called {}".format(filename))
    write_results_to_file(p_min, p_max, EoS, eps_scale, step=step, n=n, fname=filename)


def fname_creator(cr, m_sigma, p_min, p_max, bag_const):
    if cr:
        return "{}_m_simga_{}_log_p_{}_{}_bag_{}.txt".format("cr", m_sigma, log10(p_min), log10(p_max), str(round(bag_const, 4)))
    else:
        return "{}_m_simga_{}_log_p_{}_{}_bag_{}.txt".format("incr", m_sigma, log10(p_min), log10(p_max),
                                                             str(round(bag_const, 4)))



def write_to_file_hybrid_star(m_sigma, p_min, p_max, n, step=0.5, bag_add=0.0, fname="", EoS=EoS_hybrid_standard, **EoS_kwargs):
    if not fname:
        filename = "hybrid_m_sigma_{}_log_p_{}_{}_n_{}_bag_add_{}.txt".format(m_sigma, round(log10(p_min), 4), round(log10(p_max), 4), n, bag_add)
    else:
        filename = fname
    to_SI_conversion, dimful_scale = 1.56 * 10 ** 33, eps_scale
    conv_fac = to_SI_conversion / dimful_scale
    # NB: With large enough p_max, one might have to adjust the lower bound of vevs = np.linspace(epsilon, ...),
    # as those values yields the largest pressures. Do so manually by changing epsilon_left in EoS_hybrid_standard()
    hybrid_EoS = EoS(m_sigma, bag_const_add=0.0, conversion_factor=conv_fac, n_vev=3000, **EoS_kwargs)
    write_results_to_file(p_min, p_max, hybrid_EoS, eps_scale, n, step=step, fname=filename)


"""Uncomment for calculation of incr calculation."""
omega_0, d_omega_0 = omega_0_incr, omega_0_diff_incr
# More accurately around the maximum masses: (obtained from inspecting files with pressures in [10**32, 10**36])
# p_min_max_dict = {600: [4*10**34, 5.17*10**34], 700: [4.1*10**34, 5.2*10**34], 800: [5.2*10**34, 6.6*10**34]}
# for m_sigma, (p_min, p_max) in p_min_max_dict.items():
#     write_to_file_maxwell_min_bag_shift(m_sigma, p_min, p_max, 50, omega_0, d_omega_0, bag_const_add=0.0,
#                                         step=0.25, fname="incr_around_max_m_sigma_{}_log_p_{}_{}_min_bag.txt".format(m_sigma, round(math.log10(p_min), 3), round(math.log10(p_max), 3)))
# write_to_file_maxwell_min_bag_shift(800, 10**32, 10**36, 200, omega_0, d_omega_0, bag_const_add=0.0, cr=False)
write_to_file_maxwell_min_bag_shift(700, 10**33, 10**35, 50, omega_0, d_omega_0, bag_const_add=0.0, cr=False)
# write_to_file_maxwell_min_bag_shift(600, 10**32, 10**36, 200, omega_0, d_omega_0, bag_const_add=0.0, cr=False)

"""Uncomment for calculation of cr calculation."""
# omega_0, d_omega_0 = omega_0_cr, omega_0_diff_cr
# More accurately around the maximum masses: (obtained from inspecting files with pressures in [10**32, 10**36])
# p_min_max_dict = {600: [4.7*10**34, 6.2*10**34], 500: [3.9*10**34, 5.17*10**34], 400: [3.74*10**34, 4.93*10**34]}
# for m_sigma, (p_min, p_max) in p_min_max_dict.items():
#     write_to_file_maxwell_min_bag_shift(m_sigma, p_min, p_max, 50, omega_0, d_omega_0, bag_const_add=0.0,
#                                         step=0.25, fname="cr_around_max_m_sigma_{}_log_p_{}_{}_min_bag.txt".format(m_sigma, round(math.log10(p_min), 3), round(math.log10(p_max), 3)))


# write_to_file_maxwell_min_bag_shift(600, 4.7**34, 6.2**34, 100, omega_0, d_omega_0, bag_const_add=0.0, cr=True, step=0.25, fname="cr_{}_around_max")

# write_to_file_maxwell_min_bag_shift(500, 10**32, 10**36, 200, omega_0, d_omega_0, bag_const_add=0.0, cr=True)
# write_to_file_maxwell_min_bag_shift(550, 10**32, 10**36, 200, omega_0, d_omega_0, bag_const_add=0.0, cr=True)


# merge_file_list = ["test_mrp_QM.txt", "test_mrp_QM2.txt", "test_mrp_QM3.txt", "test_mrp_QM4.txt"]
# merge_mrp_files("merged_result_QM_inconst_renorm.txt", *merge_file_list)
# merge_mrp_files("merged_result_QM_inconst_renorm.txt", *["merged_result_QM_inconst_renorm.txt", "test_mrp_QM4.txt"])

# write_to_file_maxwell_min_bag_shift(600, 10**32, 10**36.5, 3, omega_0, d_omega_0, "mrp_m_sigma_600_minimising_bag_const.txt")
# write_to_file_maxwell_min_bag_shift(800, 10**32, 10**36.5, 200, omega_0, d_omega_0, "mrp_m_sigma_800_minimising_bag_const.txt")
# write_to_file_maxwell_min_bag_shift(600, 10**32, 10**36.5, 200, omega_0, d_omega_0, "mrp_m_sigma_600_minimising_bag_const.txt")
"""Uncomment for hybrid star calculation:"""
# IMPORTANT: This does NOT converge for step = 1.
# write_to_file_hybrid_star(m_sigma=400, p_min=3.8*10**34, p_max=10**35.15, n=50, step=0.5, bag_add=0, fname="")
# write_to_file_hybrid_star(m_sigma=500, p_min=3.8*10**34, p_max=10**35.15, n=50, step=0.5, bag_add=0, fname="")
# write_to_file_hybrid_star(m_sigma=550, p_min=3.8*10**34, p_max=10**35.15, n=100, step=0.5, bag_add=0, fname="")
# write_to_file_hybrid_star(m_sigma=600, p_min=5.0*10**34, p_max=10**35.15, n=50, step=0.5, bag_add=0, fname="")

# write_results_to_file(p_min=10**35, p_max=2.8*10**35,
#                       EoS=EoS_pure_APR(conversion_factor=(1.56 * 10 ** 33) / eps_g_neutron),
#                       energy_density_scale=eps_scale, n=24, step=0.5,
#                       fname="pure_APR_log_p_35_35.447_n_24.txt")

"""Uncomment for unified EoS hybrid star, n_APR = 2, n_q = 4."""
"""
n_points = 75
m_sigmas, lower_ps, upper_ps = [400, 500, 600], [10**33.367, 10**33.367, 10**33.367], [10**36, 10**36, 10**36]
for m_sigma, lower_p, upper_p in zip(m_sigmas, lower_ps, upper_ps):
    filename = "Unified_m_sigma_{}_log_p_{}_{}_n_{}.txt".format(m_sigma, round(log10(lower_p), 4),
                                                                round(log10(upper_p), 4), n_points)
    write_to_file_hybrid_star(m_sigma=m_sigma, p_min=lower_p, p_max=upper_p, n=n_points, step=0.5, bag_add=0, fname=filename,
                              EoS=EoS_unified_standard_3_3)
write_to_file_hybrid_star(600, 10**33.367, 10**36, 75, step=0.5,
                          fname="Unified_m_sigma_600_log_p_{}_{}_n_{}_corrected.txt".format(round(log10(10**33.367), 4), round(log10(10**36), 4), 75), EoS=EoS_unified_standard_3_3)
"""
# write_to_file_hybrid_star(550, 10**33.367, 10**36, 75, step=0.5,
#                           fname="Unified_m_sigma_550_log_p_{}_{}_n_{}.txt".format(round(log10(10**33.367), 4), round(log10(10**36), 4), 75), EoS=EoS_unified_standard_3_3)

"""Uncomment for unified EoS hybrid star unshifted m_n, n_APR = 2, n_q = 6."""
"""
n_points = 75
# m_sigmas = [400, 500, 600]
m_sigmas = [550]
p_lower, p_upper = 10**33.367, 10**36
for m_sigma in m_sigmas:
    write_to_file_hybrid_star(m_sigma, p_lower, p_upper, n_points, step=0.5,
                              fname="Unified_m_sigma_{0}_log_p_{1}_{2}_n_{3}_unshifted.txt".format(m_sigma, round(log10(p_lower), 4), round(log10(p_upper), 4), n_points),
                              EoS=EoS_unified_3_3_standard_unshifted, limit_APR_n=2, limit_quark_n=6)
"""

"""Uncomment for APR-mass-radius for unshifted m_n."""
# The upper limit for p in pascal from the APR data is also 2.8*10^35 also for the unshifted m-n
"""
p_lower, p_upper = 10**32, 2.8 * 10**35
n_points = 160
write_results_to_file(p_min=p_lower, p_max=p_upper,
                      EoS=EoS_pure_APR(conversion_factor=(1.56 * 10 ** 33) / eps_g_neutron, shifted=False),
                      energy_density_scale=eps_scale, n=n_points, step=0.5,
                      fname="pure_APR_log_p_{}_{}_n_{}_unshifted.txt".format(round(log10(p_lower), 3),
                                                                             round(log10(p_upper), 3), n_points))
"""

# write_to_file_hybrid_star(m_sigma=500, p_min=3.8*10**34, p_max=10**35.15, n=50, step=0.5, bag_add=0, fname="")
# write_to_file_hybrid_star(m_sigma=600, p_min=5.0*10**34, p_max=10**35.15, n=50, step=0.5, bag_add=0, fname="")

# merge_mrp_files("pure_APR_log_p_32.3010_36_n_159.txt", "pure_APR_log_p_35_35.447_n_24.txt",
#                 "pure_APR_log_p_32.3010_35_n_135.txt")


"""
merge_mrp_files("hybrid_m_sigma_400_log_p_34.5798_36.0_n_100_bag_add_0_merged.txt",
                "hybrid_m_sigma_400_log_p_34.5798_36.0_n_50_bag_add_0.txt",
                "hybrid_m_sigma_400_log_p_34.5798_35.15_n_50_bag_add_0.txt")

merge_mrp_files("hybrid_m_sigma_500_log_p_34.5798_36.0_n_100_bag_add_0_merged.txt",
                "hybrid_m_sigma_500_log_p_34.5798_36.0_n_50_bag_add_0.txt",
                "hybrid_m_sigma_500_log_p_34.5798_35.15_n_50_bag_add_0.txt")

merge_mrp_files("hybrid_m_sigma_600_log_p_34.6902_36.0_n_100_bag_add_0_merged.txt",
                "hybrid_m_sigma_600_log_p_34.6902_36.0_n_50_bag_add_0.txt",
                "hybrid_m_sigma_600_log_p_34.699_35.15_n_50_bag_add_0.txt")
"""

