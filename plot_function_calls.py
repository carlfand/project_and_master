from mass_radius_plot import *

"""Mass-radius quark star cr plot."""
mrp_cr_arr_bag_min_all = [txt_to_arr("cr_m_simga_400_log_p_32.0_36.0_bag_1.8007.txt"),
                          txt_to_arr("cr_m_simga_500_log_p_32.0_36.0_bag_0.6894.txt"),
                          txt_to_arr("cr_m_simga_550_log_p_32.0_36.0_bag_0.0117.txt"),
                          txt_to_arr("cr_m_simga_600_log_p_32.0_36.0_bag_0.0081.txt")]
sigma_mass_labels = [400, 500, 550, 600]
plot_quark_star_cr(mrp_cr_arr_bag_min_all, sigma_mass_labels, "Mass-radius_quark_star_cr.eps")
plot_quark_star_cr(mrp_cr_arr_bag_min_all, sigma_mass_labels, "Mass-radius_quark_star_cr.svg")

"""Hybrid first-order transition, m_n = 900 MeV."""
# APR must be FIRST when we plot with hybrid_star_color_plot, and then 400, 500, 600 for the annotations to be correct
"""
mrp_hybrid = [txt_to_arr("pure_APR_log_p_32.3010_36_n_159.txt"),
              txt_to_arr("hybrid_m_sigma_400_log_p_34.5798_36.0_n_100_bag_add_0_merged.txt"),
              txt_to_arr("hybrid_m_sigma_500_log_p_34.5798_36.0_n_100_bag_add_0_merged.txt"),
              txt_to_arr("hybrid_m_sigma_600_log_p_34.6902_36.0_n_100_bag_add_0_merged.txt")]

print("m_sigma = 400, max mrp: {}".format(max(txt_to_arr("hybrid_m_sigma_400_log_p_34.5798_36.0_n_100_bag_add_0_merged.txt"), key=lambda triplet: triplet[0])))
print("m_sigma = 500, max mrp: {}".format(max(txt_to_arr("hybrid_m_sigma_500_log_p_34.5798_36.0_n_100_bag_add_0_merged.txt"), key=lambda triplet: triplet[0])))
print("m_sigma = 600, max mrp: {}".format(max(txt_to_arr("hybrid_m_sigma_600_log_p_34.6902_36.0_n_100_bag_add_0_merged.txt"), key=lambda triplet: triplet[0])))
hybrid_star_color_plot(mrp_hybrid, fname="Mass_radius_hybrid_stars.eps")
"""

"""Unified with n_APR = 2 n_0, n_q = 6 n_0. m_n = 939.6 MeV."""
"""
mrp_unified_unshifted = [txt_to_arr("pure_APR_log_p_32.0_35.447_n_160_unshifted.txt"),
                         txt_to_arr("Unified_m_sigma_400_log_p_33.367_36.0_n_75_unshifted.txt"),
                         txt_to_arr("Unified_m_sigma_500_log_p_33.367_36.0_n_75_unshifted.txt"),
                         txt_to_arr("Unified_m_sigma_600_log_p_33.367_36.0_n_75_unshifted.txt")]

print("m_sigma = 400, max mrp = {}".format(max(txt_to_arr("Unified_m_sigma_400_log_p_33.367_36.0_n_75_unshifted.txt"), key=lambda triplet: triplet[0])))
print("m_sigma = 500, max mrp = {}".format(max(txt_to_arr("Unified_m_sigma_500_log_p_33.367_36.0_n_75_unshifted.txt"), key=lambda triplet: triplet[0])))
print("m_sigma = 600, max mrp = {}".format(max(txt_to_arr("Unified_m_sigma_600_log_p_33.367_36.0_n_75_unshifted.txt"), key=lambda triplet: triplet[0])))
unified_star_colour_plot_non_shifted(mrp_unified_unshifted, fname="Mass-radius_unified_unshifted.eps")
"""
