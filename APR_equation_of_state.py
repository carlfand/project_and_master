import numpy as np
import csv
from scipy.interpolate import interp1d

"""This file requires that we download the equation of state data for APR matter from compose. 
URL: https://compose.obspm.fr/eos/68.
With this data, we may find the equation of state for APR-matter, and hence the equations of state for both
hybrid and unified stars."""


def read_APR_data_into_csv():
    """
    Data from compose is not stored as .csv. This function reads the contents from eos.thermo to a csv. The same is
    done for eos.nb.
    The first line in eos.thermo are not like the rest, it displays the neutron mass, the proton mass and an integer
    indication that the leptons have been taken into account. We remove it for our .csv-file.
    The data are separated by continuous whitespace.
    """
    eos_thermo = open("eos/eos.thermo", "r")
    # Create list of list containing floats from lines in the file.
    line_list = [[float(entry) for entry in line.strip().split()] for line in eos_thermo]
    eos_thermo_csv = open("eos/eos_thermo.csv", "w", newline="")
    writer = csv.writer(eos_thermo_csv)
    print("neutron mass {}, proton mass {}".format(line_list[0][0], line_list[0][1]))
    print(line_list[1])
    for line in line_list[1:]:
        writer.writerow(line)
    eos_thermo.close()
    eos_thermo_csv.close()

    # Similarly for n_b:
    eos_nb = open("eos/eos.nb", "r")
    # Create list of list containing floats from lines in the file.
    line_list = [[float(entry) for entry in line.strip().split()] for line in eos_nb]
    eos_nb_csv = open("eos/eos_nb.csv", "w", newline="")
    writer = csv.writer(eos_nb_csv)
    # Two header lines in eos, removing both
    for line in line_list[2:]:
        writer.writerow(line)
    eos_nb.close()
    eos_nb_csv.close()


def nuclear_matter_mu_B_n_B_p_eps():
    """Returns the APR equation of state for nuclear matter.
    Data fetched from https://compose.obspm.fr/eos/68.
    eos.nb contains the baryonic number density, n_B.
    eos.thermo contains 11 columns in each ordinary line.
    The first line states the neutron mass (m_n) and the proton mass (m_p) in MeV in the model used.
    They are 939.56533 MeV and 938.272 MeV, respectively.
    We are after n_B (baryonic density), eps ([internal] energy density), p (pressure) and
    mu_B (baryonic chemical potential). From the CompOSE document, the 4th column contains p/n_B [MeV],
    the 6th contains mu_B / m_n (dimensionless), and the 9th contains eps / (n_B m_n) - 1 (dimensionless).
    To retrieve the data, we have saved the contents as a .csv file, which we can read into a numpy-array.
    This procedure is done in read_APR_data_into_csv().
    Returns dimensionless quantities, where f_pi is the energy scale."""
    m_n = 939.56533     # Mass in MeV
    full_array = np.genfromtxt("eos/eos_thermo.csv", delimiter=",")
    n_Bs = np.genfromtxt("eos/eos_nb.csv")  # in units of fm^(-3)
    p_over_n_Bs, mu_Bs, eps_over_n_Bs = full_array[:, 3], (full_array[:, 5] + 1) * m_n, (full_array[:, 8] + 1) * m_n
    # ^In units of MeV
    ps, epss = p_over_n_Bs * n_Bs, eps_over_n_Bs * n_Bs     # Now in units of MeV / fm^3.
    # We would like to work in dimensionless units, p / f_pi^4, eps / f_pi^4, mu_B / f_pi, n_B / f_pi^3
    # We found that 1 unit of 1 / f_pi^4 -> 0.00974 GeV / fm^3, which means that
    # 1 MeV / fm^3 -> 0.103 / f_pi^4
    # Additionally, 1 unit of n / f_pi^3 -> 0.1047 / fm^3, which means that 1 / fm^3 -> 9.55 / f_pi^3
    return mu_Bs / 93, n_Bs * 9.55, ps * 0.103, epss * 0.103


def nuclear_matter_shifted_neutron_mass():
    """The same as the function above, except that we shift the neutron mass to 900 MeV."""
    m_n = 900    # Mass in MeV
    full_array = np.genfromtxt("eos/eos_thermo.csv", delimiter=",")
    n_Bs = np.genfromtxt("eos/eos_nb.csv")  # in units of fm^(-3)
    p_over_n_Bs, mu_Bs, eps_over_n_Bs = full_array[:, 3], (full_array[:, 5] + 1) * m_n, (full_array[:, 8] + 1) * m_n
    # ^In units of MeV
    ps, epss = p_over_n_Bs * n_Bs, eps_over_n_Bs * n_Bs  # Now in units of MeV / fm^3.
    # We would like to work in dimensionless units, p / f_pi^4, eps / f_pi^4, mu_B / f_pi, n_B / f_pi^3
    # We found that 1 unit of 1 / f_pi^4 -> 0.00974 GeV / fm^3, which means that
    # 1 MeV / fm^3 -> 0.103 / f_pi^4
    # Additionally, 1 unit of n / f_pi^3 -> 0.1047 / fm^3, which means that 1 / fm^3 -> 9.55 / f_pi^3
    return mu_Bs / 93, n_Bs * 9.55, ps * 0.103, epss * 0.103


def EoS_pure_APR(conversion_factor=1, shifted=True):
    """Reads APR data and returns a function epsilon(p)."""
    if shifted:
        m_n = 900    # Mass in MeV
        print("Shifted neutron mass: m_n = {} MeV".format(m_n))
    else:
        m_n = 939.6
        print("Unshifted neutron mass: m_n = {} MeV".format(m_n))
    full_array = np.genfromtxt("eos/eos_thermo.csv", delimiter=",")
    n_Bs = np.genfromtxt("eos/eos_nb.csv")  # in units of fm^(-3)
    p_over_n_Bs, mu_Bs, eps_over_n_Bs = full_array[:, 3], (full_array[:, 5] + 1) * m_n, (full_array[:, 8] + 1) * m_n
    # ^In units of MeV
    ps, epss = p_over_n_Bs * n_Bs, eps_over_n_Bs * n_Bs  # Now in units of MeV / fm^3.
    # We would like to work in dimensionless units, p / f_pi^4, eps / f_pi^4, mu_B / f_pi, n_B / f_pi^3
    # We found that 1 unit of 1 / f_pi^4 -> 0.00974 GeV / fm^3, which means that
    # 1 MeV / fm^3 -> 0.103 / f_pi^4
    # Additionally, 1 unit of n / f_pi^3 -> 0.1047 / fm^3, which means that 1 / fm^3 -> 9.55 / f_pi^3
    ps_scaled, epss_scaled = ps * 0.103 * conversion_factor, epss * 0.103 * conversion_factor
    # Need to add some negative pressures to avoid interpolaton range error.
    n_extend = 20   # Must extend ps into negative domain in order to handle negative pressures
    # when numrically integrating
    ps_extend = np.array([-ps_scaled[n_extend]])
    eps_extend = np.array([np.interp(-ps_scaled[n_extend], ps_scaled, epss_scaled)])
    print("(APR-EoS) Minimal pressure that is allowed to be reached in numerical integration: {} (converted units)".format(-ps_scaled[n_extend]))
    print("(APR-EoS) Maximal pressure that is allowed to be inserted in numerical integration: {} (converted units)".format(ps_scaled[-1]))
    return interp1d(np.concatenate((ps_extend, ps_scaled)), np.concatenate((eps_extend, epss_scaled)),
                    kind="linear", bounds_error=True, assume_sorted=True)
