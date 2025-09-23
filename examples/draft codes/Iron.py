import numpy as np

from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import simpson as simps

def R_t(t_array, R_0, t_D, tau_Ia):
    """
    DTD of SNe Ia (eq. 1 in Weinberg et al. 2017)

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    R_0 : float
        Units of solar mass per year
    t_D: foat
        Minimum delay time for SN Ia (Gyr).
    tau_Ia:
        e-folding timescale of SN Ia DTD (Gyr).

    Returns
    -------
    R_t : ndarray
        SN Ia rate from population formed at t = 0 (Weinberg et al. 2017 eq. 1)
    """
    R_t = R_0 * np.exp(-(t_array - t_D)/tau_Ia)
    R_t[np.where(t_array < t_D)] = 0
    return R_t

def Mdot_avg_Ia(t, t_array, Mdot_array, R_0, t_D, tau_Ia):
    """
    Compute <Mdot_*(t)>_Ia (eq. 6 Weinberg et al. 2017)

    Parameters
    ----------
    t : float
        Current time (Gyr)
    t_array : ndarray
        Grid of times (Gyr)
    Mdot_array : ndarray
        Star formation history, Mdot_*(t) (same length as t_array)
    R_0, t_D, tau_Ia : floats
        Parameters for the DTD

    Returns
    -------
    Mdot_avg : float
        <Mdot_*(t)>_Ia at time t
    """
    # Ensure integration only up to t
    mask = t_array <= t
    t_prime = t_array[mask]
    Mdot_prime = Mdot_array[mask]

    # Convolution term: Mdot(t') * R(t - t')
    R_vals = R_t(t - t_prime, R_0, t_D, tau_Ia)
    numerator = simps(Mdot_prime * R_vals, t_prime)

    # Normalization: ∫ R(t') dt'
    t_grid = np.linspace(0, 20, 5000)  # large enough to approximate ∞
    denominator = simps(R_t(t_grid, R_0, t_D, tau_Ia), t_grid)

    return numerator / denominator


def solve_for_Z_Fe(t_array, M_g_array, tau_dep, m_Fe_Ia, m_Fe_cc, tau_star, R_0, t_D, tau_Ia):
    """
    Solve for the Iron abundance Z_O(t), given gas mass M_g(t) as a function of time.

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    M_g_array : array_like
        1D array of gas mass values corresponding to `t_array`.
    tau_dep : float
        Gas depletion timescale (Gyr).
    m_Fe_Ia : float
        IMF-integrated SN Ia Iron yield.
    m_Fe_cc : float
        IMF-integrated CCSN Iron yield.
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).
    R_0 : float
        Units of solar mass per year
    t_D: foat
        Minimum delay time for SN Ia (Gyr).
    tau_Ia:
        e-folding timescale of SN Ia DTD (Gyr).

    Returns
    -------
    Z_Fe_array : ndarray
        Iron abundance as a function of time, evaluated on `t_array`.
    
    """
    Mdot = M_g_array/tau_star
    M_dot_averages = np.zeros(len(t_array))
    for i in range(len(t_array)):
        M_dot_averages[i] = Mdot_avg_Ia(t_array[i], t_array, Mdot, R_0, t_D, tau_Ia)

    p_t = 1/tau_dep * np.ones(len(t_array))
    f_t = m_Fe_cc / tau_star * M_g_array + m_Fe_Ia * M_dot_averages

    integral_p = cumtrapz(p_t, t_array, initial=0)
    mu_t = np.exp(integral_p)
    integral_mu_f = cumtrapz(mu_t * f_t, t_array, initial=0)
    M_Fe_array = 1/mu_t * integral_mu_f
    Z_Fe_array = M_Fe_array/M_g_array
    return Z_Fe_array



def Z_Fe_const_sfr(t_array, m_Fe_Ia, m_Fe_cc, eta, r, tau_dep, t_D, tau_Ia):
    """
    Iron abundance Z(t), based on analytical solution for
    constant star formation rate (SFR). (eq. 37, Weinberg et al. 2017)

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    m_Fe_Ia : float
        IMF-integrated SN Ia Iron yield.
    eta: float
        Outflow efficiency, eta = Mdot_outflow/M_dot_star.
    r: float
        Mass recycling parameter (CCSN + AGB).
    tau_dep: float
        Gas depletion timescale (Gyr).
    t_D: foat
        Minimum delay time for SN Ia (Gyr).
    tau_Ia:
        e-folding timescale of SN Ia DTD (Gyr).
    
    Returns
    -------
    Z_Fe_Ia: ndarray
        Iron abundance from SN Ia
    Z_Fe_cc: ndarray
        Iron abundance from CCSN
    Z_Fe: ndarray
        Total Iron abundance, calculated at 't_array'
    """
    delta_t = t_array - t_D
    tau_dep_Ia = (1/tau_dep - 1/tau_Ia)**(-1)
    Z_Fe_Ia = (m_Fe_Ia / (1 + eta - r)) * (1 - np.exp(-delta_t / tau_dep) - (tau_dep_Ia/tau_dep) * (np.exp(-delta_t/tau_Ia) - np.exp(-delta_t/tau_dep)))
    Z_Fe_cc = (m_Fe_cc/ (1 + eta - r)) * (1 - np.exp(-t_array/tau_dep)) 
    Z_Fe = Z_Fe_cc + Z_Fe_Ia
    return Z_Fe_Ia, Z_Fe_cc, Z_Fe

def Z_Fe_exp_sfr(t_array, m_Fe_Ia, m_Fe_cc, t_D, tau_star, tau_dep, tau_sfh, tau_Ia):
    """
    Iron abundance Z(t), based on analytical solution for an
    exponentially declining star formation rate (SFR). (eq. 29 and 30, Weinberg et al. 2017)

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    m_Fe_Ia : float
        IMF-integrated SN Ia Iron yield.
    m_Fe_cc : float
        IMF-integrated CCSN Iron yield.
    t_D : float
        Minimum delay time for SN Ia (Gyr).
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).
    tau_dep : float
        Gas depletion timescale (Gyr).
    tau_sfh : float
        Exponential star formation history (SFH) timescale (Gyr).
    tau_Ia : float
        e-folding timescale of SN Ia DTD (Gyr).

    Returns
    -------
    Z_Fe_cc: ndarray
        Iron abundance due to CCSN
    Z_Fe_Ia: ndarray
        Iron abundance due to SN Ia
    Z_Fe_Ia + Z_Fe_cc: ndarray
        total Iron abundance
    """
    tau_dep_sfh = 1/(tau_dep**-1 - tau_sfh**-1)
    tau_dep_Ia = (1/tau_dep - 1/tau_Ia)**-1
    tau_Ia_sfh = (1/tau_Ia - 1/tau_sfh)**-1
    delta_t = t_array - t_D
    Z_Fe_cc_eq, Z_Fe_Ia_eq, Z_Fe_eq = Z_Fe_eq_exp(m_Fe_Ia, m_Fe_cc, t_D, tau_star, tau_dep, tau_sfh, tau_Ia)
    Z_Fe_cc = Z_Fe_cc_eq * (1 - np.exp(-t_array/(tau_dep_sfh)))
    Z_Fe_Ia = Z_Fe_Ia_eq * (1 - np.exp(-delta_t/tau_dep_sfh) - (tau_dep_Ia/tau_dep_sfh) * (np.exp(-delta_t/tau_Ia_sfh) - np.exp(-delta_t/tau_dep_sfh)))
    return Z_Fe_cc, Z_Fe_Ia, Z_Fe_cc + Z_Fe_Ia


def Z_Fe_eq_exp(m_Fe_Ia, m_Fe_cc, t_D, tau_star, tau_dep, tau_sfh, tau_Ia):
    """
    Equilibrium Iron abundance Z_eq, based on the analytical solution
    for a system with an exponentialy declining SFR.

    Parameters
    ----------
    m_Fe_Ia : float
        IMF-integrated SN Ia Iron yield.
    m_Fe_cc : float
        IMF-integrated CCSN Iron yield.
    t_D : float
        Minimum delay time for SN Ia (Gyr).
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).
    tau_dep : float
        Gas depletion timescale (Gyr).
    tau_sfh : float
        Exponential star formation history (SFH) timescale (Gyr).
    tau_Ia : float
        e-folding timescale of SN Ia DTD (Gyr).

    Returns
    -------
    Z_Fe_eq_cc : float
        Equilibrium Iron abundance due to CCSN
    Z_Fe_eq_Ia : float
        Equilibirum Iron abundance due to SN Ia
    Z_Fe_eq : float
        Total Iron abundance in equilibrium
    """
    tau_Ia_sfh = 1/(tau_Ia**-1 - tau_sfh**-1)
    tau_dep_sfh = 1/(tau_dep**-1 - tau_sfh**-1)

    Z_Fe_eq_cc = m_Fe_cc * tau_dep_sfh / tau_star
    Z_Fe_eq_Ia = m_Fe_Ia * (tau_dep_sfh/tau_star) * (tau_Ia_sfh/tau_Ia) * np.exp(t_D/tau_sfh)
    return Z_Fe_eq_cc, Z_Fe_eq_Ia, Z_Fe_eq_Ia + Z_Fe_eq_cc

def Z_Fe_eq_const(m_Fe_cc, m_Fe_Ia, eta, r):
    """
    Equilibrium Iron abundance Z_eq, based on the analytical solution
    for a system with constant SFR.

    Parameters
    ----------
    m_Fe_cc : float
        IMF-integrated CCSN Iron yield.
    m_Fe_Ia : float
        IMF-integrated Ia Iron yield.
    eta : float
        Outflow efficiency, defined as eta = M_dot_outflow / M_dot_star.
    r : float
        Mass recycling parameter (CCSN + AGB).

    Returns
    -------
    Z_eq : float
        Equilibrium Iron abundance.
    """
    return (m_Fe_cc + m_Fe_Ia)/(1 + eta - r)

