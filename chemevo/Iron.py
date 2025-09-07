import numpy as np

from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz


def solve_for_Z_Fe(t_array, M_g_array, tau_dep, o_yield, tau_star):
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
    o_yield : float
        IMF-integrated CCSN Iron yield.
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).

    Returns
    -------
    Z_O_array : ndarray
        Iron abundance as a function of time, evaluated on `t_array`.
    
    """

    p_t = 1/tau_dep * np.ones(len(t_array))
    f_t = o_yield / tau_star * M_g_array
    integral_p = cumtrapz(p_t, t_array, initial=0)
    mu_t = np.exp(integral_p)
    integral_mu_f = cumtrapz(mu_t * f_t, t_array, initial=0)
    M_O_array = 1/mu_t * integral_mu_f
    Z_O_array = M_O_array/M_g_array
    return Z_O_array



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

