import numpy as np

from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz

def solve_for_M_o(t_array, M_g_array, tau_dep, o_yield, tau_star):
    """
    Solve for the oxygen mass M_O(t), given gas mass M_g(t) as a function of time.

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    M_g_array : array_like
        1D array of gas mass values corresponding to `t_array`.
    tau_dep : float
        Gas depletion timescale (Gyr).
    o_yield : float
        IMF-integrated CCSN oxygen yield.
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).

    Returns
    -------
    M_O_array : ndarray
        Oxygen mass as a function of time, evaluated on `t_array`.
    
    """

    p_t = 1/tau_dep * np.ones(len(t_array))
    f_t = o_yield / tau_star * M_g_array
    integral_p = cumtrapz(p_t, t_array, initial=0)
    mu_t = np.exp(integral_p)
    integral_mu_f = cumtrapz(mu_t * f_t, t_array, initial=0)
    M_O_array = 1/mu_t * integral_mu_f
    return M_O_array


def solve_for_Z_o(t_array, M_g_array, tau_dep, o_yield, tau_star):
    """
    Solve for the oxygen abundance Z_O(t), given gas mass M_g(t) as a function of time.

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    M_g_array : array_like
        1D array of gas mass values corresponding to `t_array`.
    tau_dep : float
        Gas depletion timescale (Gyr).
    o_yield : float
        IMF-integrated CCSN oxygen yield.
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).

    Returns
    -------
    Z_O_array : ndarray
        Oxygen abundance as a function of time, evaluated on `t_array`.
    
    """

    p_t = 1/tau_dep * np.ones(len(t_array))
    f_t = o_yield / tau_star * M_g_array
    integral_p = cumtrapz(p_t, t_array, initial=0)
    mu_t = np.exp(integral_p)
    integral_mu_f = cumtrapz(mu_t * f_t, t_array, initial=0)
    M_O_array = 1/mu_t * integral_mu_f
    Z_O_array = M_O_array/M_g_array
    return Z_O_array



def Z_o_const_sfr(t_array, m_o_cc, eta, r, tau_dep):
    """
    Oxygen abundance Z(t), based on analytical solution for
    constant star formation rate (SFR). (eq. 34, Weinberg et al. 2017)

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    m_o_cc : float
        IMF-integrated CCSN oxygen yield.
    eta: float
        Outflow efficiency, eta = Mdot_outflow/M_dot_star.
    r: float
        Mass recycling parameter (CCSN + AGB).
    tau_dep: float
        Gas depletion timescale (Gyr).
    
    Returns
    -------
    Z_O: ndarray
        Oxygen abundance, calculated at 't_array'
    """
    Z_O = m_o_cc/(1 + eta - r) * (1 - np.exp(-t_array / tau_dep))
    return Z_O

def Z_o_exp_sfr(t_array, m_o_cc, eta, r, tau_dep, tau_star, tau_sfh):
    """
    Oxygen abundance Z(t), based on analytical solution for an
    exponentially declining star formation rate (SFR). (eq. 34, Weinberg et al. 2017)

    Parameters
    ----------
    t_array : array_like
        1D array of time values.
    m_o_cc : float
        IMF-integrated CCSN oxygen yield.
    eta: float
        Outflow efficiency, eta = Mdot_outflow/M_dot_star.
    r: float
        Mass recycling parameter (CCSN + AGB).
    tau_dep: float
        Gas depletion timescale (Gyr).
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).
    tau_sfh : float
        Exponential star formation history (SFH) timescale (Gyr).
    
    Returns
    -------
    Z_O: ndarray
        Oxygen abundance, calculated at 't_array'
    """
    Z_O_eq = Z_o_eq_exp(m_o_cc, eta, r, tau_star, tau_sfh)
    tau_bar = (tau_dep**-1 - tau_sfh**-1)**-1
    Z_O = Z_O_eq * (1 - np.exp(-1 * t_array/(tau_bar)))
    return Z_O


def Z_o_eq_exp(m_o_cc, eta, r, tau_star, tau_sfh):
    """
    Equilibrium oxygen abundance Z_eq, based on the analytical solution
    for a system with an exponentialy declining SFR.

    Parameters
    ----------
    m_o_cc : float
        IMF-integrated CCSN oxygen yield.
    eta : float
        Outflow efficiency, defined as eta = M_dot_outflow / M_dot_star.
    r : float
        Mass recycling parameter (CCSN + AGB).
    tau_star : float
        Star formation efficiency (SFE) timescale (Gyr).
    tau_sfh : float
        Exponential star formation history (SFH) timescale (Gyr).

    Returns
    -------
    Z_eq : float
        Equilibrium oxygen abundance.
    """
    return m_o_cc/(1 + eta - r - tau_star/tau_sfh)

def Z_o_eq_const(m_o_cc, eta, r):
    """
    Equilibrium oxygen abundance Z_eq, based on the analytical solution
    for a system with constant SFR.

    Parameters
    ----------
    m_o_cc : float
        IMF-integrated CCSN oxygen yield.
    eta : float
        Outflow efficiency, defined as eta = M_dot_outflow / M_dot_star.
    r : float
        Mass recycling parameter (CCSN + AGB).

    Returns
    -------
    Z_eq : float
        Equilibrium oxygen abundance.
    """
    return m_o_cc/(1 + eta - r)

