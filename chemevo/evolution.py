import numpy as np
from scipy.integrate import simpson as simps

class Galaxy:
    def __init__(self, t_array,
                 m_o_cc=0.015, eta=2.5, r=0.4, tau_star=1.0,
                 m_Fe_cc=0.0012, m_Fe_Ia=0.0017,
                 tau_sfh=6.0, tau_Ia=1.5, t_D=0.15, DTD_R0 = 2.2 * 10**(-3)):
        """
        Initialize the chemical evolution model.

        Parameters
        ----------
        t_array : np.ndarray
            Time array
        m_o_cc, eta, r, tau_star,
        m_Fe_cc, m_Fe_Ia, tau_sfh, tau_Ia, t_D : float | np.ndarray
            Either constants or arrays. If float, converted to array of ones.
        """
        self.t = t_array
        self.dt = t_array[1] - t_array[0]
        self.n_steps = len(t_array)
        self.DTD_R0 = DTD_R0

        # Core parameters
        self.m_o_cc_arr   = self._make_array(m_o_cc)
        self.eta_arr      = self._make_array(eta)
        self.r_arr        = self._make_array(r)
        self.tau_star_arr = self._make_array(tau_star)

        # Additional parameters for later use
        self.m_Fe_cc_arr  = self._make_array(m_Fe_cc)
        self.m_Fe_Ia_arr  = self._make_array(m_Fe_Ia)
        self.tau_sfh_arr  = self._make_array(tau_sfh)
        self.tau_Ia_arr   = self._make_array(tau_Ia)
        self.t_D = t_D

        # Placeholders for results
        self._m_O = None
        self._z_O = None
        self._m_Fe = None
        self._z_Fe = None
        self.tau_dep_arr = self.compute_tau_dep()

    def _make_array(self, param):
        """Convert a scalar into a constant array matching self.t."""
        if np.isscalar(param):
            return np.ones(self.n_steps) * param
        elif isinstance(param, np.ndarray):
            if len(param) != self.n_steps:
                raise ValueError("Array length must match time array length")
            return param
        else:
            raise TypeError("Parameter must be scalar or np.ndarray")
    
    def compute_tau_dep(self):
        """Compute depletion timescale: tau_dep = tau_star / (1 + eta - r)."""
        tau_dep = self.tau_star_arr / (1 + self.eta_arr - self.r_arr)
        self._tau_dep = tau_dep
        return tau_dep
    
    def compute_harmonic_diff_timescale(self, tau_x, tau_y):
        """compute harmonic difference timescale (WAF eq. 23)"""
        tau_hdt = (1/tau_x - 1/tau_y)**(-1)
        return tau_hdt

    def integrate_m_O(self, m_g_array):
        """
        Perform Euler integration for m_O.
        Parameters
        ----------
        m_g_array : np.ndarray
            Array of m_g values (same length as t)
        Returns
        -------
        m_O : np.ndarray
        """
        m_O = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m_O[i] = (
                m_O[i-1]
                + self.dt * (
                    (self.m_o_cc_arr[i-1] * m_g_array[i-1] / self.tau_star_arr[i-1])
                    - (m_O[i-1]/self.tau_dep_arr[i-1])
                )
            )
        self._m_O = m_O
        return m_O
    
    def compute_z_O(self, m_g_array):
        """Compute z_O = m_O / m_g."""
        if self._m_O is None:
            self.integrate_m_O(m_g_array)
        z_O = self._m_O / m_g_array
        self._z_O = z_O
        return z_O
    
    def DTD_exp(self):
        """Compute exponential DTD."""
        DTD_exp_array = self.DTD_R0 * np.exp(-(self.t - self.t_D)/self.tau_Ia_arr)
        DTD_exp_array[np.where(self.t < self.t_D)] = 0
        return DTD_exp_array
        
    def compute_mdotstar_Ia(self, m_g_array, r_t_array=None):
        if r_t_array is None:
            r_t_array = self.DTD_exp() 
        mdotstar_Ia = np.zeros(self.n_steps)
        r_t_inf = np.sum(r_t_array * self.dt)
        mdotstar = m_g_array/self.tau_star_arr
        for i in range(1, self.n_steps):
            for j in range(i):
                mdotstar_Ia[i] += (mdotstar[j] * r_t_array[i - j] * self.dt)/r_t_inf
        return mdotstar_Ia
    
    def integrate_m_Fe(self, m_g_array):
        m_Fe = np.zeros(self.n_steps)
        mdotstar_Ia = self.compute_mdotstar_Ia(m_g_array=m_g_array)
        for i in range(1, self.n_steps):
            m_Fe[i] = m_Fe[i-1] + self.dt*( (self.m_Fe_cc_arr[i-1] * m_g_array[i-1] / self.tau_star_arr[i-1])
                                           + (self.m_Fe_Ia_arr[i-1] * mdotstar_Ia[i-1])
                                            - m_Fe[i-1]/self.tau_dep_arr[i-1] )
        self._m_Fe = m_Fe
        return m_Fe
    
    def compute_z_Fe(self, m_g_array):
        if self._m_Fe is None:
            self.integrate_m_Fe(m_g_array)
        z_Fe = self._m_Fe / m_g_array
        self._z_Fe = z_Fe
        return z_Fe





    

