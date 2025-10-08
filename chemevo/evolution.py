import numpy as np
from scipy.integrate import simpson as simps

class Galaxy:
    def __init__(self, t_array, m_g_array, DTD_func = "exp",
                 m_o_cc=0.015, eta=2.5, r=0.4, tau_star=1.0,
                 m_Fe_cc=0.0012, m_Fe_Ia=0.0017,
                 tau_sfh=6.0, tau_Ia=1.5, t_D=0.15, DTD_R0 = 1.47 * 10**(-3),
                 metallicity_dep_flag = 0, K_Fe_Ia = 0.77, tau_Ia_1 = 0.5, tau_Ia_2 = 5,
                 alpha_cc_Mn = 0.17, alpha_Ia_Mn = 0.30, R_Ia_Mn = 1.97):
        """
        Initialize the chemical evolution model.

        Parameters
        ----------
        t_array : np.ndarray
            Time array
        m_g_array : np.darray
            M_g array
        DTD_func : str
            options are "exp", "double-exp", "power-law"
        m_o_cc, eta, r, tau_star,
        m_Fe_cc, m_Fe_Ia, tau_sfh, tau_Ia, t_D : float | np.ndarray
            Either constants or arrays. If float, converted to array of ones.
        """
        self.t = t_array
        self.dt = t_array[1] - t_array[0]
        self.n_steps = len(t_array)
        self.DTD_R0 = DTD_R0
        self.m_g_array = m_g_array

        # Core parameters
        self.m_o_cc_arr   = self._make_array(m_o_cc)
        self.eta_arr      = self._make_array(eta)
        self.r_arr        = self._make_array(r)
        self.tau_star_arr = self._make_array(tau_star)

        # Additional parameters for later use
        self.m_Fe_cc_arr  = self._make_array(m_Fe_cc)
        self.tau_sfh_arr  = self._make_array(tau_sfh)
        self.tau_Ia_arr   = self._make_array(tau_Ia)
        self.tau_Ia_arr_1 = self._make_array(tau_Ia_1)
        self.tau_Ia_arr_2 = self._make_array(tau_Ia_2)
        self.t_D = t_D
        self.DTD_func = DTD_func

        #Params for Mn metallicity dependance relations
        self.alpha_cc_Mn = alpha_cc_Mn
        self.alpha_Ia_Mn = alpha_Ia_Mn
        self.R_Ia_Mn = R_Ia_Mn
        self.R_Ia_Fe = 1

        if metallicity_dep_flag == 0:
            self.K_Fe_Ia_arr = self._make_array(K_Fe_Ia)
            self.m_Fe_Ia_arr  = self._make_array(m_Fe_Ia)
        elif metallicity_dep_flag == 1:
            self.K_Fe_Ia_arr = K_Fe_Ia
            self.m_Fe_Ia_arr = self._make_array(self.integrate_iron_yield())
        else:
            raise ValueError("metallicity_dep_flag must be 0 or 1.")


        # Placeholders for results
        self._m_O = None
        self._z_O = None
        self._m_Fe = None
        self._z_Fe = None
        self.tau_dep_arr = self.compute_tau_dep()

        self.SolarO = 0.0056		# solar oxygen abundance by mass
        self.SolarFe = 0.0012		# solar iron abundance by mass
        self.SolarMn = 1.29e-05

        self.Fe_H = self.ratio_to_sun(star=self.compute_z_Fe(), sun=self.SolarFe)
        self.O_H = self.ratio_to_sun(star=self.compute_z_O(), sun=self.SolarO)
        self.Mn_H = self.ratio_to_sun(star=self.compute_z_Mn(), sun=self.SolarMn)
        self.Fe_O = self.Fe_H - self.O_H
        self.O_Fe = self.O_H - self.Fe_H
        self.Mn_O = self.Mn_H - self.O_H
        self.Mn_Fe = self.Mn_H - self.Fe_H
        self.O_Mn = self.O_H - self.Mn_H

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
    
    #Oxygen (O)

    def compute_z_O(self):
        """
        Perform Euler integration for m_O.
        Parameters
        ----------
        -------
        m_O : np.ndarray
        """
        m_O = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m_O[i] = (
                m_O[i-1]
                + self.dt * (
                    (self.m_o_cc_arr[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                    - (m_O[i-1]/self.tau_dep_arr[i-1])
                )
            )
        
        z_O = m_O / self.m_g_array

        return z_O
    
    
    def DTD_exp(self):
        """Compute exponential DTD."""
        DTD_exp_array = self.DTD_R0 * np.exp(-(self.t - self.t_D)/self.tau_Ia_arr)
        DTD_exp_array[np.where(self.t < self.t_D)] = 0
        return DTD_exp_array
    
    def DTD_double_exp(self):
        DTD_double_exp_array = self.DTD_R0 * (np.exp(-(self.t - self.t_D)/self.tau_Ia_arr_1) 
                                              + np.exp(-(self.t - self.t_D)/self.tau_Ia_arr_2))
        DTD_double_exp_array[np.where(self.t < self.t_D)] = 0
        return DTD_double_exp_array
    
    def DTD_power_law(self):
        self.t[0] = 1e-20
        #DTD_power_law_array = (2.2*10**-3)/12.5 * self.t**(-1.1)
        DTD_power_law_array = self.DTD_R0 * self.t**(-1.1)
        DTD_power_law_array[np.where(self.t < self.t_D)] = 0
        return DTD_power_law_array
    
    def DTD_linear_exp(self):
        DTD_lin_exp = self.DTD_R0 * self.t * np.exp(-(self.t - self.t_D)/self.tau_Ia_arr)
        DTD_lin_exp[np.where(self.t < self.t_D)] = 0
        return DTD_lin_exp
        
    
    def get_r_t(self):
        if self.DTD_func == "exp":
            r_t_array = self.DTD_exp()
        elif self.DTD_func == "double-exp":
            r_t_array = self.DTD_double_exp()
        elif self.DTD_func == "power-law":
            r_t_array = self.DTD_power_law()
        elif self.DTD_func == "linear-exp":
            r_t_array = self.DTD_linear_exp()
        else:
            raise ValueError("DTD function not found.") 
        return r_t_array

    #Iron (Fe)

    def compute_mdotstar_Ia(self, r_t_array=None):
        r_t_array = self.get_r_t()
        mdotstar_Ia = np.zeros(self.n_steps)
        r_t_inf = np.sum(r_t_array * self.dt)
        mdotstar = self.m_g_array/self.tau_star_arr
        for i in range(1, self.n_steps):
            for j in range(i):
                mdotstar_Ia[i] += (mdotstar[j] * r_t_array[i - j] * self.dt)/r_t_inf
        return mdotstar_Ia
    
    def integrate_iron_yield(self):
        r_t_array = self.get_r_t()
        m_Fe_Ia_met_dep = np.zeros(len(self.t))
        for i in range(1,len(self.t)):
            m_Fe_Ia_met_dep[i] = m_Fe_Ia_met_dep[i-1] + self.K_Fe_Ia_arr[i-1] * r_t_array[i-1] * self.dt
        return m_Fe_Ia_met_dep[-1]
    
    
    def compute_z_Fe(self):
        m_Fe = np.zeros(self.n_steps)
        mdotstar_Ia = self.compute_mdotstar_Ia()
        for i in range(1, self.n_steps):
            m_Fe[i] = m_Fe[i-1] + self.dt*( (self.m_Fe_cc_arr[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                                           + (self.m_Fe_Ia_arr[i-1] * mdotstar_Ia[i-1])
                                            - m_Fe[i-1]/self.tau_dep_arr[i-1] )
        self._m_Fe = m_Fe
        z_Fe = m_Fe/self.m_g_array
        return z_Fe
    
    
    #Mangenese (Mn)

    def get_m_Mn_cc(self):
        f_cc_O = 1
        f_cc_Mn = (1 + self.R_Ia_Mn)**-1
        m_Mn_cc = self.m_o_cc_arr * (self.SolarMn/self.SolarO) * 10**(self.alpha_cc_Mn * self.O_H) * f_cc_Mn/f_cc_O
        return m_Mn_cc
    
    def get_K_Mn_Ia(self):
        f_cc_Fe = (1 + self.R_Ia_Fe)**-1
        f_cc_Mn = (1 + self.R_Ia_Mn)**-1
        f_Ia_Fe = 1 - f_cc_Fe
        f_Ia_Mn = 1 - f_cc_Mn
        K_Mn_Ia = self.K_Fe_Ia_arr * (self.SolarMn/self.SolarFe) * (f_Ia_Mn/f_Ia_Fe) * 10**(self.alpha_Ia_Mn * self.O_H)
        return K_Mn_Ia

    def compute_z_Mn(self, r_t_array=None):
        r_t_array = self.get_r_t()
        K_Mn_Ia = self.get_K_Mn_Ia()
        m_Mn_cc = self.get_m_Mn_cc()
        mdotstar = self.m_g_array/self.tau_star_arr
        m_dot_Ia = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            for j in range(i):
                m_dot_Ia[i] += (K_Mn_Ia[j] * mdotstar[j] * r_t_array[i - j] * self.dt)
        
        m_Mn = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m_Mn[i] = m_Mn[i-1] + self.dt*( (m_Mn_cc[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                                           + m_dot_Ia[i-1]
                                            - m_Mn[i-1]/self.tau_dep_arr[i-1] )
        z_Mn = m_Mn/self.m_g_array
        return z_Mn
    
    #Analytic Solutions

    def analytic_eq_O(self, SFR_function):
        """
        Compute equilibrium oxygen abundance analytically (WAF eq. 21).
        """
        if SFR_function == "constant":
            Z_O_eq = self.m_o_cc_arr[0] / (1 + self.eta_arr[0] - self.r_arr[0])
        elif SFR_function == "exponential":
            Z_O_eq = self.m_o_cc_arr[0] / (1 + self.eta_arr[0] - self.r_arr[0] - self.tau_star_arr[0] / self.tau_sfh_arr[0])
        else:
            raise ValueError("SFR_function must be 'constant' or 'exponential'")
        return Z_O_eq
    
    def analytic_solutions_O(self, SFR_function):
        """
        Analytic solution for z_O(t).
        Only time is an array, all other parameters are scalars.
        """
        tau_dep = self.tau_dep_arr[0]   # scalar depletion timescale

        if SFR_function == "constant":
            Z_O_eq = self.analytic_eq_O("constant")
            z_O_analytic = Z_O_eq * (1 - np.exp(-self.t / tau_dep))
            return z_O_analytic

        elif SFR_function == "exponential":
            Z_O_eq = self.analytic_eq_O("exponential")
            tau_sfh = self.tau_sfh_arr[0]
            tau_dep_sfh = self.compute_harmonic_diff_timescale(tau_dep, tau_sfh)
            z_O_analytic = Z_O_eq * (1 - np.exp(-self.t / tau_dep_sfh))
            return z_O_analytic

        else:
            raise ValueError("SFR_function must be 'constant' or 'exponential'")
        

    def analytic_eq_Fe(self, SFR_function):
        if SFR_function == "constant":
            Z_Fe_eq = (self.m_Fe_cc_arr[0] + self.m_Fe_Ia_arr[0])/(1 + self.eta_arr[0] - self.r_arr[0])
            return Z_Fe_eq
        
        elif SFR_function == "exponential":
            tau_Ia_sfh = self.compute_harmonic_diff_timescale(self.tau_Ia_arr[0], self.tau_sfh_arr[0])
            tau_dep_sfh = self.compute_harmonic_diff_timescale(self.tau_dep_arr[0], self.tau_sfh_arr[0])
            Z_Fe_eq_cc = self.m_Fe_cc_arr[0] * tau_dep_sfh / self.tau_star_arr[0]
            Z_Fe_eq_Ia = self.m_Fe_Ia_arr[0] * (tau_dep_sfh/self.tau_star_arr[0]) * (tau_Ia_sfh/self.tau_Ia_arr[0]) * np.exp(self.t_D/self.tau_sfh_arr[0])
            Z_Fe_eq = Z_Fe_eq_cc + Z_Fe_eq_Ia
            return Z_Fe_eq, Z_Fe_eq_cc, Z_Fe_eq_Ia
        
        else:
            raise ValueError("SFR_function must be 'constant' or 'exponential'")
        


    def analytic_solutions_Fe(self, SFR_function, Z_type='all'):
        """
        Z_type = all:
        all three
        Z_type = cc:
        Z_type = Ia:
        Z_type = sum:
        """
        delta_t = self.t - self.t_D

        if SFR_function == "constant":
            tau_dep_Ia = self.compute_harmonic_diff_timescale(self.tau_dep_arr, self.tau_Ia_arr)
            Z_Fe_Ia_analytic = (self.m_Fe_Ia_arr / (1 + self.eta_arr - self.r_arr)) * (1 - np.exp(-delta_t / self.tau_dep_arr) - (tau_dep_Ia/self.tau_dep_arr) * (np.exp(-delta_t/self.tau_Ia_arr) - np.exp(-delta_t/self.tau_dep_arr)))
            Z_Fe_cc_analytic = (self.m_Fe_cc_arr / (1 + self.eta_arr - self.r_arr)) * (1 - np.exp(-self.t/self.tau_dep_arr))
            Z_Fe_Ia_analytic[self.t < self.t_D] = 0
            Z_Fe_analytic = Z_Fe_cc_analytic + Z_Fe_Ia_analytic

        elif SFR_function == "exponential":
            tau_dep_sfh = self.compute_harmonic_diff_timescale(self.tau_dep_arr, self.tau_sfh_arr)
            tau_dep_Ia = self.compute_harmonic_diff_timescale(self.tau_dep_arr, self.tau_Ia_arr)
            tau_Ia_sfh = self.compute_harmonic_diff_timescale(self.tau_Ia_arr, self.tau_sfh_arr)
            Z_Fe_eq_exp, Z_Fe_eq_cc_exp, Z_Fe_eq_Ia_exp = self.analytic_eq_Fe("exponential")
            Z_Fe_cc_analytic = Z_Fe_eq_cc_exp * (1 - np.exp(-self.t/tau_dep_sfh))
            Z_Fe_Ia_analytic = Z_Fe_eq_Ia_exp * (1 - np.exp(-delta_t/tau_dep_sfh) - (tau_dep_Ia/tau_dep_sfh) * (np.exp(-delta_t/tau_Ia_sfh) - np.exp(-delta_t/tau_dep_sfh)))
            Z_Fe_Ia_analytic[self.t < self.t_D] = 0
            Z_Fe_analytic = Z_Fe_cc_analytic + Z_Fe_Ia_analytic
        
        else:
            raise ValueError("SFR_function must be 'constant' or 'exponential'")

        if Z_type == "all":
            return Z_Fe_cc_analytic, Z_Fe_Ia_analytic, Z_Fe_analytic
        elif Z_type == "cc":
            return Z_Fe_cc_analytic
        elif Z_type == "Ia":
            return Z_Fe_Ia_analytic
        elif Z_type == "sum":
            return Z_Fe_analytic
        else:
            raise ValueError("Z_type not in list.")
        
    def ratio_to_sun(self, star, sun):
        ratio = np.log10(star/sun + 1e-6)
        #ratio[np.where(ratio <= -10)] = 0
        return ratio

    
    


    

