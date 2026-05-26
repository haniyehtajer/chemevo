import numpy as np
from scipy.integrate import simpson as simps

class Galaxy:
    def __init__(self, t_array, m_g_array, DTD_func = "exp",
                 eta=2.5, r=0.4, tau_star=1.0,
                 tau_sfh=6.0, tau_Ia=1.5, t_D=0.15,
                 tau_Ia_1 = 0.5, tau_Ia_2 = 5.0,
                 R_Ia_Fe = 1, solarvalues_ref = "W2024", yields_ref = "W2024",
                 R_Ia_Mn = 1.97, alpha_cc_Mn = 0.17, alpha_Ia_Mn = 0.30, Mn_cc_scale = 1, Mn_Ia_scale = 1):
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
        m_O_cc, eta, r, tau_star,
        m_Fe_cc, m_Fe_Ia, tau_sfh, tau_Ia, t_D : float | np.ndarray
            Either constants or arrays. If float, converted to array of ones.
        DTD_R0 : float
            Normalization factor for DTD function
        K_Fe_Ia : float
            Mass of Iron made per SNIa explosion
        tau_Ia_1, tau_Ia_2 : float 
            For double exp model, under development
        R_Ia_Fe : float
            Relative SNIa contribution at solar metallicity
        solarvalues_ref : string
            options are:
            W2024: From Weinberg et al. 2024 paper
            W2017: From Weinberg et al. 2017 paper
        yields_ref : string
            optiones are:
            W2024: From Weinberg et al. 2024 paper
            W2017: From Weinberg et al. 2017 paper
        """
        self.t = t_array
        self.dt = t_array[1] - t_array[0]
        self.n_steps = len(t_array)
        self.m_g_array = m_g_array

        # Model parameters
        self.eta_arr      = self._make_array(eta)
        self.r_arr        = self._make_array(r)
        self.tau_star_arr = self._make_array(tau_star)

        # Yields
        if yields_ref == "W2024":
            self.m_O_cc_arr   = self._make_array(71.3e-4)
            self.m_Mg_cc_arr = self._make_array(6.52e-4)
            self.m_Fe_cc_arr  = self._make_array(4.73e-4)
            self.m_Fe_Ia_arr  = self._make_array(7.7e-4)
            self.K_Fe_Ia_arr = self._make_array(0.77)
            self.DTD_R0 = 1.3e-3/tau_Ia

        elif yields_ref == "W2017":
            self.m_O_cc_arr   = self._make_array(0.015)
            self.m_Mg_cc_arr = self._make_array(6.52e-4) #This is from W2024, maybe update later for W2017
            self.m_Fe_cc_arr  = self._make_array(0.0012)
            self.m_Fe_Ia_arr  = self._make_array(0.0017)
            self.K_Fe_Ia_arr = self._make_array(0.77)
            self.DTD_R0 = 2.2 * 10**(-3)/tau_Ia

        elif yields_ref == "W2024,double":
            self.m_O_cc_arr   = self._make_array(71.3e-4) * 2
            self.m_Mg_cc_arr = self._make_array(6.52e-4) * 2
            self.m_Fe_cc_arr  = self._make_array(4.73e-4) * 2
            self.m_Fe_Ia_arr  = self._make_array(7.7e-4) * 2
            self.K_Fe_Ia_arr = self._make_array(0.77)  * 2
            self.DTD_R0 = 1.3e-3/tau_Ia

        elif yields_ref == "W2024,1.3":
            self.m_O_cc_arr   = self._make_array(71.3e-4) * 1.3
            self.m_Mg_cc_arr = self._make_array(6.52e-4) * 1.3
            self.m_Fe_cc_arr  = self._make_array(4.73e-4) * 1.3
            self.m_Fe_Ia_arr  = self._make_array(7.7e-4) * 1.3
            self.K_Fe_Ia_arr = self._make_array(0.77) * 1.3
            self.DTD_R0 = 1.3e-3/tau_Ia

        elif yields_ref == "W2024,0.7":
            factor = 0.7
            self.m_O_cc_arr   = self._make_array(71.3e-4) * factor
            self.m_Mg_cc_arr = self._make_array(6.52e-4) * factor
            self.m_Fe_cc_arr  = self._make_array(4.73e-4) * factor
            self.m_Fe_Ia_arr  = self._make_array(7.7e-4) * factor
            self.K_Fe_Ia_arr = self._make_array(0.77) * factor
            self.DTD_R0 = 1.3e-3/tau_Ia

        elif yields_ref == "W2024,CC-ONLY":
            self.m_O_cc_arr   = self._make_array(71.3e-4)
            self.m_Mg_cc_arr = self._make_array(6.52e-4)
            self.m_Fe_cc_arr  = self._make_array(4.73e-4)
            self.m_Fe_Ia_arr  = self._make_array(0)
            self.K_Fe_Ia_arr = self._make_array(0)
            self.DTD_R0 = 1.3e-3/tau_Ia

        elif yields_ref == "W2024,IA-ONLY":
            self.m_O_cc_arr   = self._make_array(0)
            self.m_Mg_cc_arr = self._make_array(0)
            self.m_Fe_cc_arr  = self._make_array(0)
            self.m_Fe_Ia_arr  = self._make_array(7.7e-4)
            self.K_Fe_Ia_arr = self._make_array(0.77)
            self.DTD_R0 = 1.3e-3/tau_Ia

            
        self.Mn_cc_scale = Mn_cc_scale
        self.Mn_Ia_scale = Mn_Ia_scale

        self.tau_sfh_arr  = self._make_array(tau_sfh)
        self.tau_Ia_arr   = self._make_array(tau_Ia)
        self.tau_Ia_arr_1 = self._make_array(tau_Ia_1)
        self.tau_Ia_arr_2 = self._make_array(tau_Ia_2)
        self.t_D = t_D
        self.DTD_func = DTD_func

        #Extra params for Fe
        self.R_Ia_Fe = R_Ia_Fe 
        self.alpha_cc_Mn = alpha_cc_Mn
        self.alpha_Ia_Mn = alpha_Ia_Mn
        self.R_Ia_Mn = R_Ia_Mn


        # Placeholders for results
        self._m_O = None
        self._z_O = None
        self._m_Fe = None
        self._z_Fe = None
        self.tau_dep_arr = self.compute_tau_dep()
        self.Z_X_from_cc = None
        self.Z_X_from_Ia = None

        if solarvalues_ref == "W2024":
            self.SolarO = 73.3e-4
            self.SolarFe = 13.7e-4
            self.SolarMg = 6.71e-4
            self.SolarMn = 1.29e-05 #Magg et al. 2022

        elif solarvalues_ref == "W2017":
            self.SolarO = 0.0056		# solar oxygen abundance by mass
            self.SolarFe = 0.0012		# solar iron abundance by mass
            self.SolarMn = 1.29e-05     #Magg et al. 2022
            self.SolarMg = 6.71e-4      #W2024

        self.O_H = self.ratio_to_sun(star=self.compute_z_O(), sun=self.SolarO)
        self.Mg_H = self.ratio_to_sun(star=self.compute_z_Mg(), sun=self.SolarMg)
        self.Fe_H = self.ratio_to_sun(star=self.compute_z_Fe(), sun=self.SolarFe)

        self.Mn_H = self.ratio_to_sun(star=self.compute_z_X(element="Mn", cc_ref_element="O"), sun=self.SolarMn)
        self.Fe_O = self.Fe_H - self.O_H
        self.Fe_Mg = self.Fe_H - self.Mg_H
        self.O_Fe = self.O_H - self.Fe_H
        self.Mg_Fe = self.Mg_H - self.Fe_H
        self.Mn_O = self.Mn_H - self.O_H
        self.Mn_Mg = self.Mn_H - self.Mg_H
        self.Mn_Fe = self.Mn_H - self.Fe_H
        self.O_Mn = self.O_H - self.Mn_H
        self.Mg_Mn = self.Mg_H - self.Mn_H
        self.Mg_O = self.Mg_H - self.O_H
        self.O_Mg = self.O_H - self.Mg_H

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
                    (self.m_O_cc_arr[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                    - (m_O[i-1]/self.tau_dep_arr[i-1])
                )
            )
        
        z_O = m_O / self.m_g_array

        return z_O
    
    #Magnesium (Mg)
    def compute_z_Mg(self):
        """
        Perform Euler integration for m_Mg.
        Parameters
        ----------
        -------
        m_Mg : np.ndarray
        """
        m_Mg = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m_Mg[i] = (
                m_Mg[i-1]
                + self.dt * (
                    (self.m_Mg_cc_arr[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                    - (m_Mg[i-1]/self.tau_dep_arr[i-1])
                )
            )
        
        z_Mg = m_Mg / self.m_g_array

        return z_Mg
    
    
    def DTD_exp(self):
        """Compute exponential DTD."""
        DTD_exp_array = self.DTD_R0 * np.exp(-(self.t - self.t_D)/self.tau_Ia_arr)
        DTD_exp_array[np.where(self.t < self.t_D)] = 0
        return DTD_exp_array
    
    def DTD_double_exp(self):
        DTD_double_exp_array = self.DTD_R0 * (0.478 * np.exp(-(self.t - self.t_D)/self.tau_Ia_arr_2) 
                                              + 0.522 * np.exp(-(self.t - self.t_D)/self.tau_Ia_arr_1))
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
    
    
    def compute_z_Fe(self):
        m_Fe = np.zeros(self.n_steps)
        self.m_Fe_from_Ia = np.zeros(self.n_steps)
        self.m_Fe_from_cc = np.zeros(self.n_steps)
        mdotstar_Ia = self.compute_mdotstar_Ia()
        for i in range(1, self.n_steps):
            m_Fe[i] = m_Fe[i-1] + self.dt*( (self.m_Fe_cc_arr[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                                           + (self.m_Fe_Ia_arr[i-1] * mdotstar_Ia[i-1])
                                            - m_Fe[i-1]/self.tau_dep_arr[i-1] )
            self.m_Fe_from_cc[i] = self.m_Fe_from_cc[i-1] + self.dt * (self.m_Fe_cc_arr[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
            self.m_Fe_from_Ia[i] = self.m_Fe_from_Ia[i-1] + self.dt * (self.m_Fe_Ia_arr[i-1] * mdotstar_Ia[i-1])
        z_Fe = m_Fe/self.m_g_array
        return z_Fe
    
    
    def compute_z_X(self, element = "Mn", cc_ref_element = "O", Ia_ref_element = "Fe"):
        #Default values are for Iron      
        r_t_array = self.get_r_t()

        if element == "Mn":
            R_Ia_X = self.R_Ia_Mn
            alpha_cc_X = self.alpha_cc_Mn
            alpha_Ia_X = self.alpha_Ia_Mn
            SolarX = self.SolarMn
        
        elif element == "Fe":
            SolarX = self.SolarFe

        f_cc_O = 1
        f_cc_Mg = 1
        f_cc_Fe = (1 + self.R_Ia_Fe)**-1
        f_cc_X = (1 + R_Ia_X)**-1

        f_Ia_Fe = 1 - f_cc_Fe
        f_Ia_X = 1 - f_cc_X

        if Ia_ref_element == "Fe":
            K_X_Ia = self.K_Fe_Ia_arr * (SolarX/self.SolarFe) * (f_Ia_X/f_Ia_Fe) * 10**(alpha_Ia_X * self.O_H) * self.Mn_Ia_scale
        
        if cc_ref_element == "O":
            m_cc_ref_arr = self.m_O_cc_arr * self.Mn_cc_scale
            Solar_ref = self.SolarO
            f_cc_ref = f_cc_O

        elif cc_ref_element == "Fe":
            m_cc_ref_arr = self.m_Fe_cc_arr
            Solar_ref = self.SolarFe
            f_cc_ref = f_cc_Fe
        elif cc_ref_element == "Mg":
            m_cc_ref_arr = self.m_Mg_cc_arr
            Solar_ref = self.SolarMg
            f_cc_ref = f_cc_Mg

        m_X_cc = m_cc_ref_arr * (SolarX/Solar_ref) * 10**(alpha_cc_X * self.O_H) * f_cc_X/f_cc_ref
        mdotstar = self.m_g_array/self.tau_star_arr
        m_dot_Ia = np.zeros(self.n_steps)

        mass_X_from_cc = np.zeros(self.n_steps)
        mass_X_from_Ia = np.zeros(self.n_steps)

        for i in range(1, self.n_steps):
            for j in range(i):
                m_dot_Ia[i] += (K_X_Ia[j] * mdotstar[j] * r_t_array[i - j] * self.dt)
        
        m_X = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m_X[i] = m_X[i-1] + self.dt*( (m_X_cc[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
                                           + m_dot_Ia[i-1]
                                            - m_X[i-1]/self.tau_dep_arr[i-1] )
            mass_X_from_cc[i] = mass_X_from_cc[i-1] + self.dt * (m_X_cc[i-1] * self.m_g_array[i-1] / self.tau_star_arr[i-1])
            mass_X_from_Ia[i] = mass_X_from_Ia[i-1] + self.dt * m_dot_Ia[i-1]

        self.Z_X_from_cc = mass_X_from_cc/self.m_g_array
        self.Z_X_from_Ia = mass_X_from_Ia/self.m_g_array
        z_X = m_X/self.m_g_array
        return z_X
    
    #Analytic Solutions

    def analytic_eq_O(self, SFR_function):
        """
        Compute equilibrium oxygen abundance analytically (WAF eq. 21).
        """
        if SFR_function == "constant":
            Z_O_eq = self.m_O_cc_arr[0] / (1 + self.eta_arr[0] - self.r_arr[0])
        elif SFR_function == "exponential":
            Z_O_eq = self.m_O_cc_arr[0] / (1 + self.eta_arr[0] - self.r_arr[0] - self.tau_star_arr[0] / self.tau_sfh_arr[0])
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


class useful_functions:
    def __init__(self):
        pass

    def get_Z_from_A(A, atomic_mass):
        log_Z = A - 12 + np.log10(atomic_mass) + np.log10(0.71)
        Z = 10**log_Z
        return Z
    
    
