import numpy as np
from numba import njit

@njit
def fast_compute_mdotstar_Ia(n_steps, mdotstar, r_t_array, dt, r_t_inf):
    mdotstar_Ia = np.zeros(n_steps)
    for i in range(1, n_steps):
        for j in range(i):
            mdotstar_Ia[i] += (mdotstar[j] * r_t_array[i - j] * dt) / r_t_inf
    return mdotstar_Ia

class Galaxy:
    def __init__(self, t_array, m_g_array, DTD_func = "exp",
                 eta=2.5, r=0.4, tau_star=1.0,
                 tau_sfh=6.0, tau_Ia=1.5, t_D=0.15,
                 tau_Ia_1 = 0.5, tau_Ia_2 = 5.0,
                 solarvalues_ref = "W2024", yields_ref = "W2024",
                 alpha_cc_Mn = 0.17, alpha_Ia_Mn = 0.30,
                 g_cc_Mn = 0.345, g_ratio_Mn = 1.5, Upsilon = 2):
        """
        Initialize the chemical evolution model.

        Parameters
        ----------
        t_array : np.ndarray
            Time array
        m_g_array : np.darray
            M_g array
        DTD_func : str
            options are "exp", "double-exp", "power-law", "linear-exp"
        eta, r, tau_star, tau_sfh, tau_Ia, t_D : float | np.ndarray
            Either constants or arrays. If float, converted to array of ones.
        tau_Ia_1, tau_Ia_2 : float
            For double exp model, under development
        solarvalues_ref : string
            options are:
            W2024: From Weinberg et al. 2024 paper
            W2017: From Weinberg et al. 2017 paper
        yields_ref : string
            Selects a preset table of CCSNe/SNIa yields and the SNIa DTD
            normalization. See `_yields_preset` for the full list of options.
        """
        self.t = np.array(t_array, dtype=float)
        self.dt = self.t[1] - self.t[0]
        self.n_steps = len(self.t)
        self.m_g_array = m_g_array

        # Model parameters
        self.eta_arr      = self._make_array(eta)
        self.r_arr        = self._make_array(r)
        self.tau_star_arr = self._make_array(tau_star)

        #Met dep params
        self.alpha_cc_Mn = alpha_cc_Mn
        self.alpha_Ia_Mn = alpha_Ia_Mn
        self.g_cc_Mn = g_cc_Mn
        self.g_ratio_Mn = g_ratio_Mn
        self.Upsilon = Upsilon

        # Yields
        (self.m_O_cc_arr, self.m_Mg_cc_arr, self.m_Fe_cc_arr,
         self.m_Fe_Ia_arr, self.DTD_R0) = self._yields_preset(yields_ref, tau_Ia)

        self.tau_sfh_arr  = self._make_array(tau_sfh)
        self.tau_Ia_arr   = self._make_array(tau_Ia)
        self.tau_Ia_arr_1 = self._make_array(tau_Ia_1)
        self.tau_Ia_arr_2 = self._make_array(tau_Ia_2)
        self.t_D = t_D
        self.DTD_func = DTD_func

        # Placeholders for results
        self._m_O = None
        self._z_O = None
        self._m_Fe = None
        self._z_Fe = None
        self._mdotstar_Ia_cache = None
        self.tau_dep_arr = self.compute_tau_dep()
        self.Z_X_from_cc = None
        self.Z_X_from_Ia = None

        if solarvalues_ref == "W2024":
            self.SolarO = 73.3e-4
            self.SolarFe = 13.7e-4
            self.SolarMg = 6.71e-4
            self.SolarMn = 1.29e-05 #Magg et al. 2022
            self.SolarAl = 2.69e-6      #Magg et al. 2022

        elif solarvalues_ref == "W2017":
            self.SolarO = 0.0056		# solar oxygen abundance by mass
            self.SolarFe = 0.0012		# solar iron abundance by mass
            self.SolarMn = 1.29e-05     #Magg et al. 2022
            self.SolarMg = 6.71e-4      #W2024
            self.SolarAl = 2.69e-6      #Magg et al. 2022

        self.O_H = self.ratio_to_sun(star=self.compute_z_O(), sun=self.SolarO)
        self.Mg_H = self.ratio_to_sun(star=self.compute_z_Mg(), sun=self.SolarMg)
        self.Fe_H = self.ratio_to_sun(star=self.compute_z_Fe(), sun=self.SolarFe)
        self.Mn_H = self.ratio_to_sun(star=self.compute_z_X(element="Mn"), sun=self.SolarMn)
        self.T_H = self.ratio_to_sun(star=self.compute_z_T(alpha_cc_T=1), sun=self.SolarMn)

        self.Mn_H_CC = self.ratio_to_sun(star=self.compute_z_X_CC(element="Mn"), sun=self.SolarMn)
        self.Mn_H_Ia = self.ratio_to_sun(star=self.compute_z_X_Ia(element="Mn"), sun=self.SolarMn)
        self.Mn_Mg_CC = self.Mn_H_CC - self.Mg_H
        self.Mn_Mg_Ia = self.Mn_H_Ia - self.Mg_H
        self.Mn_Fe_CC = self.Mn_H_CC - self.Fe_H
        self.Mn_Fe_Ia = self.Mn_H_Ia - self.Fe_H


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

        self.T_O = self.T_H - self.O_H
        self.T_Mg = self.T_H - self.Mg_H

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

    def _yields_preset(self, yields_ref, tau_Ia):
        """
        Look up (m_O_cc_arr, m_Mg_cc_arr, m_Fe_cc_arr, m_Fe_Ia_arr, DTD_R0)
        for a named yields/DTD-normalization preset.
        """
        mk = self._make_array
        U = self.Upsilon
        R0_W2024 = 1.3e-3 / tau_Ia

        presets = {
            "W2024": lambda: (
                mk(71.3e-4) * U, mk(6.52e-4) * U, mk(4.73e-4) * U, mk(7.7e-4) * U, R0_W2024
            ),
            "W2017": lambda: (
                mk(0.015), mk(6.52e-4), mk(0.0012), mk(0.0017), 2.2e-3 / tau_Ia
            ),
            "W2024,double": lambda: (
                mk(71.3e-4) * 2, mk(6.52e-4) * 2, mk(4.73e-4) * 2, mk(7.7e-4) * 2, R0_W2024
            ),
            "W2024,1.3": lambda: (
                mk(71.3e-4) * 1.3, mk(6.52e-4) * 1.3, mk(4.73e-4) * 1.3, mk(7.7e-4) * 1.3, R0_W2024
            ),
            "W2024,moreFe": lambda: (
                mk(71.3e-4) * U, mk(6.52e-4) * U, mk(4.73e-4) * U, mk(7.7e-4) * 1.1 * U, R0_W2024
            ),
            "W2024,changed-plateau": lambda: (
                mk(71.3e-4) * U, mk(6.52e-4) * U,
                mk(4.73e-4) * U * 10**(-0.1), mk(7.7e-4) * 1.1 * U ** 10**(0.25),
                R0_W2024
            ),
            "W2024,moreFe30": lambda: (
                mk(71.3e-4) * U, mk(6.52e-4) * U, mk(4.73e-4) * U, mk(7.7e-4) * 1.3 * U, R0_W2024
            ),
            "W2024,0.7": lambda: (
                mk(71.3e-4) * 0.7, mk(6.52e-4) * 0.7, mk(4.73e-4) * 0.7, mk(7.7e-4) * 0.7, R0_W2024
            ),
            "W2024,CC-ONLY": lambda: (
                mk(71.3e-4), mk(6.52e-4), mk(4.73e-4), mk(0), R0_W2024
            ),
            "W2024,IA-ONLY": lambda: (
                mk(0), mk(0), mk(0), mk(7.7e-4), R0_W2024
            ),
            "sanders-test": lambda: (
                mk(0.011728) * U, mk(6.52e-4) * U, mk(4.73e-4) * U, mk(7.7e-4) * U, R0_W2024
            ),
        }
        if yields_ref not in presets:
            raise ValueError(f"Unknown yields_ref: {yields_ref!r}")
        return presets[yields_ref]()

    def compute_tau_dep(self):
        """Compute depletion timescale: tau_dep = tau_star / (1 + eta - r)."""
        tau_dep = self.tau_star_arr / (1 + self.eta_arr - self.r_arr)
        self._tau_dep = tau_dep
        return tau_dep

    def compute_harmonic_diff_timescale(self, tau_x, tau_y):
        """compute harmonic difference timescale (WAF eq. 23)"""
        tau_hdt = (1/tau_x - 1/tau_y)**(-1)
        return tau_hdt

    def _euler_integrate(self, source_arr):
        """
        Euler-integrate dm/dt = source - m/tau_dep.
        Returns the mass array m (not normalized by gas mass).
        """
        m = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m[i] = m[i-1] + self.dt * (source_arr[i-1] - m[i-1] / self.tau_dep_arr[i-1])
        return m

    def _cumulative_source(self, source_arr):
        """
        Running total of a source rate with no depletion term (i.e. total
        mass produced so far, ignoring any loss to outflows/star formation).
        """
        m = np.zeros(self.n_steps)
        for i in range(1, self.n_steps):
            m[i] = m[i-1] + self.dt * source_arr[i-1]
        return m

    #Oxygen (O)

    def compute_z_O(self):
        """
        Perform Euler integration for m_O.
        Parameters
        ----------
        -------
        m_O : np.ndarray
        """
        source = self.m_O_cc_arr * self.m_g_array / self.tau_star_arr
        m_O = self._euler_integrate(source)
        return m_O / self.m_g_array

    #Magnesium (Mg)
    def compute_z_Mg(self):
        """
        Perform Euler integration for m_Mg.
        Parameters
        ----------
        -------
        m_Mg : np.ndarray
        """
        source = self.m_Mg_cc_arr * self.m_g_array / self.tau_star_arr
        m_Mg = self._euler_integrate(source)
        return m_Mg / self.m_g_array

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
        # Avoid t=0 -> log/power blow-up. self.t is this instance's own
        # private copy (see __init__), so this can't leak into other
        # Galaxy objects that were built from the same input t_array.
        self.t[0] = 1e-20
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

    def compute_mdotstar_Ia(self):
        if self._mdotstar_Ia_cache is None:
            r_t_array = self.get_r_t()
            r_t_inf = np.sum(r_t_array * self.dt)
            mdotstar = self.m_g_array / self.tau_star_arr

            self._mdotstar_Ia_cache = fast_compute_mdotstar_Ia(
                self.n_steps,
                mdotstar,
                r_t_array,
                self.dt,
                r_t_inf
            )
        return self._mdotstar_Ia_cache

    def compute_z_Fe(self):
        mdotstar_Ia = self.compute_mdotstar_Ia()
        source_cc = self.m_Fe_cc_arr * self.m_g_array / self.tau_star_arr
        source_Ia = self.m_Fe_Ia_arr * mdotstar_Ia

        m_Fe = self._euler_integrate(source_cc + source_Ia)
        self.m_Fe_from_cc = self._cumulative_source(source_cc)
        self.m_Fe_from_Ia = self._cumulative_source(source_Ia)

        z_Fe = m_Fe/self.m_g_array
        return z_Fe

    def compute_z_Fe_CC(self):
        source = self.m_Fe_cc_arr * self.m_g_array / self.tau_star_arr
        self.m_Fe_from_cc = self._cumulative_source(source)
        z_Fe_CC = self.m_Fe_from_cc/self.m_g_array
        return z_Fe_CC

    def compute_z_Fe_Ia(self):
        mdotstar_Ia = self.compute_mdotstar_Ia()
        source = self.m_Fe_Ia_arr * mdotstar_Ia
        self.m_Fe_from_Ia = self._cumulative_source(source)
        z_Fe_Ia = self.m_Fe_from_Ia/self.m_g_array
        return z_Fe_Ia

    def _mn_yields(self, element):
        if element != "Mn":
            raise ValueError(f"element={element!r} is not supported (only 'Mn' is implemented)")
        return self.alpha_cc_Mn, self.alpha_Ia_Mn, self.SolarMn

    def compute_z_X(self, element = "Mn"):
        z_X_cc, z_X_Ia = self.compute_z_X_component(element)
        return z_X_cc + z_X_Ia

    def compute_z_X_CC(self, element = "Mn"):
        z_X_cc, _ = self.compute_z_X_component(element)
        return z_X_cc

    def compute_z_X_Ia(self, element = "Mn"):
        _, z_X_Ia = self.compute_z_X_component(element)
        return z_X_Ia

    def compute_y_Mn_Ia(self):
        alpha_Ia_X = self.alpha_Ia_Mn
        g_Ia_Mn = self.g_ratio_Mn * self.g_cc_Mn
        y_X_Ia = self.Upsilon * g_Ia_Mn * self.SolarMn * 10**(alpha_Ia_X * self.Mg_H)
        return y_X_Ia

    def compute_y_Mn_CC(self):
        alpha_cc_X = self.alpha_cc_Mn
        y_X_cc = self.Upsilon * self.g_cc_Mn * self.SolarMn * 10**(alpha_cc_X * self.Mg_H)
        return y_X_cc

    def compute_z_X_component(self, element = "Mn"):
        self._mn_yields(element)  # validates `element`

        y_X_cc = self.compute_y_Mn_CC()
        y_X_Ia = self.compute_y_Mn_Ia()

        mdotstar_Ia = self.compute_mdotstar_Ia()

        source_cc = y_X_cc * self.m_g_array / self.tau_star_arr
        source_Ia = y_X_Ia * mdotstar_Ia

        m_X_cc = self._euler_integrate(source_cc)
        m_X_Ia = self._euler_integrate(source_Ia)

        Z_X_cc = m_X_cc/self.m_g_array
        Z_X_Ia = m_X_Ia/self.m_g_array
        return Z_X_cc, Z_X_Ia

    def compute_z_T(self, alpha_cc_T=1, ): # A test element that is metallicity dependant but is only produced through CCSNe
        m_T_cc = self.Upsilon * self.g_cc_Mn * 2. * self.SolarO * 10**(alpha_cc_T * self.Mg_H)
        source = m_T_cc * self.m_g_array / self.tau_star_arr
        m_T = self._euler_integrate(source)
        return m_T / self.m_g_array

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
        ratio = np.log10(np.maximum(star/sun, 0) + 1e-6)
        return ratio


class useful_functions:
    def __init__(self):
        pass

    @staticmethod
    def get_Z_from_A(A, atomic_mass):
        log_Z = A - 12 + np.log10(atomic_mass) + np.log10(0.71)
        Z = 10**log_Z
        return Z
