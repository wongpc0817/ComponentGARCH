import numpy as np
from models.utils import transform_p, reverse_p
class RVGARCH:
    def __init__(self, lambda_param=0):
        """
        Initialize the Component GARCH model with default or provided parameters.
        
        Args:
            lambda_param (float): A model-specific parameter. Default is 0.
        """
        self.lambda_param = lambda_param  # Avoid using `lambda` as it's a reserved keyword.
        self.params = None  # Placeholder for model parameters
        self.h = None
        self.q = None
        self.w_sn = None
        self.w_qn = None

        self.bounds = {
            'lambda': (0,None), # ARCH parameter 0
            'sigma2': (0,1),   # GARCH parameter 1
            'alpha_s': (0,1),      # Long-run mean variance 2
            'beta_s': (0.7,1),      # Long-run mean variance 3
            'gamma_s': (0,None), # 4
            'alpha_q': (0,1),      # Long-run mean variance 5
            'beta_q': (0.7,1),      # Long-run mean variance 6
            'gamma_q': (0,None) #7
        }
        # Set parameter constraints for optimization (e.g., ω > 0, α, β >= 0, etc.)
        self.constraints = (
            {'type': 'ineq', 'fun': lambda params: params[6] - params[3] - params[2]*params[4]**2 + 1e-7 },
            # {'type': 'ineq', 'fun': lambda params: 1 - params[6] - params[5]*params[7]**2 },
            # {'type': 'ineq', 'fun': lambda params: 1 - params[1] - params[6] },
            # {'type': 'ineq', 'fun': lambda params: 1 - params[6] - params[5]*(1+params[0]**2)}
            )
        
    def initialize_params(self):
        """
        Initialize model parameters.
        This sets some default values, which can be tuned later during fitting.
        """
        # Example: Initialize parameters for ω, α, β, q (mean component)
        self.params = {
            # 'r': 0.1,  # GARCH constant
            'lambda': 1, # ARCH parameter
            'sigma2': 0.003,   # GARCH parameter
            'alpha_s': 1e-7,      # Long-run mean variance
            'beta_s': 0.7,      # Long-run mean variance
            'gamma_s': 1,
            'alpha_q': 1e-7,      # Long-run mean variance
            'beta_q': 0.8,      # Long-run mean variance
            'gamma_q': 1
        }
        
        self.check_feasibilty(list(self.params.values()))
        print("Parameters initialized:", self.params)
    def get_params(self):
        return self.params
    def get_history(self):
        return self.h, self.q
    def check_feasibilty(self, params):
        for i, con in enumerate(self.constraints):
            if con['fun'](params) < 0:
                raise ValueError(f"Initial parameters violate constraint {i + 1}")
    def set_params(self, params):
        self.check_feasibilty(list(params.values()))
        self.params = params

    def set_params_data(self, data):
        self.params['r'] = np.mean(data)

    def evaluate_likelihood(self, data, RV):
        """
        Define the log likelihood function.
        
        Args:
            data (array): time steps.

        Returns:
            float: log likelihood value.

        """
        # r = self.params['r']
        lam = self.params['lambda']
        sigma2 = self.params['sigma2']
        beta_s = self.params['beta_s']
        alpha_s = self.params['alpha_s']
        gamma_s = self.params['gamma_s']
        beta_q = self.params['beta_q']
        alpha_q = self.params['alpha_q']
        gamma_q = self.params['gamma_q']

        T = data.shape[0]
        
        # Initialize variables
        h = np.zeros(T)
        q = np.zeros(T)
        v_s = np.zeros(T)
        v_q = np.zeros(T)
        r = 0.01
        # Initial values
        # sigma2 = np.var(data)
        h[0] = max(np.var(data), 1e-6)
        # h[0] = alpha_s/(1-beta_s-alpha_s*gamma_s**2)
        # q[0] = alpha_s/(1-beta_s-alpha_s*gamma_s**2)
        q[0] = RV[0]
        log_likelihood = 0.0
        
        for t in range(1, T):
            # Compute epsilon
            h_tmp = h[t-1]
            if h_tmp < 1e-5:
                epsilon = 0
            else:
                epsilon = (data[t-1] - r - (lam - 0.5)*(h_tmp))/np.sqrt(np.abs(h_tmp))
            # Compute innovations for short-run and long-run components
            v_s[t-1] = epsilon**2 - 1 -2*gamma_s*np.sqrt(np.abs(h_tmp))*epsilon
            v_q[t-1] = epsilon**2 - 1 -2*gamma_q*np.sqrt(np.abs(h_tmp))*epsilon
            # Update the conditional variances
            q[t] = sigma2 + beta_q * (q[t-1] - sigma2) + alpha_q * v_q[t-1]
            # print("q update",q[t-1] - sigma2, v_q[t])
            h[t] = q[t] + beta_s * (h_tmp - q[t-1]) + alpha_s * v_s[t-1] 
            # Log-likelihood contribution for the current time point
            log_likelihood += -0.5 * np.log(np.abs(h[t])) - 0.5 * ((data[t]-r-(lam-0.5)*h[t])**2 / h[t]) \
                - np.log(2*np.pi)*0.5
            log_likelihood -= np.mean((RV-q)**2)
        
        return -log_likelihood  # Negative because we minimize


    
    def fit(self, data, RV):
        """
        Fit the Two Component GARCH model to the data using quasi-MLE.
        
        Args:
            data (array-like): Time-series data to fit the model.
        
        Returns:
            dict: Optimized parameters.
        """
        from scipy.optimize import minimize
        
        # Define the objective function (negative log-likelihood)
        def objective(params):
            self.params['lambda'], self.params['sigma2'],self.params['alpha_s'], \
            self.params['beta_s'], self.params['gamma_s'],\
            self.params['alpha_q'], self.params['beta_q'],self.params['gamma_q'] = params
            # print(params)
            return -self.evaluate_likelihood(data, RV)
        
        # Initial parameter values
        initial_params = list(self.params.values())
        # initial_params = list(self.params.values())
        
        # Parameter bounds
        bounds = list(self.bounds.values())
        options = {
            'maxiter': 100,  # Set maximum iterations
            'disp': True     # Display optimization details
        }

        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='SLSQP',constraints = self.constraints,
                          options = options)
        
        
        # if result.success:
        optimized_params = result.x
        self.params['lambda'], self.params['sigma2'],self.params['alpha_s'], \
        self.params['beta_s'], self.params['gamma_s'],\
        self.params['alpha_q'], self.params['beta_q'],self.params['gamma_q']  = optimized_params
        #     print("Optimization successful. Parameters:", self.params)
        # else:
        #     raise RuntimeError("Optimization failed:", result.message)
        return result
    
    def simulate(self , params=None, n=1000,r = 0.01, current=0, h0 = None):
        """
        Simulate a time series using the fitted Component GARCH model.
        
        Args:
            n (int): Length of the simulated time series.
        
        Returns:
            np.ndarray: Simulated time-series data.
        """
        if params is None:
            lam = self.params['lambda']
            sigma2 = self.params['sigma2']
            alpha_s = self.params['alpha_s']
            beta_s = self.params['beta_s']
            gamma_s = self.params['gamma_s']
            alpha_q = self.params['alpha_q']
            beta_q = self.params['beta_q']
            gamma_q = self.params['gamma_q']
        else:
            lam = params['lambda']
            sigma2 = params['sigma2']
            alpha_s = params['alpha_s']
            beta_s = params['beta_s']
            gamma_s = params['gamma_s']
            alpha_q = params['alpha_q']
            beta_q = params['beta_q']
            gamma_q = params['gamma_q']

        
        h = np.zeros(n)
        q = np.zeros(n)
        v_s = np.zeros(n)
        v_q = np.zeros(n)
        
        data = np.random.normal(0,1,n)
        data[0]=current

        # Initial values
        if h0 is None:
            h[0] = sigma2/(1-beta_q)
        else:
            h[0] = 0
        q[0] = alpha_s/(1-beta_s-alpha_s*gamma_s**2)
        epsilon = np.random.normal(0,1)
        for t in range(1, n):
            v_s[t-1] = epsilon**2 - 1 -2*gamma_s*np.sqrt(np.abs(h[t-1]))*epsilon
            v_q[t-1] = epsilon**2 - 1 -2*gamma_q*np.sqrt(np.abs(h[t-1]))*epsilon
            q[t] = sigma2 + beta_q * (q[t-1]-sigma2) + alpha_q * v_q[t-1]
            h[t] = q[t] + beta_s * (h[t-1] - q[t-1]) + alpha_s * v_s[t-1]
            epsilon = np.random.normal(0,1)
            data[t] = r + (lam -0.5)* h[t] +np.abs(h[t])**0.5 * epsilon
        self.h = h
        self.q = q
        return data, h, q
    
    def risk_neutral_parameters(self):
        alpha_q = self.params['alpha_q']
        alpha_s = self.params['alpha_s']
        beta_q = self.params['beta_q']
        beta_s = self.params['beta_s']
        gamma_q = self.params['gamma_q']
        gamma_s = self.params['gamma_s']
        lambda_ = self.params['lambda']
        sigma2 = self.params['sigma2']
        gamma_s_rn = lambda_ + gamma_s
        gamma_q_rn = lambda_ + gamma_q
        beta_s_rn = beta_s + alpha_s * (gamma_s_rn**2 - gamma_s**2) + \
                    alpha_q * (gamma_q_rn**2 - gamma_q**2)
        beta_q_rn = beta_q + alpha_s * (gamma_s_rn**2 - gamma_s**2) + \
                    alpha_q * (gamma_q_rn**2 - gamma_q**2)
        sigma2_rn = sigma2 * (1-beta_q)/(1-beta_q_rn)
        self.params.update({
            "gamma_s_rn": gamma_s_rn,
            "gamma_q_rn": gamma_q_rn,
            "beta_s_rn": beta_s_rn,
            "beta_q_rn": beta_q_rn,
            "sigma2_rn": sigma2_rn,
        })

    def rolling_window_simulate(self, R, window_size=100, step_size=1):
        """
        Perform rolling window simulation of the model.
        
        Parameters:
            R (numpy array): The observed returns.
            RVu (numpy array): The observed realized volatility.
            window_size (int): The size of the rolling window.
            step_size (int): The step size by which the window rolls forward.
        
        Returns:
            simulated_R (numpy array): Simulated returns from the rolling window.
            simulated_h (numpy array): Simulated volatilities from the rolling window.
            simulated_h_RV (numpy array): Simulated realized volatilities from the rolling window.
        """
        num_steps = len(R) - window_size
        simulated_R = np.zeros(len(R))
        simulated_h = np.zeros(len(R))
        simulated_q = np.zeros(len(R))

        # Loop through the rolling window
        for start in range(0, num_steps, step_size):
            window_R = R[start:start + window_size]
            data, h, q = self.simulate(n=len(R),h0=np.std(window_R))
            # Update the simulated values from the model
            simulated_R[start + window_size] = data[start + window_size]
            simulated_h[start + window_size] = h[start + window_size]
            simulated_q[start + window_size] = q[start + window_size]

        return simulated_R, simulated_h, simulated_q
    ######## Options 
    def get_vix(self, n=22, trading_days=252, t=0):
        if self.h is None:
            pred, self.h , self.q = self.simulate()
        self.risk_neutral_parameters()
        beta_q_rn = self.params['beta_q_rn']
        beta_s_rn = self.params['beta_s_rn']
        omega_rn = self.params['sigma2_rn']

        w_sn = (1-beta_s_rn**n)/(1-beta_s_rn)/n
        w_qn = (1-beta_q_rn**n)/(1-beta_q_rn)/n
        short_var = w_sn * (self.h[t+1] - self.q[t+1])
        long_var = w_qn * self.q[t+1] + (1-w_qn)* omega_rn
        self.w_sn = w_sn
        self.w_qn = w_qn
        return np.sqrt((short_var + long_var) * trading_days) * 100
    def mgf_terms(self, phis, phiq , T):
        self.risk_neutral_parameters()
        # Initialize the values for the final time step T
        A_T = 0
        B1_T = phis
        B2_T = phiq
        
        # Prepare to store results for each time step
        t_values = []
        B1_values = []
        B2_values = []
        alpha_q = self.params['alpha_q']
        alpha_s = self.params['alpha_s']
        lambda_ = self.params['lambda']
        beta_q = self.params['beta_q']
        beta_s = self.params['beta_s']
        gamma2 = self.params['gamma_q']
        gamma1 = self.params['gamma_s']
        omega = self.params['sigma2']
        beta_q_rn = self.params['beta_q_rn']
        beta_s_rn = self.params['beta_s_rn']
        gamma2_rn = self.params['gamma_q_rn']
        gamma1_rn = self.params['gamma_s_rn']
        omega_rn = self.params['sigma2_rn']
        # Loop backwards from T-1 to 0 (T is already initialized)
        for _ in range(T):
            # Recursive calculation for t(phi)
            A_phi = A_T + B2_T * omega * (1-beta_q_rn) - 0.5 * np.log(1 - 2 * (alpha_s * B1_T + alpha_q * B2_T))\
                - (alpha_s* B1_T + alpha_q* B2_T )
            
            # Recursive calculation for B1(phi)
            B1_phi = B1_T * beta_s_rn + 2* (alpha_s * gamma1_rn*B1_T + alpha_q * gamma2_rn * B2_T)**2\
            / (1-2* (alpha_s * B1_T + alpha_q * B2_T))
            
            B2_phi = B2_T * beta_q_rn + 2* (alpha_s * gamma1_rn*B1_T + alpha_q * gamma2_rn * B2_T)**2\
            / (1-2* (alpha_s * B1_T + alpha_q * B2_T))
            
            # Update the values for the previous time step
            A_T = A_phi
            B1_T = B1_phi
            B2_T = B2_phi
            
            # Store the results for this time step
            t_values.append(A_phi)
            B1_values.append(B1_phi)
            B2_values.append(B2_phi)
        
        return t_values, B1_values, B2_values
    
    def mgf(self, phis, phiq, m):
        a_term, b_term, c_term = self.mgf_terms(phis, phiq, m)
        if self.h is None:
            _, self.h , self.q = self.simulate()
        return np.exp(a_term[1] + b_term[1]*(self.h[1]-self.q[1]) + c_term[1]*self.q[1])


    def vix_futures(self,T, t=0):
        if self.h is None:
            _, self.h , self.q = self.simulate()
        if self.w_qn is None:
            self.get_vix()
        omega_rn = self.params['sigma2_rn']

        a = 252 * (1-self.w_qn) * omega_rn
        bq = 252 * self.w_qn
        bs = 252 * self.w_sn
        print(a, bq, bs)
        def integrand(x, a, bs, bq, T, t, f):
            # print(f(-bs*x,-bq*x,T-t))
            return (1-np.exp(- a * x)* f(-bs*x,-bq*x,T-t))/x**(3/2) 
        from scipy.integrate import quad
        integral_val, _ = quad(integrand, 0, np.inf, args=(a, bs, bq, T, t, self.mgf))

        return integral_val * 100 * (2/np.sqrt(np.pi))