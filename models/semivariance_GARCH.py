import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class SemiGARCH:
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
        self.r = None
        self.bounds = {
            'lambda': (0,None), # ARCH parameter 0
            'alpha_s': (0,1),      # Long-run mean variance 2
            'beta_s': (0,1),      # Long-run mean variance 3
            'gamma_s': (0,None), # 4
            'alpha_q': (0,1),      # Long-run mean variance 5
            'beta_q': (0,1),      # Long-run mean variance 6
            'gamma_q': (0,None), #7
            'omega': (0,None),   # GARCH parameter 1
            'sigma2': (0,None),   # GARCH parameter 1
            'rho': (-1,1),   # GARCH parameter 1
        }
        # Set parameter constraints for optimization (e.g., ω > 0, α, β >= 0, etc.)
        self.constraints = (
            # {'type': 'ineq', 'fun': lambda params: params[5] - params[2] - params[1]*params[3]**2 + 1e-7},
            {'type': 'ineq', 'fun': lambda x: x[0] - 0},        # lambda >= 0
            {'type': 'ineq', 'fun': lambda x: x[1] - 0},        # alpha_s >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[1]},        # alpha_s <= 1
            {'type': 'ineq', 'fun': lambda x: x[2] - 0},        # beta_s >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[2]},        # beta_s <= 1
            {'type': 'ineq', 'fun': lambda x: x[3] - 0},        # gamma_s >= 0
            {'type': 'ineq', 'fun': lambda x: x[4] - 0},        # alpha_q >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[4]},        # alpha_q <= 1
            {'type': 'ineq', 'fun': lambda x: x[5] - 0},        # beta_q >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[5]},        # beta_q <= 1
            {'type': 'ineq', 'fun': lambda x: x[6] - 0},        # gamma_q >= 0
            {'type': 'ineq', 'fun': lambda x: x[7] - 0},        # omega >= 0
            {'type': 'ineq', 'fun': lambda x: x[8] - 0},        # sigma2 >= 0
            {'type': 'ineq', 'fun': lambda x: x[9] + 1},        # rho >= -1
            {'type': 'ineq', 'fun': lambda x: 1 - x[9]},        # rho <= 1
            # {'type': 'ineq', 'fun': lambda params: 1- np.abs(params[9])},
            # {'type': 'ineq', 'fun': lambda params: 1 - params[6] - params[5]*params[7]**2 },
            # {'type': 'ineq', 'fun': lambda params: params[1] - params[6] },
            # {'type': 'ineq', 'fun': lambda params: 1 - params[6] - params[5]*(1+params[0]**2)}
            )
    def initialize_params(self):
        """
        Initialize model parameters.
        This sets some default values, which can be tuned later during fitting.
        """
        # Example: Initialize parameters for ω, α, β, q (mean component)
        self.params = {
            'lambda': 2, # ARCH parameter 0
            'alpha_s': 1e-9,      # Long-run mean variance 2
            'beta_s': 0.7,      # Long-run mean variance 3
            'gamma_s': 4, # 4
            'alpha_q': 1e-6,      # Long-run mean variance 5
            'beta_q': 0.99,      # Long-run mean variance 6
            'gamma_q': 6, #7
            # 'omega': 1e-3,   # GARCH parameter 1
            'sigma2': 1e-3,   # GARCH parameter 1
            'rho': 0.03,   # GARCH parameter 1
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
    def simulate(self, T=1000, h0 = None, r=0.001):
        """
        Simulate the process for T time steps.
        Returns simulated returns (R), volatility (h), and realized volatility (h_RV).
        """
        lambda_ = self.params['lambda']
        alpha_s =self.params['alpha_s'] 
        beta_s = self.params['beta_s']
        gamma_s = self.params['gamma_s'] 
        alpha_q = self.params['alpha_q'] 
        beta_q = self.params['beta_q'] 
        gamma_q = self.params['gamma_q']
        omega = self.params['omega'] 
        sigma = self.params['sigma2']
        rho = self.params['rho']
        R = np.zeros(T)
        h = np.zeros(T)
        h_RV = np.zeros(T)
        RV_sim = np.zeros(T)
        v2 = np.zeros(T)
        # v1 = np.zeros(T)
        epsilon_1 = np.random.normal(size=T)
        epsilon_2 = np.random.normal(size=T)

        # Generate epsilon_2 with correlation rho with epsilon_1
        epsilon_2 = rho * epsilon_1 + np.sqrt(1 - rho**2) * epsilon_2

        # Initialize values
        if h0 is None:
            h[0] = 1e-6 # Initial volatility
        else:
            h[0] = h0
        h_RV[0] = alpha_s/(1-beta_s-alpha_s*gamma_s**2)  # Initial realized volatility

        # Simulate the process
        for t in range(1, T):
            v2[t] = (epsilon_1[t] - gamma_q * np.sqrt(np.abs(h[t-1])))**2 - (1 + gamma_q**2 * h[t-1])
            h_RV[t] = omega + beta_q * (h_RV[t-1] - omega) + alpha_q * v2[t]
            RV_sim[t] = h_RV[t] + sigma * (epsilon_2[t] - gamma_q * np.sqrt(np.abs(h[t-1])))**2 -\
                  (1 + gamma_q**2 * h[t-1])
            h[t] = h_RV[t] + beta_s * (h[t-1] - h_RV[t]) + alpha_s * (epsilon_1[t] - \
                gamma_s * np.sqrt(np.abs(h[t-1])))**2
            R[t] = r + (lambda_ - 0.5) * h[t] + np.sqrt(np.abs(h[t])) * epsilon_1[t]

        return R, h, RV_sim, h_RV

    def evaluate_likelihood(self, R, RVu):
        """
        Log-likelihood function for parameter estimation.
        We estimate the parameters by maximizing the likelihood.
        """
        # Simulate the model
        simulated_R, simulated_h, simulated_RV, simulated_h_RV = self.simulate(len(R))

        # Calculate the log-likelihood based on the simulated returns and observed values
        log_likelihood_R = -0.5 * np.sum(np.log(2 * np.pi * np.abs(simulated_h)) - 0.5*(R - simulated_R)**2 / simulated_h)
        log_likelihood_RVu = -0.5 * np.sum(np.log(2 * np.pi * np.abs(simulated_h_RV)) - 0.5*(RVu - simulated_RV)**2 \
                                           / simulated_h_RV)

        # Total log-likelihood is the sum of the two components
        return -(log_likelihood_R + log_likelihood_RVu)  # Return negative log-likelihood for minimization

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
            self.params['lambda'],self.params['alpha_s'] ,\
            self.params['beta_s'] ,self.params['gamma_s'] ,\
                self.params['alpha_q'] ,self.params['beta_q'] ,\
                    self.params['gamma_q'] ,self.params['omega'] ,\
                        self.params['sigma2'] ,self.params['rho'] = params
            # print(params)
            return -self.evaluate_likelihood(data,RV)
        
        # Initial parameter values
        initial_params = list(self.params.values())
        
        # Parameter bounds
        bounds = list(self.bounds.values())
        options = {
            'maxiter': 1e6,  # Set maximum iterations
            'disp': True     # Display optimization details
        }

        # Optimize
        result = minimize(objective, initial_params, method='SLSQP',\
                          options = options,\
                             constraints = self.constraints)
        
        # if result.success:
        optimized_params = result.x
        self.params['lambda'],self.params['alpha_s'] ,\
            self.params['beta_s'] ,self.params['gamma_s'] ,\
                self.params['alpha_q'] ,self.params['beta_q'] ,\
                    self.params['gamma_q'] ,self.params['omega'] ,\
                        self.params['sigma2'] ,self.params['rho']  = optimized_params
        #     print("Optimization successful. Parameters:", self.params)
        # else:
        #     raise RuntimeError("Optimization failed:", result.message)
        return result
    
    def rolling_window_simulate(self, R, RVu, window_size=100, step_size=1):
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
        simulated_h_RV = np.zeros(len(R))
        simulated_RV = np.zeros(len(R))

        # Loop through the rolling window
        for start in range(0, num_steps, step_size):
            window_RVu = RVu[start:start + window_size]
            R_simulated, h_simulated, RV_simulated, h_RV_simulated = self.simulate(T=len(R),h0=window_RVu[-1])
            # Update the simulated values from the model
            simulated_R[start + window_size] = R_simulated[start + window_size]
            simulated_h[start + window_size] = h_simulated[start + window_size]
            simulated_h_RV[start + window_size] = h_RV_simulated[start + window_size]
            simulated_RV[start + window_size] = RV_simulated[start + window_size]

        return simulated_R, simulated_h, simulated_h_RV
    


