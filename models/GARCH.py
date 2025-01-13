import numpy as np
from scipy.optimize import minimize

class GARCH:
    def __init__(self):
        """
        Initialize the GARCH(1,1) model with default parameters.
        """
        self.params = None  # Placeholder for model parameters
        
    def initialize_params(self):
        """
        Initialize model parameters.
        """
        # GARCH(1,1) parameters: omega, alpha, beta
        self.params = {
            'omega': 0.1,  # Constant term
            'alpha': 0.1,  # ARCH parameter (coefficient for epsilon^2)
            'beta': 0.85,  # GARCH parameter (coefficient for previous variance)
        }
        
        print("Parameters initialized:", self.params)
    
    def get_params(self):
        return self.params
    
    def evaluate_likelihood(self, data):
        """
        Define the log likelihood function for GARCH(1,1).
        
        Args:
            data (array): time series data.
        
        Returns:
            float: negative log likelihood value.
        """
        omega = self.params['omega']
        alpha = self.params['alpha']
        beta = self.params['beta']
        T = len(data)
        
        # Initialize variables
        h = np.zeros(T)  # Conditional variances (volatilities squared)
        
        # Initial value for h (often set to variance of first data point)
        h[0] = max(np.var(data), 1e-6)
        log_likelihood = 0.0
        
        for t in range(1, T):
            # Calculate residuals
            epsilon = data[t-1]/h[t-1]  # Error term (returns here)
            h[t] = omega + beta *(h[t-1]) + alpha * (epsilon)**2
            log_likelihood += -0.5 * np.log(h[t]) - 0.5 * ((data[t])**2 / h[t]) - 0.5*np.log(2*np.pi)
        
        return -log_likelihood  # Return negative likelihood for minimization
    
    def fit(self, data):
        """
        Fit the GARCH(1,1) model to the data using Maximum Likelihood Estimation (MLE).
        
        Args:
            data (array-like): Time-series data to fit the model.
        
        Returns:
            dict: Optimized parameters.
        """
        # Define the objective function (negative log-likelihood)
        def objective(params):
            self.params['omega'], self.params['alpha'], self.params['beta']= params
            # print(params)
            return self.evaluate_likelihood(data)
        
        # Initial parameter values
        initial_params = list(self.params.values())
        
        # Optimize using 'L-BFGS-B' method with parameter bounds (omega > 0, alpha, beta >= 0)
        bounds = [(1e-6, 1000), (0, 1), (0, 1)]  # Bounds for omega, alpha, beta
        
        result = minimize(objective, initial_params, bounds=bounds, method='Nelder-Mead')
        
        optimized_params = result.x
        self.params['omega'], self.params['alpha'], self.params['beta']= optimized_params
        if result.success:
            print("Optimization successful. Parameters:", self.params)
        else:
            raise RuntimeError("Optimization failed:", result.message)
    
    def simulate(self, params=None, n=1000):
        """
        Simulate a time series using the fitted GARCH(1,1) model.
        
        Args:
            n (int): Length of the simulated time series.
        
        Returns:
            np.ndarray: Simulated time-series data.
        """
        
        if params is None:
            omega = self.params['omega']
            alpha = self.params['alpha']
            beta = self.params['beta']
        else:
            omega = params['omega']
            alpha = params['alpha']
            beta = params['beta']

        # Initialize arrays
        h = np.zeros(n)  # Conditional variances (volatilities squared)
        data = np.zeros(n)  # Simulated data
        
        # Initial values
        h[0] = np.var(data)  # Can initialize with sample variance
        data[0] = 0  # Assuming no initial shock
        epsilon = np.random.normal(0, 1)
        for t in range(1, n):
            # Update conditional variance (h_t)
            h[t] = omega + beta * h[t-1] + alpha * epsilon**2
            epsilon = np.random.normal(0, 1)
            # Simulate data (return) using the conditional variance
            data[t] = epsilon * np.sqrt(h[t])  # Simulated return (or log return)
        
        return data, h
