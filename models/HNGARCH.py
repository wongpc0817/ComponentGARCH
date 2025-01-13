import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
class HNGARCH:
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
            'omega': 0.0001,  # Constant term
            'alpha': 0.001,  # ARCH parameter (coefficient for epsilon^2)
            'beta': 0.9,  # GARCH parameter (coefficient for previous variance)
            'lambda': 1,
            'gamma': 0.1
        }
        self.constraints = [{
          'type': 'ineq', 'fun': lambda params: params[2]- params[1]*params[4]**2
        }]
        self.check_feasibilty(list(self.params.values()))
        print("Parameters initialized:", self.params)

    def get_stat(self, data, r=0.001):
        model_name = "HNGARCH"
        import scipy.stats as stats
        window_size = 10
        cumulative_variance = pd.Series((data**2).cumsum())
        realized_variance = cumulative_variance - cumulative_variance.shift(window_size, fill_value=0)
        pred_data, pred_h = self.simulate(n=data.shape[0])
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        omega = self.params['omega']
        lambda_ = self.params['lambda']
        print(f"Persistence: {beta + alpha*(gamma+lambda_)**2: 3e}")
        print(f"Risk Neutral Unconditional expectation: {(omega+alpha)/(1-(beta + alpha*(gamma+lambda_)**2)): 3e}")
        
        # Q-Q plot
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        stats.probplot(pred_data-data, dist="norm", plot=ax[0])
        stats.probplot(pred_h-realized_variance, dist="norm", plot=ax[1])
        ax[1].set_title(r'Q-Q Plot of $h_t$ Residuals')
        ax[0].set_title(r'Q-Q Plot of $R_t$ Residuals')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{model_name}_qqplot.png")
        residuals = pred_data - data
        from scipy.stats import kstest, jarque_bera, anderson, shapiro
        stat, p_value = kstest(residuals, 'norm')
        print(f"Kolmogorov-Smirnov Test Statistic: {stat}, p-value: {p_value}")
        stat, p_value = jarque_bera(residuals)
        print(f"Jarque-Bera Test Statistic: {stat}, p-value: {p_value}")
        result = anderson(residuals, dist='norm')
        print(f"Anderson-Darling Test Statistic: {result.statistic}")
        print("Critical Values:", result.critical_values)
        print("Significance Levels:", result.significance_level)
        stat, p_value = shapiro(residuals)
        print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p_value}")

        ## test epsilon
        
        ep = np.zeros(data.shape[0])
        for t in range(data.shape[0]):
            ep[t] = (pred_data[t] - (r + (lambda_-0.5)*pred_h[t]))/np.abs(pred_h[t])**0.5
        print("Testing epsilon......")
        stat, p_value = kstest(ep, 'norm')
        print(f"Kolmogorov-Smirnov Test Statistic: {stat}, p-value: {p_value}")
        stat, p_value = jarque_bera(ep)
        print(f"Jarque-Bera Test Statistic: {stat}, p-value: {p_value}")
        result = anderson(ep, dist='norm')
        print(f"Anderson-Darling Test Statistic: {result.statistic}")
        print("Critical Values:", result.critical_values)
        print("Significance Levels:", result.significance_level)
        stat, p_value = shapiro(ep)
        print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p_value}")

    def get_params(self):
        return self.params
    def set_params(self,params):
        self.params=  params
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
        lam = self.params['lambda']
        gamma = self.params['gamma']
        T = len(data)
        
        # Initialize variables
        h = np.zeros(T)  # Conditional variances (volatilities squared)
        r = 0.01
        # Initial value for h (often set to variance of first data point)
        h[0] = max(np.var(data), 1e-6)
        log_likelihood = 0.0
        
        for t in range(1, T):
            # Calculate residuals
            epsilon = (data[t-1]-(lam-0.5)*h[t-1] - r)/h[t-1]  # Error term (returns here)
            h[t] = omega + beta *(h[t-1]) + alpha * (epsilon-gamma*np.sqrt(np.abs(h[t-1])))**2
            log_likelihood += -0.5 * np.log(np.abs(h[t])) - 0.5 * ((data[t]-r-(lam-0.5)*h[t])**2 / h[t]) - 0.5*np.log(2*np.pi)
        
        return -log_likelihood  # Return negative likelihood for minimization
    def check_feasibilty(self, params):
        for i, con in enumerate(self.constraints):
            if con['fun'](params) < 0:
                raise ValueError(f"Initial parameters violate constraint {i + 1}")
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
            self.params['omega'], self.params['alpha'], self.params['beta'],\
            self.params['lambda'], self.params['gamma']= params
            # print(params)
            return self.evaluate_likelihood(data)
        
        # Initial parameter values
        initial_params = list(self.params.values())
        
        # Optimize using 'L-BFGS-B' method with parameter bounds (omega > 0, alpha, beta >= 0)
        bounds = [(1e-6, 1000), (0, 1), (0, 1), (0,None), (0,None)]  # Bounds for omega, alpha, beta
        options = {
            'disp': True
        }
        result = minimize(objective, initial_params, bounds=bounds, method='SLSQP',constraints=self.constraints,
                          options=options)
        
        optimized_params = result.x
        self.params['omega'], self.params['alpha'], self.params['beta'],\
        self.params['lambda'], self.params['gamma'] = optimized_params
        # if result.success:
            # print("Optimization successful. Parameters:", self.params)
        # else:
            # raise RuntimeError("Optimization failed:", result.message)
        return result
    
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
            lam = self.params['lambda']
            gamma = self.params['gamma']
        else:
            omega = params['omega']
            alpha = params['alpha']
            beta = params['beta']
            lam = params['lambda']
            gamma = params['gamma']

        # Initialize arrays
        h = np.zeros(n)  # Conditional variances (volatilities squared)
        data = np.zeros(n)  # Simulated data
        r = 0.01
        # Initial values
        h[0] = np.var(data)  # Can initialize with sample variance
        data[0] = 0  # Assuming no initial shock
        epsilon = np.random.normal(0, 1)
        for t in range(1, n):
            # Update conditional variance (h_t)
            h[t] = omega + beta *(h[t-1]) + alpha * (epsilon-gamma*np.sqrt(np.abs(h[t-1])))**2
            epsilon = np.random.normal(0, 1)
            # Simulate data (return) using the conditional variance
            data[t] = r  +(lam-0.5)*h[t] + epsilon * np.sqrt(np.abs(h[t]))  # Simulated return (or log return)
        
        return data, h
    
    def risk_neutral_parameters(self):
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        lambda_ = self.params['lambda']
        omega = self.params['omega']
        gamma_rn = lambda_ + gamma
        beta_rn = beta + alpha * (gamma_rn**2 - gamma**2) + \
                    alpha * (gamma_rn**2 - gamma**2)
        omega_rn = omega + alpha
        self.params.update({
            "gamma_rn": gamma_rn,
            "beta_rn": beta_rn,
            "omega_rn": omega_rn,
        })
    def get_vix(self, n=22, trading_days=252, t=0):
        if self.h is None:
            _, self.h = self.simulate()
            self.h = np.abs(self.h)
        self.risk_neutral_parameters()
        beta_rn = self.params['beta_rn']
        omega_rn = self.params['omega_rn']
        w_qn = (1-beta_rn**n)/(1-beta_rn)/n
        long_var = omega_rn/(1-beta_rn) * (1-w_qn)
        short_var = self.h[t+1] * w_qn
        self.w_qn = w_qn
        return np.sqrt((short_var + long_var) * trading_days) * 100
    
    def mgf_terms(self, phi , T):
        beta_rn = self.params['beta_rn']
        omega_rn = self.params['omega_rn']
        gamma_rn = self.params['gamma_rn']
        alpha = self.params['alpha']
        beta = self.params['beta_rn']
        omega = self.params['omega']
        C_values = [0]  # C(phi, 0)
        H_values = [phi]  # H(phi, 0)

        for n in range(T):
            H_n = H_values[-1]
            
            # Compute H(phi, n+1)
            H_next = (alpha * gamma_rn * H_n) / (1 - 2 * alpha * H_n) + beta * H_n

            # Compute C(phi, n+1)
            C_next = C_values[-1] - 0.5 * np.log(1 - 2 * alpha * H_n) + omega * H_n

            # Append to lists
            H_values.append(H_next)
            C_values.append(C_next)
        
        return C_values, H_values
    
    def mgf(self, phi, m, t):
        c_term, h_term = self.mgf_terms(phi, m)
        if self.h is None:
            _, self.h = self.simulate()
            self.h = np.abs(self.h)
        return np.exp(c_term[m] + h_term[m]*(self.h[t+1]))

    def vix_futures(self,n=22, T=22, t=0):
        if self.h is None:
            _, self.h = self.simulate()
        vix = self.get_vix()
        omega_rn = self.params['sigma2_rn']
        beta_rn = self.params['beta_rn']
        omega_rn = self.params['sigma2_rn']
        Gamma = (1-beta_rn**n)/(n*(1-beta_rn))
        a = 252 * (1-Gamma) * omega_rn/(1-beta_rn)
        b = 252 * Gamma
        # print(a, bq, bs)
        def integrand(x, a, b, T, t, f):
            return (1-np.exp(- a * x)* f(-x*b,T-t,((vix/100)**2-a)/b))/x**(3/2) 
        from scipy.integrate import quad
        integral_val, _ = quad(integrand, 0, np.inf, args=(a, b, T, t, self.mgf))

        return integral_val * 100 /2/np.sqrt(np.pi)