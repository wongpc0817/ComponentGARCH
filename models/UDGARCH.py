import numpy as np
from scipy.optimize import minimize

class UDGARCH:
    def __init__(self):
        """
        Initialize the GARCH(1,1) model with default parameters.
        """
        self.params = None  # Placeholder for model parameters
        self.rv_u = None 
        self.rv_d = None 
    def initialize_params(self):
        """
        Initialize model parameters.
        """
        # GARCH(1,1) parameters: omega, alpha, beta
        self.params = {
            'omegau': 0.001,  # Constant term
            'omegad': 0.001,  # Constant term
            'alphau': 1e-3,  # ARCH parameter (coefficient for epsilon^2)
            'alphad': 1e-3,  # ARCH parameter (coefficient for epsilon^2)
            'betau': 0.9,  # GARCH parameter (coefficient for previous variance)
            'betad': 0.9,  # GARCH parameter (coefficient for previous variance)
            'gammau': 10,
            'gammad': 10,
            'lambdau': 10,
            'lambdad': 10,
            'rhou': 0.4,
            'rhod': 0.4,
            'sigma2u': 0.01,
            'sigma2d': 0.01,
        }
        
        print("Parameters initialized:", self.params)
    
    def get_params(self):
        return self.params
    
    def set_realized_variance(self, rv_u, rv_d): 
        self.rv_u = rv_u
        self.rv_d = rv_d

    def evaluate_likelihood(self, data, epsilon1, epsilon2):
        """
        Define the log likelihood function for GARCH(1,1).
        
        Args:
            data (array): time series data.
        
        Returns:
            float: negative log likelihood value.
        """
        import scipy.stats as stats

        r = 0.01
        omega = [self.params[f'omega{i}'] for i in ['u','d']]
        alpha = [self.params[f'alpha{i}'] for i in ['u','d']]
        beta = [self.params[f'beta{i}'] for i in ['u','d']]
        gamma = [self.params[f'gamma{i}'] for i in ['u','d']]
        sigma2 = [self.params[f'sigma2{i}'] for i in ['u','d']]
        lam = [self.params[f'lambda{i}'] for i in ['u','d']]
        rho = [self.params[f'rho{i}'] for i in ['u','d']]
        T = len(data)
        if self.rv_u is None:
            rv_u = np.random.normal(0,1,T)
            rv_d = np.random.normal(0,1,T)
        else:
            rv_u = self.rv_u
            rv_d = self.rv_d
        hu = np.zeros(T)
        hd = np.zeros(T)
        RVu = np.zeros(T)
        RVd = np.zeros(T)
        log_likelihood = 0.0
        for t in range(1, T):
            zu1 = epsilon1[0][t-1]
            zu2 = epsilon1[0][t-1]*rho[0]+ epsilon2[0][t-1]*np.sqrt(1-rho[0]**2)
            zd1 = epsilon1[1][t-1]
            zd2 = epsilon1[1][t-1]*rho[1]+ epsilon2[1][t-1]*np.sqrt(1-rho[1]**2)
            RVu[t] = hu[t-1] + sigma2[0]*((zu2-gamma[0]*np.sqrt(hu[t-1]))-(1+gamma[0]**2*hu[t-1]))
            RVd[t] = hd[t-1] + sigma2[1]*((zd2-gamma[1]*np.sqrt(hd[t-1]))-(1+gamma[1]**2*hd[t-1]))
            hu[t] = omega[0]+beta[0]*hu[t-1] + alpha[0]*RVu[t]
            hd[t] = omega[1]+beta[1]*hd[t-1] + alpha[1]*RVd[t]
            ret = r + (lam[0]-0.5)*hu[t] + (lam[1]-0.5)*hd[t]+np.sqrt(hu[t])*zu1+np.sqrt(hd[t])*zd1
            log_likelihood += -0.5 * np.log(hu[t]+hd[t]) - 0.5 * ((ret - data[t])**2 / (hu[t]+hd[t])) \
                - 0.5*np.log(2*np.pi)
            k = (rv_u[t]-omega[0]-hu[0])/sigma2[0] + (1+gamma[0]**2*hu[t])
            log_likelihood += np.log(stats.ncx2.pdf(k, df = 1, nc= -gamma[0]*np.sqrt(hu[t])))
            k = (rv_d[t]-omega[1]-hd[0])/sigma2[1] + (1+gamma[1]**2*hd[t])
            log_likelihood += np.log(stats.ncx2.pdf(k, df = 1, nc= -gamma[1]*np.sqrt(hd[t])))
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
        def objective(params,data,epsilon1,epsilon2):
            self.params['omegau'], self.params['omegad'], self.params['alphau'],\
                 self.params['alphad'], self.params['betau'],self.params['betad'],\
                self.params['gammau'],self.params['gammad'],\
                      self.params['lambdau'], self.params['lambdad'],\
            self.params['rhou'], self.params['rhod'],\
            self.params['sigma2u'], self.params['sigma2d'] = params
            likelihood = self.evaluate_likelihood(data,epsilon1,epsilon2)
            print(likelihood)
            return likelihood
        
        # Initial parameter values
        initial_params = list(self.params.values())
        
        # Optimize using 'L-BFGS-B' method with parameter bounds (omega > 0, alpha, beta >= 0)
        bounds = [(1e-6, 1e6), (1e-6, 1e6), (0, 1), (0, 1),(0, 1),(0, 1),\
                  (0,1e6), (0,1e6), (0,1e6), (0,1e6),(-1,1),(-1,1),(0,1e6),(0,1e6)]  # Bounds for omega, alpha, beta
        epsilon1 = np.random.normal(0,1, [2, data.shape[0]])
        epsilon2 = np.random.normal(0,1, [2, data.shape[0]])
        result = minimize(objective, initial_params, args=(data,epsilon1,epsilon2),
                          bounds=bounds, method='L-BFGS-B')
        
        optimized_params = result.x
        self.params['omegau'], self.params['omegad'], self.params['alphau'],\
                 self.params['alphad'], self.params['betau'],self.params['betad'],\
                self.params['gammau'],self.params['gammad'],\
                      self.params['lambdau'], self.params['lambdad'],\
            self.params['rhou'], self.params['rhod'],\
            self.params['sigma2u'], self.params['sigma2d']= optimized_params
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
            omega = [self.params[f'omega{i}'] for i in ['u','d']]
            alpha = [self.params[f'alpha{i}'] for i in ['u','d']]
            beta = [self.params[f'beta{i}'] for i in ['u','d']]
            gamma = [self.params[f'gamma{i}'] for i in ['u','d']]
            sigma2 = [self.params[f'sigma2{i}'] for i in ['u','d']]
            lam = [self.params[f'lambda{i}'] for i in ['u','d']]
            rho = [self.params[f'rho{i}'] for i in ['u','d']]
        else:
            omega = [params[f'omega{i}'] for i in ['u','d']]
            alpha = [params[f'alpha{i}'] for i in ['u','d']]
            beta = [params[f'beta{i}'] for i in ['u','d']]
            gamma = [params[f'gamma{i}'] for i in ['u','d']]
            sigma2 = [params[f'sigma2{i}'] for i in ['u','d']]
            lam = [params[f'lambda{i}'] for i in ['u','d']]
            rho = [params[f'rho{i}'] for i in ['u','d']]

        # Initialize arrays
        h = np.zeros(n)  # Conditional variances (volatilities squared)
        data = np.zeros(n)  # Simulated data
        r = 0.01
        # Initial values
        h[0] = np.var(data)  # Can initialize with sample variance
        data[0] = 0  # Assuming no initial shock
        epsilon1 = np.random.normal(0, 1, [2,n])
        epsilon2 = np.random.normal(0, 1, [2,n])
        hu = np.zeros(n)
        hd = np.zeros(n)
        RVu = np.zeros(n)
        RVd = np.zeros(n)
        for t in range(1, n):
            zu1 = epsilon1[0][t-1]
            zu2 = epsilon1[0][t-1]*rho[0]+ epsilon2[0][t-1]*np.sqrt(1-rho[0]**2)
            zd1 = epsilon1[1][t-1]
            zd2 = epsilon1[1][t-1]*rho[1]+ epsilon2[1][t-1]*np.sqrt(1-rho[1]**2)
            RVu[t] = hu[t-1] + sigma2[0]*((zu2-gamma[0]*np.sqrt(hu[t-1]))-(1+gamma[0]**2*hu[t-1]))
            RVd[t] = hd[t-1] + sigma2[1]*((zd2-gamma[1]*np.sqrt(hd[t-1]))-(1+gamma[1]**2*hd[t-1]))
            hu[t] = omega[0]+beta[0]*hu[t-1] + alpha[0]*RVu[t]
            hd[t] = omega[1]+beta[1]*hd[t-1] + alpha[1]*RVd[t]
            data[t] = r + (lam[0]-0.5)*hu[t] + (lam[1]-0.5)*hd[t]+np.sqrt(hu[t])*zu1+np.sqrt(hd[t])*zd1
        
        return data, hu, hd
