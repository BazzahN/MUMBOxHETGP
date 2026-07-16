import numpy as np
from scipy.integrate import simpson
from scipy.optimize import bisect
from scipy.stats import norm

def _fit_gumbel(fmean, fsd):
    """
    Helper function to fit gumbel distribution when initialising the MES and MUMBO acquisition functions.

    The Gumbel distribution for minimas has a cumulative density function of f(y)= 1 - exp(-1 * exp((y - a) / b)), i.e. the q^th quantile is given by
    Q(q) = a + b * log( -1 * log(1 - q)). We choose values for a and b that match the Gumbel's
    interquartile range with that of the observed empirical cumulative density function of Pr(y*<y)
    i.e.  Pr(y* < lower_quantile)=0.25 and Pr(y* < upper_quantile)=0.75.
    """
	#TODO Edit to sample max values instead of min

	#CORRECT, but I standardise, in botorch they don't 
    def probf(x: np.ndarray) -> float:
        # Build empirical CDF function
        return np.exp(np.sum(norm.logcdf((x - fmean) / fsd), axis=0))

    # initialise end-points for binary search (the choice of 5 standard deviations ensures that these are outside the IQ range)
    left = np.min(fmean - 3 * fsd) #minux 3 for some reason on botorch?
    right = np.max(fmean + 5 * fsd)

    def binary_search(val: float) -> float:
        return bisect(lambda x: probf(x) - val, left, right, maxiter=10000)

    # Binary search for 3 percentiles
    lower_quantile, medium, upper_quantile = map(binary_search, [0.25, 0.5, 0.75])

    # solve for Gumbel scaling parameters CORRECT
    b = (lower_quantile - upper_quantile) / (np.log(np.log(4.0 / 3.0)) - np.log(np.log(4.0)))
    a = medium + b * np.log(np.log(2.0))

    return a, b


class MUMBO():

	def __init__(self,
					model,
					bounds = [0,1],
					max_value_samples = 10,
					grid_size = 40):
		
		self.model = model
		self.bounds = bounds   
		self.max_value_samples = max_value_samples
		self.grid_size = grid_size
		self.mvs = None

	def _required_parameters_initialized(self):
		"""
		Checks if all required parameters are initialized.
		"""
		return self.mvs is not None

	def update_parameters(self):
		"""
		MUMBO requires acces to a sample of possible minimum values y* of the objective function.
		To build this sample we approximate the empirical c.d.f of Pr(y*<y) with a Gumbel(a,b) distribution.
		This Gumbel distribution can then be easily sampled to yield approximate samples of y*

		This needs to be called once at the start of each BO step.
		"""
		
		grid = np.linspace(self.bounds[0],self.bounds[1],self.grid_size)
		grid = np.expand_dims(grid,axis=1)

		#Concatinate evenly space grid and 
		grid = np.vstack([self.model.model['X0'],grid])

		preds = self.model.predict(grid)

		mean_f = preds['mean']
		sd_f = np.sqrt(preds['sd2'])

		a, b = _fit_gumbel(mean_f,sd_f)

		uniform_samples = np.random.rand(self.max_value_samples)
		gumbel_samples = -np.log(-np.log(uniform_samples)) * b + a

		self.mvs = gumbel_samples

	def evaluate(self,x):
		"""
		Evaluate the MUMBO acquisition function at a given point x.
		
		Parameters
		----------
		x : ndarray
			Input location (shape: (m,d+1))
		
		Returns
		-------
		float
			Value of the MUMBO acquisition function at x
		"""
		#Calculates max value samples if None
		if not self._required_parameters_initialized():
			self.update_parameters()
		
		#Extract replication index
		n = x[:, -1].astype(int).reshape(-1,1)
		x = x[:, :-1]
		
		print("The x shape is: ", x.shape)
		print("The n shape is: ", n.shape)

		# Get predictive mean and variance from the model
		preds = self.model.predict(x)   

		mean_f = preds["mean"].reshape(-1,1)
		var_f = preds['sd2'].reshape(-1,1) 
		mean_eps = preds['nugs'].reshape(-1,1)

		std_f = np.sqrt(var_f)
		std_f = np.maximum(std_f, 1e-10)  # Avoid division by zero

		#Calculate Correlation Term
		rho = np.sqrt(n) * std_f / np.sqrt(n * var_f + mean_eps)
		
		gammas = (self.mvs - mean_f) / std_f

		ESGmean = rho * (norm.pdf(gammas)) / norm.cdf(gammas)
		ESGvar = 1 - rho * ESGmean * (gammas + norm.pdf(gammas)/norm.cdf(gammas))
		ESGvar = np.maximum(ESGvar,0)


		upper_limit = ESGmean + 8 * np.sqrt(ESGvar)
		lower_limit = ESGmean - 8 * np.sqrt(ESGvar)

		theta = np.linspace(lower_limit,upper_limit,num = 5000)
		
		denominator = np.sqrt(1 - rho**2)

		density = norm.pdf(theta) * (norm.cdf((gammas - rho * theta)/denominator))
		entropy_function = -density * np.log(density,out=np.zeros_like(density), where=(density != 0))

		approx_entropy = simpson(entropy_function.T,x=theta.T)
		#print("The approx_entropy shape is: ", approx_entropy.shape)
		approx_entropy = np.mean(approx_entropy,axis = 0)

		f_acqu_x = 0.5 *np.log(2*np.pi*np.e) - approx_entropy

		#Hard Code for now
		b0 = 1
		b1 = 0.5

		#cost = 1/n
		cost = 1
		#print("The cost shape is: ", cost.shape)
		#print("The f_acqu_x shape is: ", f_acqu_x.shape)
		f_acqu_x= f_acqu_x.reshape(-1,1) * cost
		return f_acqu_x.reshape(-1,1)