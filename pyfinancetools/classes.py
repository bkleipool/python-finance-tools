import numpy as np
from math import erf


#Martingale class
class MG():
	inst = 0
	cov_mat = []
	"""
	Gaussian martingale class for simulating covariant securites.
	This class stores all the variables needed for a monte-carlo simulation.
	inst = number of MG instances
	cov_mat = covariance matrix of all MG instances
	"""

	@staticmethod
	def covar(Id, dictionary=False):
		if not dictionary:
			C = np.copy(MG.cov_mat[Id])
			#C[Id] = 0
			return C
		else:
			C = MG.cov_mat[Id]
			C[Id] = np.copy(MG.cov_mat[Id])
			return dict(enumerate(C))

	def __init__(self, state, mu, sigma, cov=None):
		"""
		state = initial state
		mu = expected value per period
		sigma = volatility per period (stdev)
		id = unique identifier
		cov = dict or array of covariances with other MG instances
		"""
		self.state = state
		self.mu = mu
		self.sigma = sigma
		self.id = MG.inst

		##TODO: check for cov entries out of range

		#convert covariance dict to array is applicable
		if not isinstance(cov, (list, np.ndarray, dict)):
			cov = np.zeros(MG.inst)

		elif isinstance(cov, dict):
			cov_dict = cov
			cov = np.zeros(MG.inst)
			for i,c in list(cov_dict.items()):
				cov[i] = c

		elif len(cov) != self.id:
			raise Exception("Nr. of covariance inputs does not match nr. of MG instances")

		#extend covariance matrix and add entries
		if len(MG.cov_mat) > 1:
			MG.cov_mat = np.hstack((MG.cov_mat, np.zeros((MG.inst,1))))
			MG.cov_mat = np.vstack((MG.cov_mat, np.zeros(MG.inst+1)))
			MG.cov_mat[self.id,self.id] = self.sigma**2
			for i,c in enumerate(cov):
				if i != self.id:
					MG.cov_mat[self.id,i] = MG.cov_mat[i,self.id] = c

		elif len(MG.cov_mat) == 1:
			MG.cov_mat = np.append(MG.cov_mat, cov[0])
			MG.cov_mat = np.vstack((MG.cov_mat, [cov[0], self.sigma**2]))

		elif len(MG.cov_mat) == 0:
			MG.cov_mat = [self.sigma**2]

		MG.inst += 1


	def __str__(self):
		cov = dict(enumerate(MG.cov_mat[self.id]))
		cov.pop(self.id)

		return 'id: '+str(self.id)+', state: '+str(self.state)+', mu: '+str(self.mu)+', sigma: '+str(self.sigma)+', cov: '+str(cov)


	def __add__(self, MG2):
		if isinstance(MG2, MG):
			cov1 = MG.covar(self.id)
			cov2 = MG.covar(MG2.id)

			return MG(state=self.state + MG2.state,
					  mu=self.mu + MG2.mu,
					  sigma=np.sqrt(self.sigma**2 + MG2.sigma**2 + 2*MG.cov_mat[self.id, MG2.id]),
					  cov=cov1+cov2)

		elif isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(state=self.state + x,
					  mu=self.mu + x,
					  sigma=self.sigma,
					  cov=MG.covar(self.id))

		else:
			raise TypeError


	def __radd__(self, MG2):
		if isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2
			
			return MG(state=self.state + x,
					  mu=self.mu + x,
					  sigma=self.sigma,
					  cov=MG.covar(self.id))

		else:
			raise TypeError


	def __sub__(self, MG2):
		if isinstance(MG2, MG):
			cov1 = MG.covar(self.id)
			cov2 = MG.covar(MG2.id)

			return MG(state=self.state - MG2.state,
					  mu=self.mu - MG2.mu,
					  sigma=np.sqrt(self.sigma**2 + MG2.sigma**2 - 2*MG.cov_mat[self.id, MG2.id]),
					  cov=cov1-cov2)

		elif isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(state=self.state - x,
					  mu=self.mu - x,
					  sigma=self.sigma,
					  cov=MG.covar(self.id))

		else:
			raise TypeError


	def __rsub__(self, MG2):
		if isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(state=x - self.state,
					  mu=x - self.mu,
					  sigma=self.sigma,
					  cov=-MG.covar(self.id))

		else:
			raise TypeError


	##TODO: MG*MG multiplication not implemented
	def __mul__(self, MG2):
		#https://stats.stackexchange.com/questions/15978/variance-of-product-of-dependent-variables
		#https://math.stackexchange.com/questions/2926888/find-operatornamecovx2-y2
		#https://stats.stackexchange.com/questions/352110/prove-that-textcorrx2-y2-rho2-where-x-y-are-jointly-n0-1-variab

		#https://stats.stackexchange.com/questions/15978/variance-of-product-of-dependent-variables/182822#182822
		#https://math.stackexchange.com/questions/1915366/expected-value-of-product-of-square-of-random-variable
		#https://en.wikipedia.org/wiki/Distribution_of_the_product_of_two_random_variables
		#https://math.stackexchange.com/questions/2044833/need-help-to-get-covxy-z

		if isinstance(MG2, MG):
			"""
			var = (MG2.sigma**2 - MG.cov_mat[self.id, MG2.id]**2/self.sigma**2)*(self.mu**2+self.sigma**2) \
				+ (self.mu*MG2.mu+MG.cov_mat[self.id, MG2.id])**2 \
				+ MG.cov_mat[self.id, MG2.id]**2/(self.sigma**2 * MG2.sigma**2) * (self.mu**4+6*self.mu**2*self.sigma**3+3*self.sigma**4)
			cov1 = MG.covar(self.id)
			cov2 = MG.covar(MG2.id)

			return MG(state=self.state*MG2.state,
					  mu=self.mu*MG2.mu + MG.cov_mat[self.id, MG2.id],
					  sigma=np.sqrt((self.sigma**2+self.mu**2)*(MG2.sigma**2+MG2.mu**2)-(self.mu*MG2.mu)**2),
					  cov=self.mu*cov1 + MG2.mu*cov2)
			"""
			raise NotImplementedError

		elif isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(state=x*self.state,
					  mu=x*self.mu,
					  sigma=np.abs(x)*self.sigma,
					  cov=x*MG.covar(self.id))

		else:
			raise TypeError


	def __rmul__(self, MG2):
		if isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(state=x*self.state,
					  mu=x*self.mu,
					  sigma=np.abs(x)*self.sigma,
					  cov=x*MG.covar(self.id))

		else:
			raise TypeError


	def __neg__(self):
		return MG(state=-self.state,
				  mu=-self.mu,
				  sigma=self.sigma,
				  cov=-MG.covar(self.id))

	##TODO: not implemented
	def __abs__(self):
		#https://en.wikipedia.org/wiki/Folded_normal_distribution
		#https://stats.stackexchange.com/questions/89899/variance-of-absolute-value-of-a-rv
		#https://mathhelpforum.com/t/how-to-get-the-covariance-of-x-and-x-cov-x-x.182125/
		"""
		mu_y = self.sigma*np.sqrt(2/np.pi) * np.exp(-self.mu**2/(2*self.sigma**2)) + self.mu*erf(self.mu/np.sqrt(2*self.sigma**2))
		sigma_y = np.sqrt(self.mu**2 + self.sigma**2 - mu_y**2)

		return MG(state = abs(self.state),
				  mu = mu_y,
				  sigma = sigma_y)
				  #cov = MG.covar(self.id))
		"""
		raise NotImplementedError




#Time series class
class timeSeries():
	def __init__(self, MGs, init_state=None):
		"""
		MGs = MG objects that it simulates
		deltaHist = history of differences of states
		propegator = propegation function
		"""

		self.MGs = MGs
		self.init_state = [i.state for i in self.MGs] if init_state == None else init_state
		self.ids = [i.id for i in self.MGs]
		self.mu = [i.mu for i in self.MGs]
		self.cov = MG.cov_mat[np.ix_(self.ids, self.ids)] #select rows & columns from MG.cov_mat


	def propagate(self, N=1):
		self.stateHist = np.zeros((len(self.init_state),N+1))
		self.deltaHist = np.zeros((len(self.init_state),1))

		self.deltaHist = np.append(self.deltaHist, np.random.multivariate_normal(self.mu, self.cov, N).T, axis=1)
		self.stateHist = np.cumsum(self.deltaHist, axis=1) + np.tile(self.init_state, (N+1,1)).T

		return self.stateHist, self.deltaHist




#option class
class option():
	#Black-scholes PDE
	@staticmethod
	def BlackScholes(r, K, sigma, t_max, S_max, N, M, bounds):
		"""
		Calculates the option price according to the Black-Scholes PDE for European options, with the given parameters and boundary conditions.
		r = risk-free interest rate
		K = Strike price(s)
		C = net credit per share (only for Backspread options)
		sigma = Underlying volatity (stdev)
		t_max = Time till maturity (in interest periods)
		S_max = Maximum considered stock price (advised +4*sigma)
		N = Number of time (t) samples
		M = Number of stock price (S) samples
		bounds = function boundary conditions V(0,t), V(S_max,t), V(S, t_max)
		
		returns: t = (N-array) array of time steps
		returns: S = (M-array) array of stock prices
		returns: V = (N*M matrix) array of option prices
		"""

		#setup domain & boundary conditions
		t = np.linspace(0, t_max, N) 	#time range
		S = np.linspace(0, S_max, M) 	#Stock price range
		dt = max(t)/len(t) 				#t interval size
		dS = max(S)/len(S) 				#S interval size

		V = np.zeros((len(t),len(S)))
		V[:,0], V[:,-1], V[-1] = bounds(t=t, S=S, K=K, r=r, t_max=t_max, S_max=S_max)

		#Solve PDE
		for i in reversed(range(1, len(t))):
			for j in range(0, len(S)-1):
				V[i-1, j] = (dt*r*S[j]/dS)*(V[i,j+1]-V[i,j]) + (dt*sigma**2*S[j]**2)/(2*dS**2)*(V[i,j+1]+V[i,j-1]-2*V[i,j]) - dt*r*V[i,j] + V[i,j]

		return t, S, V

	#Boundary conditions
	@staticmethod
	def boundsCall(t, S, K, r, t_max, S_max, **kwargs):
		"""
		Lower, upper and maturity boundaries for a call option.
		t = array of timesteps
		S = array of stock prices
		K = Strike price(s)
		r = risk-free interest rate
		t_max = Time till maturity (in interest periods)
		S_max = Maximum considered stock price
		"""
		lbound = 0
		ubound = S_max - K*np.exp(-r*(t_max-t))
		matbound = np.maximum(S-K, 0)
		return lbound, ubound, matbound

	@staticmethod
	def boundsPut(t, S, K, r, t_max, S_max, **kwargs):
		"""
		Lower, upper and maturity boundaries for a put option.
		t = array of timesteps
		S = array of stock prices
		K = Strike price(s)
		r = risk-free interest rate
		t_max = Time till maturity (in interest periods)
		S_max = Maximum considered stock price
		"""
		lbound = S_max - K*np.exp(-r*(t_max-t))
		ubound = 0
		matbound = np.maximum(K-S, 0)
		return lbound, ubound, matbound

	@staticmethod
	def boundsButterfly(S, K, **kwargs):
		"""
		Lower, upper and maturity boundaries for a butterfly option.
		S = array of stock prices
		K = Strike price(s)
		"""
		lbound = 0
		ubound = 0
		matbound = np.minimum(np.maximum(S-K[0], 0), np.maximum(K[1]-S, 0))
		return lbound, ubound, matbound

	@staticmethod
	def boundsStraddle(t, S, K, r, t_max, S_max, **kwargs):
		"""
		Lower, upper and maturity boundaries for a straddle option.
		S = array of stock prices
		K = Strike price(s)
		"""
		lbound = K
		ubound = S_max - K*np.exp(-r*(t_max-t))
		matbound = np.maximum(K-S, S-K)
		return lbound, ubound, matbound

	@staticmethod
	def boundsStrangle(t, S, K, r, t_max, S_max, **kwargs):
		"""
		Lower, upper and maturity boundaries for a strangle option.
		S = array of stock prices
		K = Strike price(s)
		"""
		lbound = S_max - K[0]*np.exp(-r*(t_max-t))
		ubound = S_max - K[1]*np.exp(-r*(t_max-t))
		matbound = np.maximum(K[0]-S, np.maximum(S-K[1], 0))
		return lbound, ubound, matbound

	@staticmethod
	def boundsCollar(S, K, **kwargs):
		"""
		Lower, upper and maturity boundaries for a collar option.
		S = array of stock prices
		K = Strike price(s)
		"""
		lbound = 0
		ubound = K[1]-K[0]
		matbound = np.minimum(np.maximum(S-K[0], 0), K[1]-K[0])
		return lbound, ubound, matbound

	@staticmethod
	def boundsBackspread(t, S, K, r, t_max, S_max, **kwargs):
		"""
		Lower, upper and maturity boundaries for a backspread option.
		S = array of stock prices
		K = Strike price(s)
		"""
		lbound = K[0]
		ubound = K[1]-K[0]
		matbound = np.maximum(K[0]-S, np.minimum(S-K[0],K[1]-K[0]))
		return lbound, ubound, matbound


	def __init__(self, underlying, bounds):
		"""
		Instantiate option object
		und = underlying asset
		bounds = boundary conditions of the option
		"""
		self.bounds = bounds

		if isinstance(underlying, MG):
			self.und = underlying
		else:
			raise Exception('Underlying must be of type MG')


	def pricing(self, r, K, r_period, t_max, S_max, N, M):
		"""
		Calculate prices for option object
		K = strike price
		r = risk-free interest rate
		r_period = nr. of propegation steps per interest period
		t_max = maturity date (in interest periods)
		"""

		##TODO: idk if self.und.sigma/np.sqrt(self.r_period) is entirely correct
		self.r_period = r_period
		self.t, self.S, self.V = option.BlackScholes(r=r, K=K, sigma=self.und.sigma/np.sqrt(self.r_period), t_max=t_max, S_max=S_max, N=N, M=M, bounds=self.bounds)


	def toSeries(self, P):
		"""
		Convert the option price-time matrix to a time series given the price history data of the underlying MG.
		P = price history data of the underlying MG
		"""
		#V = np.delete(self.V, [i for i in range(len(self.t)) if i%int(len(self.V)/len(P)) != 0], axis=0)
		#t = np.delete(self.t, [i for i in range(len(self.t)) if i%int(len(self.t)/len(P)) != 0])

		T = np.linspace(0, len(P)/self.r_period, len(P)) #time steps in original time series
		t = [self.t[np.argmin(np.abs(self.t-np.full(len(self.t), x)))] for x in T]

		print(P)
		print(t)

		S_indices = [np.argmin(np.abs(self.S-np.full(len(self.S), x))) for x in P]
		V_prices = [V[i,j] for i,j in zip(range(len(t)), S_indices)]


		#ax1.plot(X, Stock.price, label='stock', color='black')
		#ax2.plot(X, V_prices, label='option', color='red')

		return P[S_indices] #V_prices




#Portfolio selection
def Markowitz_minrisk(MGs, cap=None, weights=False):
	"""
	Returns the minimum volatility portfolio on the efficient frontier.
	MGs = Arrray of available stocks
	cap = Starting capital (optional)
	weights = Wether or not to return the portfolio or simply the weights
	returns: w = portfolio OR portfolio weights vector
	"""
	#calculate covariance matrix
	ids = np.array([i.id for i in MGs])
	states = np.array([i.state for i in MGs])
	cov = MG.cov_mat[np.ix_(ids, ids)]

	#normalise cov matrix
	cov_n = np.multiply(np.array([1/states]).T*(1/states), cov)

	one = np.ones(len(MGs))
	covi = np.linalg.inv(cov_n)
	C = one @ covi @ one

	w = 1/C*(covi @ one)
	if cap != None:
		w = cap * w/(w @ states)

	if weights:
		return w
	else:
		return w @ MGs


def Markowitz_ef(MGs, mu, cap=None, weights=False):
	"""
	Returns the portfolio on the efficient frontier with a given target return mu.
	MGs = Arrray of available stocks
	mu = target return percentage (per propegation step)
	cap = Starting capital (optional)

	weights = Wether or not to return the portfolio or simply the weights
	returns: w = portfolio OR portfolio weights vector
	"""

	#calculate values
	ids = np.array([i.id for i in MGs])
	states = np.array([i.state for i in MGs])
	cov = MG.cov_mat[np.ix_(ids, ids)]
	MU = np.array([i.mu for i in MGs])

	#normalise values
	MU_n = MU*(1/states)
	cov_n = np.multiply(np.array([1/states]).T*(1/states), cov)

	one = np.ones(len(MGs))
	covi = np.linalg.inv(cov)
	A = MU_n @ covi @ MU_n
	B = MU_n @ covi @ one
	C = one @ covi @ one

	w = (mu*C-B)/(A*C-B**2)*(covi @ MU_n) + (A-mu*B)/(A*C-B**2)*(covi @ one)
	if cap != None:
		w = cap * w/(w @ states)

	if weights:
		return w
	else:
		return w @ MGs





if __name__ == '__main__':
	import matplotlib.pyplot as plt

	#define MGs
	S1 = MG(state=36, mu=0.035*(36/365), sigma=0.24)
	S2 = MG(state=45, mu=0.09*(45/365), sigma=0.48, cov={S1.id:0.011})
	S3 = MG(state=51, mu=0.10*(51/365), sigma=0.68, cov={S2.id:-0.016})
	#S4 = MG(state=48, mu=0.07*(48/365), sigma=0.38, cov={S2.id:0.012, S3.id:0.012})


	print(Markowitz_ef((S1, S2, S3), mu=0.075/365, weights=True))
	Pf = Markowitz_ef((S1, S2, S3), cap=5000, mu=0.075/365) 

	#define timeseries
	TS = timeSeries((S1, S2, S3, Pf))
	P, dP = TS.propagate(N=365)

	#print((P[4,-1]-P[4,0])/P[4,0])
	
	#plot
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.plot(P[0], label='S1', linestyle='--', linewidth=1)
	ax1.plot(P[1], label='S2', linestyle='--', linewidth=1)
	ax1.plot(P[2], label='S3', linestyle='--', linewidth=1)
	#ax1.plot(P[3], label='S4', linestyle='--', linewidth=1)
	ax2.plot(P[3], label='Portfolio', color='black')
	plt.tight_layout(), ax1.legend(), ax2.legend(), plt.show()


	"""

	#define option
	Opt = option(S1, bounds=option.boundsPut)
	Opt.pricing(r=0.015, K=65, r_period=150, t_max=1, S_max=130, N=3000, M=100)

	V = Opt.toSeries(P[0])
	plt.plot(P[0], label='P')
	plt.plot(V, label='V')
	plt.show()
	"""

