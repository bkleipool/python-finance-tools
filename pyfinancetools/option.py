import numpy as np
from .portfolio import MG

#option class
class option():

	#TODO: not implemented
	@staticmethod
	def MonteCarlo(self):
		#https://web.archive.org/web/20100318060412/http://www.puc-rio.br/marco.ind/monte-carlo.html#mc-eur
		#https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing
		#https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance
		return NotImplementedError

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
		#Compress timesteps of time-price matrix to match price data
		T = np.linspace(0, len(P)/self.r_period, len(P)) #time steps in original time series
		t = [self.t[np.argmin(np.abs(self.t-np.full(len(self.t), x)))] for x in T]
		V = np.array([self.V[np.argmin(np.abs(self.t-np.full(len(self.t), x)))] for x in T])

		#Sample from time-price matrix (select closest points to price data)
		S_indices = [np.argmin(np.abs(self.S-np.full(len(self.S), x))) for x in P]
		V_prices = [V[i,j] for i,j in zip(range(len(t)), S_indices)]

		return V_prices





#Greeks
def Delta(und, der, undMG=None):
	"""
	Finds the first order partial derivative of the option price V w.r.t. the stock price S.
	und = Underlying asset (array of stock prices OR MG object)
	der = Derived asset (Option object OR MG object)
	undMG = MG corresponding to the price history of `und' (optional, assumed underlying asset of option if empty)
	"""

	#Find delta between MGs from cov matrix
	if isinstance(und, MG) and isinstance(der, MG):
		return MG.cov_mat[und.id, der.id] / (und.sigma * der.sigma)

	#Find delta between option and underlying (price-time matrix + price history)
	elif isinstance(und, np.ndarray) and isinstance(der, option) and undMG == None:

		#Same procedure as option.toSeries()
		T = np.linspace(0, len(und)/der.r_period, len(und)) #time steps in original time series
		t = [der.t[np.argmin(np.abs(der.t-np.full(len(der.t), x)))] for x in T]
		V = np.array([der.V[np.argmin(np.abs(der.t-np.full(len(der.t), x)))] for x in T])

		dS = max(und)/len(und)
		dV = np.diff(V)
		dV = np.column_stack((dV[:,0], dV))

		S_indices = [np.argmin(np.abs(der.S-np.full(len(der.S), x))) for x in und]
		Delta = [dV[i,j]/dS for i,j in zip(range(len(t)), S_indices)]

		return Delta

	elif not isinstance(und, np.ndarray) and not isinstance(und, MG):
		TypeError("Underlying must be of type 'MG' or a numpy array")
	elif not isinstance(der, option) and not isinstance(der, MG):
		TypeError("Derived must be of type 'option' or 'MG'")
	#elif not isinstance(undMG, MG):
		#TypeError("undMG parameter must be of type 'MG' or None")

	elif isinstance(der, option) and not isinstance(und, np.ndarray):
		TypeError("For a derived of type 'option', underlying must be a numpy array")
	elif isinstance(der, MG) and not isinstance(und, MG):
		TypeError("For a derived of type 'MG', underlying must be of type 'MG'")
	#elif isinstance(der, MG) and not isinstance(undMG, type(None)):
		#TypeError("For a derived of type 'MG', undMG paramter may not be used")