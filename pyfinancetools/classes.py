import numpy as np
from math import erf


#Gaussian martingale class
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


	#TODO: implement MG*MG multiplication
	def __mul__(self, MG2):
		#https://stats.stackexchange.com/questions/15978/variance-of-product-of-dependent-variables
		#https://math.stackexchange.com/questions/2926888/find-operatornamecovx2-y2
		#https://stats.stackexchange.com/questions/352110/prove-that-textcorrx2-y2-rho2-where-x-y-are-jointly-n0-1-variab
		#https://mathhelpforum.com/t/how-to-get-the-covariance-of-x-and-x-cov-x-x.182125/


		if isinstance(MG2, MG):
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

	#TODO: implement covariance
	def __abs__(self):
		#https://en.wikipedia.org/wiki/Folded_normal_distribution
		#https://stats.stackexchange.com/questions/89899/variance-of-absolute-value-of-a-rv
		#https://mathhelpforum.com/t/how-to-get-the-covariance-of-x-and-x-cov-x-x.182125/

		mu_y = self.sigma*np.sqrt(2/np.pi) * np.exp(-self.mu**2/(2*self.sigma**2)) + self.mu*erf(self.mu/np.sqrt(2*self.sigma**2))
		sigma_y = np.sqrt(self.mu**2 + self.sigma**2 - mu_y**2)

		return MG(state = abs(self.state),
				  mu = mu_y,
				  sigma = sigma_y)
				  #cov = MG.covar(self.id))


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


	def propegate(self, N=1):
		self.stateHist = np.zeros((len(self.init_state),N+1))
		self.deltaHist = np.zeros((len(self.init_state),1))

		self.deltaHist = np.append(self.deltaHist, np.random.multivariate_normal(self.mu, self.cov, N).T, axis=1)
		self.stateHist = np.cumsum(self.deltaHist, axis=1) + np.tile(self.init_state, (N+1,1)).T

		return self.stateHist, self.deltaHist


#option class
class option():
	def __init__(self, underlying, strike_price, risk_free_rate, t_max):
		"""
		Instantiate option object
		und = underlying asset
		K = strike price
		r = risk-free interest rate
		r_period = nr. of propegation steps per interest period
		t_max = maturity date (in interest periods)
		"""
		self.K = strike_price
		self.r = risk_free_rate
		self.r_period = r_period
		self.t_max = t_max

		if isinstance(underlying, MG):
			self.und = underlying
		else:
			raise Exception('Underlying must be of type MG')


	@classmethod
	def BlackScholes_Euro(r, K, sigma, t_max, S_max, N, M, bounds):
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


	def toSeries(self, P):
		"""
		Convert the option price-time matrix to a time series given the price history data of the underlying MG.
		P = price history data of the underlying MG
		"""
		return





if __name__ == '__main__':
	import matplotlib.pyplot as plt


	S1 = MG(state=0, mu=0.001, sigma=0.3)
	S2 = MG(state=2, mu=0.005, sigma=0.4, cov={S1.id:0.035})
	#print(MG.cov_mat)


	S = timeSeries((S1, S2, S2-S1))
	#print(MG.cov_mat)


	P, dP = S.propegate(N=150)

	plt.plot(P[0], label='S1', linestyle='--')
	plt.plot(P[1], label='S2', linestyle='--')
	plt.plot(P[2], label='S2-S1', color='black')

	plt.legend(), plt.show()




