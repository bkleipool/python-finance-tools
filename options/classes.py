import numpy as np

#Martingale class
class MG():
	inst = 0
	cov_mat = []
	"""
	Martingale class for simulating covariant securites.
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

		return 'id: '+str(self.id)+'  state: '+str(self.state)+'  mu: '+str(self.mu)+'  sigma: '+str(self.sigma)+'  cov: '+str(cov)


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


	def __mul__(self, MG2):
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



S1 = MG(state=0, mu=0, sigma=0.1)
S2 = MG(state=0, mu=0, sigma=0.2, cov={S1.id:0.015})
print(MG.cov_mat)

S3 = 2*S2 - S2 #S2+S2
print(S3)

print(MG.cov_mat)






class timeSeries():
	def __init__(self, MGs, init_state=None):
		"""
		MGs = MG objects that it simulates
		deltaHist = history of differences of states
		propegator = propegation function
		params = dict of parameters for the propegator function

		"""
		self.stateHist = np.array([init_state]) if type(init_state) != list else init_state 
		self.deltaHist = np.array([0])
		self.propegator = propegator
		self.params = kwargs

		#print(self.params)

	def propegate(self, N=1):
		self.deltaHist = np.append(self.deltaHist, self.propegator(self.params, N))
		self.stateHist = np.cumsum(self.deltaHist, axis=0) + self.initP



class option():
	def __init__(self, underlying, strike_price, exp_date):
		return


#S = timeSeries(propegator=None, init_state=95.47, mu=0.00167593, sigma=1.1647)


