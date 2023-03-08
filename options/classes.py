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
	def __init__(self, mu, sigma, cov=None):
		"""
		mu = expected value per period
		sigma = volatility per period (stdev)
		id = unique identifier
		cov = dict of covariances with other MG instances
		"""
		self.mu = mu
		self.sigma = sigma
		self.id = MG.inst

		#convert covariance dict to array
		if cov == None:
			cov = np.zeros(MG.inst)
		else:
			cov_dict = cov
			cov = np.zeros(MG.inst)
			for i,c in list(cov_dict.items()):
				cov[i] = c

		##TODO: check for cov entries out of range

		#extend covariance matrix and add entries
		if len(MG.cov_mat) > 1:
			MG.cov_mat = np.hstack((MG.cov_mat, np.zeros((MG.inst,1))))
			MG.cov_mat = np.vstack((MG.cov_mat, np.zeros(MG.inst+1)))
			MG.cov_mat[self.id,self.id] = self.sigma**2
			for i,c in enumerate(cov):
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

		return 'id: '+str(self.id)+'  mu: '+str(self.mu)+'  sigma: '+str(self.sigma)+'  cov: '+str(cov)

	@staticmethod
	def covar(Id, dictionary=False):
		if not dictionary:
			return np.delete(MG.cov_mat[Id], Id)
		else:
			return dict(enumerate(np.delete(MG.cov_mat[Id], Id)))


	def __add__(self, MG2):
		if isinstance(MG2, MG):
			return MG(mu=self.mu + MG2.mu,
					  sigma=np.sqrt(self.sigma**2 + MG2.sigma**2 + 2*MG.cov_mat[self.id, MG2.id]))

		elif isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2
			return MG(mu=self.mu + x,
					  sigma=self.sigma,
					  cov=MG.covar(self.id, dictionary=True))

		else:
			raise TypeError


	def __radd__(self, MG2):
		if isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(mu=self.mu + x,
					  sigma=self.sigma,
					  cov=MG.covar(self.id, dictionary=True))

		else:
			raise TypeError


	def __sub__(self, MG2):
		if isinstance(MG2, MG):
			return MG(mu=self.mu - MG2.mu,
					  sigma=np.sqrt(self.sigma**2 + MG2.sigma**2 - 2*MG.cov_mat[self.id, MG2.id]))

		elif isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2

			return MG(mu=self.mu - x,
					  sigma=self.sigma,
					  cov=MG.covar(self.id, dictionary=True))

		else:
			raise TypeError


	def __rsub__(self, MG2):
		if isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2
			cov = MG.covar(self.id)

			return MG(mu=x - self.mu,
					  sigma=self.sigma,
					  cov=dict(enumerate(-cov)))

		else:
			raise TypeError


	def __mul__(self, MG2):
		if isinstance(MG2, MG):
			raise NotImplementedError

		elif isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2
			cov = x*MG.covar(self.id)

			return MG(mu=x*self.mu,
					  sigma=np.abs(x)*self.sigma,
					  cov=dict(enumerate(cov)))

		else:
			raise TypeError


	def __rmul__(self, MG2):
		if isinstance(MG2, int) or isinstance(MG2, float):
			x = MG2
			cov = x*MG.covar(self.id)

			return MG(mu=x*self.mu,
					  sigma=np.abs(x)*self.sigma,
					  cov=dict(enumerate(cov)))

		else:
			raise TypeError



S1 = MG(mu=0, sigma=0.1)
S2 = MG(mu=0, sigma=0.2, cov={S1.id:0.015})
S3 = MG(mu=0, sigma=0.3)
S4 = MG(mu=1, sigma=0.4, cov={S2.id:-0.025})


S5 = S4*2
print(S5)

print(MG.cov_mat)






#shit class
class timeseries():
	def __init__(self, init_state, propegator, **kwargs):
		"""
		stateHist = history of states
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


S = timeseries(propegator=None, init_state=95.47, mu=0.00167593, sigma=1.1647)


