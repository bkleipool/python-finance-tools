import numpy as np
import matplotlib.pyplot as plt

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


class option():
	def __init__(self, underlying, strike_price, exp_date):
		return



if __name__ == '__main__':

	S1 = MG(state=0, mu=0, sigma=0.3)
	S2 = MG(state=0, mu=0, sigma=0.4, cov={S1.id:-0.035})
	print(MG.cov_mat)


	S = timeSeries((S1,S2, S1+2*S2))
	print(MG.cov_mat)


	P, dP = S.propegate(N=100)

	"""
	for i,p in enumerate(P):
		plt.plot(p, label=str(i+1))
	"""
	plt.plot(P[2], label='sim')
	plt.plot(P[0]+2*P[1], label='real')


	plt.legend(), plt.show()






