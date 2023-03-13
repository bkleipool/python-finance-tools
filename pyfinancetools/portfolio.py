import numpy as np


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

