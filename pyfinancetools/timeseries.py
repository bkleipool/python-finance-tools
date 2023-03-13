import numpy as np
from .portfolio import MG

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

