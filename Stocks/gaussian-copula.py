import numpy as np
import scipy.special as sc
from matplotlib import pyplot as plt

class stockBundle():
	def __init__(self, init_prices, mu_vec, cov_matrix):
		self.mu = np.array(mu_vec)
		self.cov = cov_matrix
		self.initP = np.array(init_prices)
		self.deltaP = np.full((len(self.initP),1), 0)
		self.price = None

	def propegate(self, N=1):
		self.price = np.zeros((len(self.initP),N+1))
		self.deltaP = np.append(self.deltaP, np.random.multivariate_normal(self.mu, self.cov, N).T, axis=1)
		for i in range(len(self.initP)):
			self.price[i] = np.cumsum(self.deltaP[i]) + np.full(N+1, self.initP[i])

	def movingAverage(self, index, w):
		avg = np.convolve(self.price[index], np.ones(w), 'valid') / w
		return np.append(np.full(len(self.price[index])-len(avg), None), avg)

	def summary(self, index):
		print("Stock", index+1, "summary")
		print("change:", np.round(self.price[index,-1]-self.price[index,0], 0),
	  		  "min:",	 np.round(np.min(self.price[index]), 0),
	  		  "mean:",	 np.round(np.mean(self.price[index]), 0),
	  		  "max:",	 np.round(np.max(self.price[index]), 0))

		print("mean daily gain ($):", np.round(np.mean(self.deltaP[index]), 2))
		print("mean annual gain (%):", np.round(((self.price[index,-1]/self.price[index,0])**(365/len(self.price[index]))-1)*100, 2), '\n')


class stock():
	def __init__(self, init_price, mu, sigma):
		self.mu = mu
		self.sigma = sigma
		self.initP = init_price
		self.deltaP = np.array([0])
		self.price = np.array([])

	def propegate(self, N=1):
		self.deltaP = np.append(self.deltaP, np.random.normal(self.mu ,self.sigma ,N))
		self.price = np.cumsum(self.deltaP) + self.initP

	def movingAverage(self, w):
		avg = np.convolve(self.price, np.ones(w), 'valid') / w
		return np.append(np.full(len(self.price)-len(avg), None), avg)


	def summary(self):
		print("change:", np.round(self.price[-1]-self.price[0], 0),
	  		  "min:",	 np.round(np.min(self.price), 0),
	  		  "mean:",	 np.round(np.mean(self.price), 0),
	  		  "max:",	 np.round(np.max(self.price), 0))

		print("mean daily gain ($):", np.round(np.mean(self.deltaP), 2))
		print("mean annual gain (%):", np.round(((self.price[-1]/self.price[0])**(365/len(self.price))-1)*100, 2), '\n')


N = 3*365
X = np.arange(0,N)
Stock = stock(init_price=95.74, mu=0.00167593, sigma=1.89649152648)
Stock.propegate(N-1)
Stock.summary()

plt.plot(X, Stock.price)
plt.show()


"""
#example with a bundle of covariant stocks:
if __name__ == '__main__':
	N = 3*365
	X = np.arange(0,N)
	Stocks = stockBundle(init_prices = [95.74, 149.46, 261.53],
						 mu_vec = [0.00167593, 0.05834956, 0.08673436],
						 cov_matrix = np.array([[3.59668011,  3.85891464,  6.83758698],
												[3.85891464,  8.4526874,  10.58978451],
												[6.83758698, 10.58978451, 22.88250926]]))

	Stocks.propegate(N-1)

	Stocks.summary(index=0)
	Stocks.summary(index=1)
	Stocks.summary(index=2)

	M = 40
	w = np.random.rand(3*M).reshape((M,3))
	w /= np.tile(np.sum(w, axis=1).reshape((M,1)), (1,3))
	w[0] = [0.24485031, 0.2856884,  0.46946129]



	fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

	ax1.plot(X,Stocks.price[0], label="$GOOG$", color='black')
	ax1.plot(X,Stocks.price[1], label="$AAPL$", color='gray')
	ax1.plot(X,Stocks.price[2], label="$MSFT$", color='lightgray')
	ax1.ticklabel_format(useOffset=False, style='plain')	#no scientific notation
	ax1.legend()

	for i in reversed(range(len(w[:,0]))):
		Portfolio = np.sum([w[i,j]*Stocks.price[j] for j in range(len(w[0]))],axis=0)
		Portfolio *= 1000/np.sum(Portfolio[0])
		color = np.linalg.norm(w[i]-w[0])
		if i == 0:
			ax2.plot(X, Portfolio, color='red')
		else:
			ax2.plot(X, Portfolio, color=(color, color,color), linewidth=1)

	plt.show()
"""