import numpy as np

#Boundary conditions
#https://www.optiontradingtips.com/strategies/
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




#Greeks
def Delta(S, V):
	"""
	Finds the first order partial derivative of the option price V w.r.t. the stock price S.
	S = Array of stock prices
	V = Array of corresponding option prices
	"""
	dS = max(S)/len(S)
	return np.diff(V)/dS

def Gamma(S, V):
	"""
	Finds the second order partial derivative of the option price V w.r.t. the stock price S.
	S = Array of stock prices
	V = Array of corresponding option prices
	"""
	dS = max(S)/len(S)
	return np.diff(np.diff(V))/dS**2

def Theta(t, V):
	"""
	Finds the first order partial derivative of the option price V w.r.t. the time t.
	t = Array of timesteps
	V = Array of corresponding option prices
	"""
	dt = max(t)/len(t)
	return np.diff(V)/dt


#Black-scholes equations
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


def BlackScholes_USA(r, K, sigma, t_max, S_max, N, M, bounds):
	"""
	Calculates the option price according to the Black-Scholes PDE for European options, with the given parameters and boundary conditions.
	r = risk-free interest rate
	K = Strike price(s)
	sigma = Underlying volatity (stdev)
	t_max = Time till maturity (in interest periods)
	S_max = Maximum considered stock price (advised +4*sigma)
	N = Number of time (t) samples
	M = Number of stock price (S) samples
	lbound = function for lower boundary condition V(0,t)
	ubound = function for upper boundary condition V(S_max,t)
	matbound = function for maturity condition V(S, t_max)
	
	returns: t = (N-array) array of time steps
	returns: S = (M-array) array of stock prices
	returns: V = (N*M matrix) array of option prices
	"""

	#Find European pricing
	t, S, V = BlackScholes_Euro(r=r, K=K, sigma=sigma, t_max=t_max, S_max=S_max, N=N, M=M, bounds=bounds)

	#Calculate early stopping price
	#https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3068888
	return NotImplementedError()

	#return t, S, V



#example with butterfly option
if __name__ == '__main__':
	t, S, V = BlackScholes_Euro(r=0.015, K=[20,40], sigma=0.356, t_max=2.0, S_max=70, N=1800, M=80, bounds=boundsBackspread)

	import matplotlib.pyplot as plt
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.imshow(V, extent=(0,S[-1],0,t[-1]), cmap='viridis', aspect='auto', origin='lower', interpolation='none')

	ax2.plot(S, V[-1], label='t='+str(np.round(t[-1], 2)), color='black')
	ax2.plot(S, V[int(len(V)/1.2)], label='t='+str(np.round(t[int(len(t)/1.2)], 2)))
	ax2.plot(S, V[int(len(V)/1.5)], label='t='+str(np.round(t[int(len(t)/1.5)], 2)))
	ax2.plot(S, V[0], label='t='+str(np.round(t[0], 2)))
	ax2.legend(), plt.show()


	#plot underlying greeks
	plt.plot(S, V[0], label='V')
	plt.plot(S[0:-1], Delta(S, V[0]), label=r'$\Delta$')
	plt.plot(S[0:-2], Gamma(S, V[0]), label=r'$\Gamma$')
	plt.axhline(0, linestyle='--', color='grey')
	#plt.legend(), plt.show()

	#plot time greeks
	plt.plot(t, V[:,36], label='S='+str(np.round(S[36], 2)))
	plt.plot(t[0:-1], Theta(t, V[:,36]), label=r'$\Theta$')
	plt.axhline(0, linestyle='--', color='grey')
	#plt.legend(), plt.show()
