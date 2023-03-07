import numpy as np
from matplotlib import pyplot as plt

from options.black_scholes import BlackScholes_Euro, boundsButterfly
from stocks.gaussian_copula import stock




#stock
N = int(12000/100)
X = np.arange(0,N)
Stock = stock(init_price=75.74, mu=0.00167593, sigma=1.89649152648)
Stock.propegate(N-1)
Stock.summary()

#option
t, S, V = BlackScholes_Euro(r=0.015, K=[60, 80], sigma=0.29649152648, t_max=1.0, S_max=150, N=12000, M=100, bounds=boundsButterfly)
V = np.delete(V, [i for i in range(len(t)) if i%100 != 0], axis=0)
t = np.delete(t, [i for i in range(len(t)) if i%100 != 0])



"""
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.imshow(V, extent=(0,S[-1],0,t[-1]), cmap='viridis', aspect='auto', origin='lower', interpolation='none')

ax2.plot(S, V[-1], label='t='+str(np.round(t[-1], 2)), color='black')
ax2.plot(S, V[int(len(V)/1.2)], label='t='+str(np.round(t[int(len(t)/1.2)], 2)))
ax2.plot(S, V[int(len(V)/1.5)], label='t='+str(np.round(t[int(len(t)/1.5)], 2)))
ax2.plot(S, V[0], label='t='+str(np.round(t[0], 2)))
ax2.legend(), plt.show()
"""

#plot
S_indices = [np.argmin(np.abs(S-np.full(len(S), x))) for x in Stock.price]
V_prices = [V[i,j] for i,j in zip(range(len(t)), S_indices)]

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.axhline(65, label='K1', linestyle='--', color='grey')
ax1.axhline(80, label='K2', linestyle='--', color='grey')
ax1.plot(X, Stock.price, label='stock', color='black')

ax2.plot(X, V_prices, label='option', color='red')
ax1.legend(), ax2.legend(), plt.show()
