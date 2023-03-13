import pyfinancetools as ft
import numpy as np
import matplotlib.pyplot as plt



#define MGs
S1 = ft.MG(state=36, mu=0.015*(36/365), sigma=0.24)
S2 = ft.MG(state=45, mu=0.09*(45/365), sigma=0.88, cov={S1.id:0.011})
S3 = ft.MG(state=51, mu=0.10*(51/365), sigma=1.18, cov={S2.id:-0.016})
S4 = ft.MG(state=48, mu=0.07*(48/365), sigma=0.88, cov={S2.id:0.012, S3.id:0.014})

#define portfolio
print(ft.Markowitz_ef((S1, S2, S3, S4), mu=0.075/365, weights=True))
Pf = ft.Markowitz_ef((S1, S2, S3, S4), cap=5000, mu=0.075/365) 

#define timeseries
TS = ft.timeSeries((S1, S2, S3, S4, Pf))
P, dP = TS.propagate(N=150)


print(ft.Delta(und=S1, der=Pf))


#define option
Opt = ft.option(S1, bounds=ft.option.boundsStraddle)
Opt.pricing(r=0.015, K=36, r_period=150, t_max=3, S_max=60, N=3000, M=200)
V = Opt.toSeries(P[0])

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(P[0], label=r'stock price $P$')
ax2.plot(V, label=r'option price $V$')
ax2.plot(ft.Delta(und=P[0], der=Opt), label=r'Delta $\frac{\partial P}{\partial V}$')
plt.tight_layout(), ax1.legend(), ax2.legend(), plt.show()