import numpy as np
from math import erf















if __name__ == '__main__':
	import matplotlib.pyplot as plt

	#define MGs
	S1 = MG(state=36, mu=0.015*(36/365), sigma=0.24)
	S2 = MG(state=45, mu=0.09*(45/365), sigma=0.88, cov={S1.id:0.011})
	S3 = MG(state=51, mu=0.10*(51/365), sigma=1.18, cov={S2.id:-0.016})
	S4 = MG(state=48, mu=0.07*(48/365), sigma=0.88, cov={S2.id:0.012, S3.id:0.014})

	#define portfolio
	print(Markowitz_ef((S1, S2, S3, S4), mu=0.075/365, weights=True))
	Pf = Markowitz_ef((S1, S2, S3, S4), cap=5000, mu=0.075/365) 

	#define timeseries
	TS = timeSeries((S1, S2, S3, S4, Pf))
	P, dP = TS.propagate(N=150)

	#plot
	fig, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.plot(P[0], label='S1', linestyle='--', linewidth=1)
	ax1.plot(P[1], label='S2', linestyle='--', linewidth=1)
	ax1.plot(P[2], label='S3', linestyle='--', linewidth=1)
	ax1.plot(P[3], label='S4', linestyle='--', linewidth=1)
	ax2.plot(P[4], label='Portfolio', color='black')
	plt.tight_layout(), ax1.legend(), ax2.legend(), plt.show()

