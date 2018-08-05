import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
import BatchSteepestGradientModel as BSGM
import NewtonBasedModel as NBM

np.random.seed(42) # If value of seed is same,, result is always same.

def J(W, X, y, gamma):
	return np.log(1 + np.exp(-y * W.T.dot(X))).sum() + gamma * W.T.dot(W)

gamma = 5
eta = 1 / (2 * 10 * gamma)

X, y = dataset.dataset2()

bsg = BSGM(eta = eta, gamma = gamma)
nb = NBM(eta = eta, gamma = gamma)

W_BSG = bsg.fit(X, y)
W_NB = nb.fit(X, y)

W_BSG_J = np.array([J(W, X.T, y.reshape(1, -1), gamma).reshape(-1) for W in W_BSG])
W_NB_J = np.array([J(W, X.T, y.reshape(1, -1), gamma).reshape(-1) for W in W_NB])

plt.plot(range(100), W_BSG_J[:100], "r", label="Batch Steepest Gradient Method")
plt.plot(range(100), W_NB_J[:100], "b", label="Newton Based Method")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("$J(W)$")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu")
plt.show()

print(BSG.score(X, y), NB.score(X, y))
