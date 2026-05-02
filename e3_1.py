import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 1000000

x1 = 4 * rng.random(N) # Uniform noise: mean 2, amplitude ±2 → [0, 4]

# Gaussian noise
x2 = rng.normal(loc=2, scale=np.sqrt(0.5), size=N)
x3 = rng.normal(loc=2, scale=np.sqrt(1.5), size=N)

print("x1 mean:", np.mean(x1), "variance:", np.var(x1))
print("x2 mean:", np.mean(x2), "variance:", np.var(x2))
print("x3 mean:", np.mean(x3), "variance:", np.var(x3))

edges = np.arange(-4, 8, 0.1)  # wide enough for all distributions
centers = 0.5 * (edges[1:] + edges[:-1])

h1, _ = np.histogram(x1, edges)
h2, _ = np.histogram(x2, edges)
h3, _ = np.histogram(x3, edges)

plt.figure()
plt.plot(centers, h1, label="Uniform")
plt.plot(centers, h2, label="Gaussian var=0.5")
plt.plot(centers, h3, label="Gaussian var=1.5")
plt.xlabel("x")
plt.ylabel("Counts")
plt.legend()
plt.show()

dx = edges[1] - edges[0]

p1 = h1 / np.sum(h1)
p2 = h2 / np.sum(h2)
p3 = h3 / np.sum(h3)

#c1 = np.cumsum(p1) * dx
#c2 = np.cumsum(p2) * dx
#c3 = np.cumsum(p3) * dx

c1 = np.cumsum(p1) 
c2 = np.cumsum(p2) 
c3 = np.cumsum(p3) 

plt.figure()
plt.plot(centers, c1, label="Uniform")
plt.plot(centers, c2, label="Gaussian var=0.5")
plt.plot(centers, c3, label="Gaussian var=1.5")
plt.xlabel("x")
plt.ylabel("CDF")
plt.legend()
plt.show()

idx = np.searchsorted(centers, 1)

P1 = 1 - c1[idx]
P2 = 1 - c2[idx]
P3 = 1 - c3[idx]

print("P1(x ≥ 1) ≈", P1)
print("P2(x ≥ 1) ≈", P2)
print("P3(x ≥ 1) ≈", P3)