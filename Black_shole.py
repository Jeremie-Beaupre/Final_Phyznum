import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres
K = 100
r = 0.1
sigma = 0.20
T = 1

Smax = 3*K
M = 300
N = 500

dS = Smax / M
dt = T / N

S = np.linspace(0, Smax, M+1)
t = np.linspace(0, T, N+1)

# V[i, j] = V(t_i, S_j)
V = np.zeros((N+1, M+1))

# Condition finale : payoff à t = T
V[-1, :] = np.maximum(S - K, 0)

# Conditions frontières
V[:, 0] = 0
for i in range(N+1):
    tau = T - t[i]
    V[i, -1] = Smax - K*np.exp(-r*tau)

def gauss_seidel(lower, diag, upper, rhs, x0, tol=1e-8, max_iter=10000):
    x = x0.copy()
    n = len(x)

    for _ in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            left = lower[i] * x[i-1] if i > 0 else 0
            right = upper[i] * x_old[i+1] if i < n-1 else 0
            x[i] = (rhs[i] - left - right) / diag[i]

        if np.max(np.abs(x - x_old)) < tol:
            break

    return x

j = np.arange(1, M)

A = 0.5 * sigma**2 * j**2
B = r * j

lower = -dt * (A - B/2)
diag  = 1 + dt * (2*A + r)
upper = -dt * (A + B/2)

# On remonte le temps : T -> 0
for i in range(N-1, -1, -1):

    rhs = V[i+1, 1:M].copy()

    # frontières au temps t_i
    rhs[0]  -= lower[0] * V[i, 0]
    rhs[-1] -= upper[-1] * V[i, M]

    x0 = V[i+1, 1:M].copy()

    V[i, 1:M] = gauss_seidel(lower, diag, upper, rhs, x0)

# Prix pour S0
S0 = 100
prix = np.interp(S0, S, V[0, :])
print(f"Prix du call à t=0 pour S0={S0} : {prix:.4f} $")

# Courbe à t=0
plt.figure()
plt.plot(S, V[0, :])
plt.xlabel("Prix de l'action S")
plt.ylabel("Prix de l'option V(S,0)")
plt.grid(True)
plt.show()

# Carte 2D V(t,S)
plt.figure()
plt.imshow(
    V.T,
    extent=[0, T, 0, Smax],
    origin="lower",
    aspect="auto"
)
plt.colorbar(label="V(S,t)")
plt.xlabel("Temps t")
plt.ylabel("Prix de l'action S")
plt.title("Solution de Black-Scholes avec relaxation")
plt.show()



# Grille 2D pour le graphique 3D
T_grid, S_grid = np.meshgrid(t, S)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(T_grid, S_grid, V.T, cmap="viridis")

ax.set_xlabel("Temps t [années]")
ax.set_ylabel("Prix de l'action S [$]")
ax.set_zlabel("Prix de l'option V(S,t) [$]")
ax.set_title("Surface 3D de la solution Black-Scholes")

plt.show()




N = 800          # nombre de pas de temps
n_paths = 500  # nombre de simulations Monte Carlo

dt = T / N
t = np.linspace(0, T, N+1)

# Matrice des prix simulés
S = np.zeros((n_paths, N+1))
S[:, 0] = S0

# Simulation Monte Carlo
for i in range(N):
    Z = np.random.normal(0, 1, n_paths)
    S[:, i+1] = S[:, i] * np.exp(
        (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    )

# Graphique de quelques trajectoires
plt.figure()
plt.plot(t, S[:50].T)
plt.xlabel("Temps [années]")
plt.ylabel("Prix de l'action S(t) [$]")
plt.title("Simulation Monte Carlo d'une action")
plt.grid(True)
plt.show()

# Prix final moyen
print(f"Prix moyen final : {np.mean(S[:, -1]):.2f} $")
print(f"Prix final théorique espéré : {S0*np.exp(r*T):.2f} $")

W = []
for i in S[:, -1]:
    profit = max(i - K, 0)
    W.append(np.exp(-r*T)*profit - prix)

print(np.mean(W))




