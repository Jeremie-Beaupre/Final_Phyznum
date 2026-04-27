# Rapport de contexte — Final_Phyznum

> Generated automatically by analysis of all source files (2026-04-25).  
> Files analyzed: `Amazon.py`, `Black_shole.py`, `FFT_stock.py`, `Pepsi_and_coca.py`, `Tradding_lagged.py`, `Rapport.ipynb` (empty).  
> No Julia files found. No `.jl` files present in the repository.

---

## 1. Project overview

This project explores quantitative finance through two independent axes: (1) empirical analysis of real stock price time series retrieved from Yahoo Finance via the `stockdex` library (price visualization, Fourier analysis, a momentum trading strategy) and (2) numerical solution of the Black-Scholes PDE for European call option pricing using an implicit finite-difference scheme (fully implicit backward Euler) combined with a Gauss-Seidel iterative solver, supplemented by a Monte Carlo simulation of geometric Brownian motion.

### Main scripts and their roles

| File | Role |
|------|------|
| `Amazon.py` | Fetches 1 year of daily AMZN closing prices; prints and plots the time series. No analysis. |
| `Pepsi_and_coca.py` | Fetches 5 years of daily PEP, KO, and AMZN prices; produces a dual-axis comparison plot of Pepsi vs Coca-Cola. No correlation computed. |
| `FFT_stock.py` | Fetches 5 years of daily prices for a chosen ticker (default: COST); applies the DFT and plots the magnitude spectrum. |
| `Tradding_lagged.py` | Implements a 2-day momentum (lagged) trading strategy; compares it against a buy-and-hold baseline on BRK.B. |
| `Black_shole.py` | Solves the Black-Scholes PDE via implicit FD + Gauss-Seidel; then simulates Monte Carlo stock paths and computes the discounted expected payoff. |
| `Rapport.ipynb` | Empty notebook (0 bytes). |

---

## 2. Data

### Financial data used

| File | Tickers | Period | Granularity | Source |
|------|---------|--------|-------------|--------|
| `Amazon.py` | AMZN | 1 year | Daily | Yahoo Finance via `stockdex` |
| `Pepsi_and_coca.py` | PEP, KO, AMZN | 5 years | Daily | Yahoo Finance via `stockdex` |
| `FFT_stock.py` | COST (default; 50 tickers available) | 5 years | Daily | Yahoo Finance via `stockdex` |
| `Tradding_lagged.py` | BRK.B (default; 50 tickers available) | 5 years | Daily | Yahoo Finance via `stockdex` |
| `Black_shole.py` | None — synthetic data only | — | — | — |

### Available ticker universe (50 companies, defined in `FFT_stock.py` and `Tradding_lagged.py`, lines 5–56)

Sectors represented:
- **Technology:** AAPL, MSFT, AMZN, GOOGL, GOOG, META, TSLA, NVDA, INTC, CSCO, ADBE, NFLX, CRM, AVGO, ORCL, TXN, QCOM, AMD, IBM
- **Financials:** BRK-B, JPM, V, MA, GS, MS, AXP
- **Consumer staples:** JNJ, PG, KO, PEP, WMT, ABBV, MRK, COST
- **Consumer discretionary:** HD, MCD, NKE, SBUX
- **Energy:** CVX, XOM
- **Healthcare:** PFE
- **Industrials:** GE, F, GM, LMT, BA, MMM, CAT, UPS

### Data loading

All files use the same pattern (`stockdex` library):

```python
from stockdex import Ticker
ticker = Ticker(ticker="<SYMBOL>")
result = ticker.yahoo_api_price(range='5y', dataGranularity='1d')
price = result["close"].to_numpy()
```

The `"close"` column is extracted as a NumPy array. No preprocessing (normalization, log-returns, etc.) is applied before analysis in any file.

---

## 3. Correlation analysis

**No correlation analysis is implemented in any file of this project.**

`Pepsi_and_coca.py` plots PEP and KO on a dual-axis chart (visual comparison only), but does not compute Pearson, Spearman, or any other correlation coefficient. No clustering (k-means, hierarchical), no dendrogram, no Minimum Spanning Tree (MST), and no heatmap of a correlation matrix are present in the codebase.

---

## 4. Black-Scholes implementation

**File:** `Black_shole.py`

### Parameters

| Parameter | Symbol | Value | Line |
|-----------|--------|-------|------|
| Strike price | K | 100 | 6 |
| Risk-free rate | r | 0.10 (10% p.a.) | 7 |
| Volatility | σ | 0.20 (20% p.a.) | 8 |
| Time to maturity | T | 1 year | 9 |
| Max stock price on grid | S_max | 3K = 300 | 11 |
| Spatial grid points | M | 300 | 12 |
| Temporal grid points | N | 500 | 13 |
| Spatial step | ΔS | S_max / M = 1.0 | 15 |
| Temporal step | Δt | T / N = 0.002 yr | 16 |
| Initial spot price | S₀ | 100 | 73 |

The spatial grid is S_j = j·ΔS for j = 0, 1, …, M (301 points, 0 $ to 300 $).  
The time grid is t_i = i·Δt for i = 0, 1, …, N (501 points, 0 to 1 year).  
V[i, j] denotes the option value at time t_i and stock price S_j.

### Numerical method: fully implicit backward Euler finite differences

The Black-Scholes PDE (backward in calendar time t, with τ = T − t as time-to-maturity):

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0$$

is discretized on the uniform grid (S_j = j·ΔS, ΔS = 1) at interior nodes j = 1, …, M−1 using a **fully implicit (backward Euler) scheme in time** and **central differences in S**:

$$\frac{V_j^i - V_j^{i+1}}{\Delta t} + \frac{1}{2}\sigma^2 j^2 \frac{V_{j-1}^i - 2V_j^i + V_{j+1}^i}{(\Delta S)^2} + r j \frac{V_{j+1}^i - V_{j-1}^i}{2\,\Delta S} - r V_j^i = 0$$

Since ΔS = 1 exactly (by construction), this simplifies to the tridiagonal system solved at each backward time step:

$$\ell_j\, V_{j-1}^i + d_j\, V_j^i + u_j\, V_{j+1}^i = V_j^{i+1}$$

with coefficients (defined at lines 50–57):

```
j = np.arange(1, M)          # j = 1, ..., 299

A_j = 0.5 * sigma**2 * j**2   # diffusion coefficient  (line 52)
B_j = r * j                    # advection coefficient  (line 53)

lower_j = -dt * (A_j - B_j/2) # sub-diagonal          (line 55)
diag_j  =  1 + dt*(2*A_j + r) # main diagonal         (line 56)
upper_j = -dt * (A_j + B_j/2) # super-diagonal        (line 57)
```

Numerically with the given parameters (σ=0.2, r=0.1, Δt=0.002):

| j | A_j | B_j | lower_j | diag_j | upper_j |
|---|-----|-----|---------|--------|---------|
| 1 | 0.02 | 0.1 | −0.002·(0.02−0.05)=+0.00006 | 1+0.002·(0.04+0.1)=1.00028 | −0.002·(0.02+0.05)=−0.00014 |
| j | 0.02j² | 0.1j | −0.002(0.02j²−0.05j) | 1+0.002(0.04j²+0.1) | −0.002(0.02j²+0.05j) |

Time marching runs **backward** from i = N−1 down to i = 0 (lines 60–70).

### Linear solver: Gauss-Seidel iteration

At each time step, the tridiagonal system is solved iteratively (lines 33–48):

```
tol      = 1e-8
max_iter = 10 000

x[i] = (rhs[i] - lower[i]*x[i-1] - upper[i]*x_old[i+1]) / diag[i]

Convergence: max_j |x_j^(k) - x_j^(k-1)| < tol
```

Note: because the Gauss-Seidel update at node i uses the already-updated x[i−1] (lower sweep), this is the standard forward-sweep Gauss-Seidel, not a symmetric Gauss-Seidel. The initial guess x0 is taken as V[i+1, 1:M] (the previous time step solution).

### Boundary conditions

| Boundary | Condition | Code location | Physical meaning |
|----------|-----------|---------------|------------------|
| Final time (t = T) | V[-1, j] = max(S_j − K, 0) | Line 25 | European call payoff |
| S = 0 (j = 0) | V[i, 0] = 0 for all i | Line 28 | Call is worthless if S → 0 |
| S = S_max (j = M) | V[i, M] = S_max − K·e^(−r·τ_i) | Lines 29–31 | Deep in-the-money asymptote: C ≈ S − K·e^(−rτ) |

where τ_i = T − t_i is the time to maturity at time step i.

### Monte Carlo simulation (lines 119–154)

The Monte Carlo section re-uses the same market parameters (K=100, r=0.1, σ=0.2, T=1, S₀=100) with a different grid:

| Parameter | Value | Line |
|-----------|-------|------|
| N (MC time steps) | 800 | 119 |
| n_paths | 500 | 120 |
| Δt (MC) | T/N = 0.00125 yr | 122 |

Geometric Brownian Motion — Euler-Maruyama (log-Euler) scheme (lines 130–134):

$$S_{n+1}^{(k)} = S_n^{(k)} \cdot \exp\!\Bigl[\bigl(r - \tfrac{1}{2}\sigma^2\bigr)\Delta t + \sigma\sqrt{\Delta t}\; Z_n^{(k)}\Bigr], \quad Z_n^{(k)} \sim \mathcal{N}(0,1)$$

This is the exact simulation of GBM (no discretization error in the stock price distribution), also known as the log-normal scheme.

The discounted payoff is computed at lines 149–154:

$$W^{(k)} = e^{-rT}\,\max\!\bigl(S_T^{(k)} - K,\, 0\bigr) - \texttt{prix}_{\text{FD}}$$

where `prix_FD` is the finite-difference price at S₀=100. `mean(W)` is the difference between the MC option price estimate and the FD result.

### Option type

European call option. No American option (no early-exercise constraint) is implemented.

### Comparison with analytical solution

The code does **not** explicitly call the analytical Black-Scholes formula. The analytical price for a European call with K=100, S₀=100, r=0.10, σ=0.20, T=1 is (for reference):

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} = \frac{0 + 0.12}{0.20} = 0.60$$
$$d_2 = d_1 - \sigma\sqrt{T} = 0.40$$
$$C_{\text{BS}} = S_0\,\mathcal{N}(d_1) - K e^{-rT}\,\mathcal{N}(d_2) \approx 100 \times 0.7257 - 100\,e^{-0.1}\times 0.6554 \approx 13.27\;\$$$

The finite-difference price printed at line 75 (`prix`) can be compared manually against this reference. No convergence study (varying M, N) is implemented.

---

## 5. Figures generated

### `Amazon.py`

| # | Variable/label | What is shown | File + line |
|---|----------------|---------------|-------------|
| 1 | `price` | Time series of AMZN daily closing prices over 1 year; x-axis = trading day index (implicit), y-axis = price ($) | `Amazon.py` lines 15–16 |

### `Pepsi_and_coca.py`

| # | Variable/label | What is shown | File + line |
|---|----------------|---------------|-------------|
| 2 | `price_pepsi` / `price_coca` | Dual-axis comparison: left y-axis = PEP price ($) in blue, right y-axis = KO price ($) in red; x-axis = trading day index over 5 years; title = "Pepsi vs Coca-Cola Stock Prices" | `Pepsi_and_coca.py` lines 21–40 |

### `FFT_stock.py`

| # | Variable/label | What is shown | File + line |
|---|----------------|---------------|-------------|
| 3 | `price` | Time series of COST daily closing prices over 5 years; x-axis = trading day index, y-axis = price ($) | `FFT_stock.py` lines 78–79 |
| 4 | `freq` vs `magnitude` | Magnitude spectrum of the DFT; x-axis = frequency bin k (centered, range ≈ [−N/2, N/2]); y-axis = |F[k]|; scatter plot in green; title = ticker symbol ("COST") | `FFT_stock.py` lines 82–88 |

### `Tradding_lagged.py`

| # | Variable/label | What is shown | File + line |
|---|----------------|---------------|-------------|
| 5 | `portfolio_strat` / `buy_hold` | Portfolio value ($) over 5 years (~1 250 trading days); green = lagged momentum strategy (initial $2 000), blue = buy-and-hold rescaled to $2 000; title = ticker symbol ("BRK.B") | `Tradding_lagged.py` lines 109–116 |

### `Black_shole.py`

| # | Variable/label | What is shown | File + line |
|---|----------------|---------------|-------------|
| 6 | `V[0, :]` vs `S` | European call option value at t = 0 (initial time); x-axis = S ∈ [0, 300] ($), y-axis = V(S, 0) ($); classic hockey-stick shape convex curve | `Black_shole.py` lines 78–83 |
| 7 | `V.T` (imshow) | 2-D heatmap of option value surface V(t, S); x-axis = t ∈ [0, 1] (years), y-axis = S ∈ [0, 300] ($), colorbar = V ($); title = "Solution de Black-Scholes avec relaxation" | `Black_shole.py` lines 86–97 |
| 8 | `V.T` (plot_surface) | 3-D surface of V(S, t); x-axis = t [années], y-axis = S [$], z-axis = V(S,t) [$]; colormap "viridis"; title = "Surface 3D de la solution Black-Scholes" | `Black_shole.py` lines 101–114 |
| 9 | `S[:50].T` | 50 Monte Carlo trajectories of stock price S(t); x-axis = t ∈ [0, 1] (years), y-axis = S(t) ($); title = "Simulation Monte Carlo d'une action" | `Black_shole.py` lines 137–143 |

---

## 6. Key equations implemented

### 6.1 Black-Scholes PDE — fully implicit discretization (`Black_shole.py`)

The PDE discretized on the uniform grid (ΔS = 1 by design):

$$\ell_j\, V_{j-1}^i + d_j\, V_j^i + u_j\, V_{j+1}^i = V_j^{i+1} \qquad j = 1,\ldots,M{-}1,\quad i = N{-}1,\ldots,0$$

with:

$$\ell_j = -\Delta t\!\left(\tfrac{1}{2}\sigma^2 j^2 - \tfrac{r j}{2}\right), \quad d_j = 1 + \Delta t\!\left(\sigma^2 j^2 + r\right), \quad u_j = -\Delta t\!\left(\tfrac{1}{2}\sigma^2 j^2 + \tfrac{r j}{2}\right)$$

The boundary corrections applied to the RHS before solving (lines 65–66):

$$\text{rhs}[0] \mathrel{-}= \ell_1\cdot V^i_0, \qquad \text{rhs}[M{-}2] \mathrel{-}= u_{M-1}\cdot V^i_M$$

### 6.2 Gauss-Seidel sweep (`Black_shole.py`, lines 40–46)

For k = 0, 1, 2, … until convergence:

$$x_j^{(k+1)} = \frac{\text{rhs}_j - \ell_j\, x_{j-1}^{(k+1)} - u_j\, x_{j+1}^{(k)}}{d_j}, \qquad j = 1,\ldots,M{-}1$$

Convergence criterion: $\max_j |x_j^{(k+1)} - x_j^{(k)}| < 10^{-8}$.

### 6.3 Geometric Brownian Motion — log-Euler scheme (`Black_shole.py`, lines 130–134)

$$S_{n+1}^{(k)} = S_n^{(k)} \cdot \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\;Z_n^{(k)}\right], \quad Z_n^{(k)} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$$

This scheme is **exact** for GBM (the distribution of S_T is exactly log-normal), so there is no time-discretization error in the terminal stock price.

### 6.4 Monte Carlo option price estimate (`Black_shole.py`, lines 149–154)

$$\hat{C}_{\text{MC}} = e^{-rT} \cdot \frac{1}{n_{\text{paths}}} \sum_{k=1}^{n_{\text{paths}}} \max\!\bigl(S_T^{(k)} - K,\, 0\bigr)$$

The code computes W[k] = e^(−rT)·max(S_T^(k) − K, 0) − prix_FD and prints mean(W), which equals Ĉ_MC − prix_FD (the difference between the MC and FD prices).

### 6.5 Discrete Fourier Transform (`FFT_stock.py`, lines 68–71)

$$F[k] = \sum_{n=0}^{N-1} \text{price}[n]\, e^{-2\pi i k n / N}$$

implemented as:

```python
price_fft = np.fft.fftshift(np.fft.fft(price))
freq      = np.fft.fftshift(np.fft.fftfreq(len(price))) * len(price)
magnitude = np.abs(price_fft)
phase     = np.angle(price_fft)
```

The `fftshift` reorders the output so that k = 0 (DC component) is at the center of the array. The frequency axis is rescaled by N so that `freq` runs from −N/2 to N/2 in units of cycles per N days.

### 6.6 Lagged momentum trading rule (`Tradding_lagged.py`, lines 79–93)

Signal at day i (using prices at i−1 and i−2):

$$\text{signal}_i = \text{sign}(\text{price}_{i-1} - \text{price}_{i-2})$$

- **+1 (BUY ALL):** if price_{i−1} > price_{i−2}: shares = cash / price_i; cash = 0
- **−1 (SELL ALL):** if price_{i−1} < price_{i−2}: cash = shares × price_i; shares = 0
- **0 (HOLD):** if price_{i−1} = price_{i−2}: no action

Portfolio value: V_i = cash_i + shares_i × price_i

Buy-and-hold baseline: V_i^(BH) = (price_i / price_0) × 2000

### 6.7 Deviations from standard Black-Scholes assumptions

| Assumption | Standard | This code |
|------------|----------|-----------|
| Spatial grid | Often log-transformed (x = ln S) | Uniform in S |
| Scheme | Often Crank-Nicolson (2nd-order in time) | Fully implicit backward Euler (1st-order in time) |
| Linear solver | Direct tridiagonal (Thomas algorithm) | Iterative Gauss-Seidel |
| Dividends | None | None (correct) |
| American early exercise | N/A | Not implemented |
| Analytical comparison | Often included | Not implemented in code |

---

## 7. Results and numbers

### 7.1 Hardcoded numerical parameters

| Quantity | Value | File | Line |
|----------|-------|------|------|
| K (strike) | 100 $ | `Black_shole.py` | 6 |
| r (rate) | 0.10 | `Black_shole.py` | 7 |
| σ (vol) | 0.20 | `Black_shole.py` | 8 |
| T (maturity) | 1 yr | `Black_shole.py` | 9 |
| S_max | 300 $ | `Black_shole.py` | 11 |
| M (FD spatial pts) | 300 | `Black_shole.py` | 12 |
| N (FD time pts) | 500 | `Black_shole.py` | 13 |
| ΔS | 1.0 $ | `Black_shole.py` | 15 |
| Δt (FD) | 0.002 yr | `Black_shole.py` | 16 |
| S₀ | 100 $ | `Black_shole.py` | 73 |
| N (MC time steps) | 800 | `Black_shole.py` | 119 |
| n_paths (MC) | 500 | `Black_shole.py` | 120 |
| Δt (MC) | 0.00125 yr | `Black_shole.py` | 122 |
| Gauss-Seidel tol | 1×10⁻⁸ | `Black_shole.py` | 33 |
| Gauss-Seidel max_iter | 10 000 | `Black_shole.py` | 33 |
| Initial cash (strategy) | 2 000 $ | `Tradding_lagged.py` | 61 |

### 7.2 Computed results printed to stdout (`Black_shole.py`)

| Print statement | Formula | Expected numerical value |
|----------------|---------|--------------------------|
| `Prix du call à t=0 pour S0=100 : {prix:.4f} $` (line 75) | `np.interp(100, S, V[0,:])` — FD interpolated option price | ~13.xx $ (close to analytical ≈ 13.27 $) |
| `Prix moyen final : {np.mean(S[:,-1]):.2f} $` (line 146) | Mean of 500 terminal MC stock prices | ~110.xx $ (random; expectation = 100·e^0.1 ≈ 110.52 $) |
| `Prix final théorique espéré : {S0*np.exp(r*T):.2f} $` (line 147) | S₀·e^(rT) = 100·e^0.1 | **110.52 $** (deterministic) |
| `{np.mean(W)}` (line 154) | mean(e^(−rT)·max(S_T−K,0)) − prix_FD | ~0 if MC and FD agree (random; expectation ≈ 0) |

### 7.3 Analytical reference values (not in code, for report verification)

| Quantity | Formula | Value |
|----------|---------|-------|
| d₁ | [ln(S₀/K) + (r + σ²/2)T] / (σ√T) | 0.60 |
| d₂ | d₁ − σ√T | 0.40 |
| N(d₁) | Φ(0.60) | ≈ 0.7257 |
| N(d₂) | Φ(0.40) | ≈ 0.6554 |
| C_BS (European call) | S₀·N(d₁) − K·e^(−rT)·N(d₂) | ≈ **13.27 $** |
| S₀·e^(rT) | 100·e^0.1 | **110.52 $** |

### 7.4 Grid stability (Von Neumann / CFL analysis — not in code)

For the implicit scheme, the method is unconditionally stable for all Δt > 0. However, note that:
- Spatial Courant number: r·j·Δt / (2·ΔS²) — can be large for large j; the implicit scheme handles this.
- The condition ΔS = 1, Δt = 0.002 gives a ratio Δt/ΔS² = 0.002, well within reasonable bounds.
