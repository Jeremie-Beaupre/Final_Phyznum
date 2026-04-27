"""
signal_filter.py — Module 3 : Filtrage Fréquentiel (FFT + Butterworth passe-bande)
Pipeline AlphaPulse
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def compute_power_spectrum(signal_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la densité spectrale de puissance (PSD) via FFT.

    Parameters
    ----------
    signal_series : pd.Series
        Série temporelle (sans NaN).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (freqs, power) : fréquences (en cycles/jour) et puissance correspondante.
        Seules les fréquences positives sont retournées.
    """
    clean = signal_series.dropna().values
    n = len(clean)

    # FFT et fréquences (fs = 1 jour)
    fft_vals = np.fft.rfft(clean)
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles par jour

    # Puissance = |FFT|² normalisée
    power = (np.abs(fft_vals) ** 2) / n

    return freqs, power


def find_spectral_breakpoints(
    freqs: np.ndarray,
    power: np.ndarray,
) -> dict:
    """
    Détecte les ruptures de pente dans le spectre log-log pour suggérer
    les bornes naturelles du filtre passe-bande.

    Parameters
    ----------
    freqs : np.ndarray
        Fréquences (cycles/jour), issues de compute_power_spectrum.
    power : np.ndarray
        Puissance correspondante.

    Returns
    -------
    dict avec clés :
        - "low_freq"  : borne basse suggérée (cycles/jour)
        - "high_freq" : borne haute suggérée (cycles/jour)
        - "low_days"  : période correspondante à low_freq (jours)
        - "high_days" : période correspondante à high_freq (jours)
    """
    # Ignorer la fréquence zéro (tendance DC)
    mask = freqs > 0
    f = freqs[mask]
    p = power[mask]

    # Travailler en log-log pour détecter les ruptures de pente
    log_f = np.log10(f)
    log_p = np.log10(p + 1e-30)

    # Dérivée discrète de la pente (différences secondes = courbure)
    deriv2 = np.diff(np.diff(log_p))

    # Identifier les deux extrema de courbure les plus prononcés
    # comme candidats aux ruptures de pente
    idx_low = int(np.argmin(deriv2[:len(deriv2)//2])) + 1
    idx_high = int(np.argmax(deriv2[len(deriv2)//2:])) + len(deriv2)//2 + 1

    low_freq = float(f[idx_low])
    high_freq = float(f[idx_high])

    # S'assurer que low < high et que les valeurs sont physiquement sensées
    if low_freq >= high_freq:
        low_freq = float(f[max(0, idx_low - 5)])
        high_freq = float(f[min(len(f) - 1, idx_high + 5)])

    low_days = 1.0 / high_freq if high_freq > 0 else 252.0
    high_days = 1.0 / low_freq if low_freq > 0 else 3.0

    result = {
        "low_freq": low_freq,
        "high_freq": high_freq,
        "low_days": low_days,
        "high_days": high_days,
    }
    logger.info(
        "Ruptures spectrales suggérées : [%.1f j, %.1f j] (freqs: %.4f, %.4f c/j)",
        low_days,
        high_days,
        low_freq,
        high_freq,
    )
    return result


def apply_bandpass_filter(
    signal_series: pd.Series,
    low_days: float,
    high_days: float,
    fs: float = 1.0,
    order: int = 4,
) -> pd.Series:
    """
    Applique un filtre Butterworth passe-bande sur la série temporelle.

    Parameters
    ----------
    signal_series : pd.Series
        Série temporelle (les NaN en tête sont gérés).
    low_days : float
        Période minimale à conserver (coupure haute en fréquence).
        Ex. : 3 → élimine les oscillations < 3 jours.
    high_days : float
        Période maximale à conserver (coupure basse en fréquence).
        Ex. : 252 → élimine les tendances > 1 an.
    fs : float
        Fréquence d'échantillonnage (1.0 = 1 point par jour).
    order : int
        Ordre du filtre Butterworth (défaut : 4).

    Returns
    -------
    pd.Series
        Signal filtré, même index que l'entrée (NaN préservés en tête).
    """
    # Travailler sur les valeurs sans NaN initiaux
    valid_mask = ~signal_series.isna()
    valid_data = signal_series[valid_mask].values

    # Convertir les périodes (jours) en fréquences normalisées (0, 1]
    # f_normalisee = f_reelle / (fs/2) = (1/periode) / (1/2) = 2/periode
    nyquist = fs / 2.0
    low_cut = 1.0 / high_days   # Coupure basse (élimine tendances lentes)
    high_cut = 1.0 / low_days   # Coupure haute (élimine bruit rapide)

    low_norm = low_cut / nyquist
    high_norm = high_cut / nyquist

    # Borner pour éviter les valeurs hors (0, 1)
    low_norm = np.clip(low_norm, 1e-6, 0.99)
    high_norm = np.clip(high_norm, 1e-6, 0.99)

    if low_norm >= high_norm:
        logger.warning(
            "Bornes du filtre incohérentes (low=%.4f >= high=%.4f). "
            "Vérifier low_days=%g et high_days=%g.",
            low_norm, high_norm, low_days, high_days,
        )
        return signal_series.copy()

    logger.info(
        "Filtre Butterworth ordre %d : [%.1f j, %.1f j] (freq normalisées: %.4f, %.4f)",
        order, low_days, high_days, low_norm, high_norm,
    )

    try:
        b, a = scipy_signal.butter(order, [low_norm, high_norm], btype="bandpass")
        # filtfilt : filtre à phase nulle (pas de décalage temporel)
        filtered = scipy_signal.filtfilt(b, a, valid_data)
    except Exception as exc:
        logger.error("Erreur Butterworth : %s", exc)
        return signal_series.copy()

    # Reconstruire la série avec le même index
    result = signal_series.copy()
    result[valid_mask] = filtered
    return result


def plot_filter_analysis(
    ticker: str,
    raw_alpha: pd.Series,
    filtered_alpha: pd.Series,
    freqs: np.ndarray,
    power: np.ndarray,
    low_days: float,
    high_days: float,
    breakpoints: dict | None = None,
    save_path: str | None = None,
) -> None:
    """
    Génère un graphique en 2 panneaux :
    - Panneau 1 : spectre de puissance (FFT) annoté des bornes du filtre
    - Panneau 2 : signal brut vs signal filtré

    Parameters
    ----------
    ticker : str
        Nom de l'actif.
    raw_alpha : pd.Series
        Alpha résiduel brut (entrée du filtre).
    filtered_alpha : pd.Series
        Alpha filtré (sortie du filtre).
    freqs : np.ndarray
        Fréquences du spectre (cycles/jour).
    power : np.ndarray
        Puissance du spectre.
    low_days : float
        Borne basse du filtre (jours).
    high_days : float
        Borne haute du filtre (jours).
    breakpoints : dict | None
        Ruptures spectrales détectées automatiquement (pour annotation).
    save_path : str | None
        Chemin de sauvegarde PNG.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, f"filter_{ticker}.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # --- Panneau 1 : Spectre de puissance ---
    ax1 = axes[0]
    mask_pos = freqs > 0
    periods = np.where(freqs[mask_pos] > 0, 1.0 / freqs[mask_pos], np.inf)

    ax1.semilogy(periods, power[mask_pos], color="navy", alpha=0.7, linewidth=0.8)
    ax1.axvline(low_days, color="red", linestyle="--", linewidth=1.5,
                label=f"Borne basse filtre : {low_days:.0f} j")
    ax1.axvline(high_days, color="orange", linestyle="--", linewidth=1.5,
                label=f"Borne haute filtre : {high_days:.0f} j")

    if breakpoints:
        ax1.axvline(breakpoints["low_days"], color="green", linestyle=":",
                    linewidth=1.2, label=f"Rupture spectrale basse : {breakpoints['low_days']:.0f} j")
        ax1.axvline(breakpoints["high_days"], color="purple", linestyle=":",
                    linewidth=1.2, label=f"Rupture spectrale haute : {breakpoints['high_days']:.0f} j")

    ax1.set_xlabel("Période (jours)")
    ax1.set_ylabel("Puissance (log)")
    ax1.set_title(f"{ticker} — Spectre de puissance (FFT) | Bornes filtre : [{low_days:.0f}j, {high_days:.0f}j]")
    ax1.set_xlim(1, 500)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panneau 2 : Signal brut vs filtré ---
    ax2 = axes[1]
    valid = filtered_alpha.dropna()
    raw_aligned = raw_alpha.loc[valid.index]

    ax2.plot(raw_aligned.index, raw_aligned.values, label="Alpha brut",
             alpha=0.5, color="steelblue", linewidth=0.8)
    ax2.plot(valid.index, valid.values, label="Alpha filtré",
             alpha=0.9, color="crimson", linewidth=1.1)

    ax2.set_title(f"{ticker} — Alpha brut vs Alpha filtré (Butterworth [{low_days:.0f}j–{high_days:.0f}j])")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Alpha résiduel")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Graphique sauvegardé : %s", save_path)


# ---------------------------------------------------------------------------
# Validation autonome
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import config
    from data_engine import load_data
    from alpha_denoiser import AlphaDeNoiser

    print("\n=== Validation — signal_filter.py ===\n")

    # Charger données et extraire alphas
    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=config.START_DATE,
        end=config.END_DATE,
    )
    denoiser = AlphaDeNoiser(window=config.ROLLING_WINDOW)
    alphas = denoiser.extract_alpha(data, data[config.MARKET_TICKER]["log_return"], config.SECTOR_MAP)

    # Travailler sur l'alpha de PEP
    pep_alpha = alphas["PEP"].dropna()

    # 1. Spectre de puissance
    freqs, power = compute_power_spectrum(pep_alpha)
    print(f"Spectre calculé : {len(freqs)} fréquences, f_max={freqs[-1]:.4f} c/j")

    # 2. Ruptures spectrales
    breakpoints = find_spectral_breakpoints(freqs, power)
    print(f"Ruptures spectrales : {breakpoints}")

    # 3. Filtre avec les paramètres de config
    pep_filtered = apply_bandpass_filter(
        pep_alpha,
        low_days=config.FILTER_LOW_DAYS,
        high_days=config.FILTER_HIGH_DAYS,
    )
    print(f"\nAlpha PEP filtré : {pep_filtered.dropna().shape[0]} points valides")

    # 4. Graphique comparatif
    print("\nGeneration du graphique comparatif...")
    plot_filter_analysis(
        ticker="PEP",
        raw_alpha=pep_alpha,
        filtered_alpha=pep_filtered,
        freqs=freqs,
        power=power,
        low_days=config.FILTER_LOW_DAYS,
        high_days=config.FILTER_HIGH_DAYS,
        breakpoints=breakpoints,
    )
    print(f"Graphique sauvegarde dans : {OUTPUTS_DIR}/filter_PEP.png")

    # Validation : le filtre ne doit pas retourner une série entièrement nulle
    assert pep_filtered.dropna().abs().sum() > 0, "ERREUR : signal filtré entièrement nul !"
    print("\nValidation OK — signal filtré non nul.")
