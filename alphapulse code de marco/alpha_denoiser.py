"""
alpha_denoiser.py — Module 2 : Isolation de l'Alpha Pur (OLS Roulant)
Pipeline AlphaPulse
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour sauvegarde PNG

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


class AlphaDeNoiser:
    """
    Calcule le bêta roulant d'un actif par rapport à un benchmark
    et extrait l'Alpha résiduel pur par débruitage séquentiel :
    1. Retire le mouvement de marché (VTI)
    2. Retire le mouvement sectoriel sur les résidus obtenus
    """

    def __init__(self, window: int = 60) -> None:
        """
        Parameters
        ----------
        window : int
            Taille de la fenêtre roulante pour l'OLS (en jours de bourse).
            Les `window - 1` premières valeurs retournées seront NaN.
        """
        self.window = window

    def fit_rolling_beta(
        self,
        asset_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.Series:
        """
        Calcule le bêta roulant via régression OLS sur une fenêtre glissante.

        Parameters
        ----------
        asset_returns : pd.Series
            Rendements logarithmiques de l'actif.
        benchmark_returns : pd.Series
            Rendements logarithmiques du benchmark (marché ou ETF sectoriel).

        Returns
        -------
        pd.Series
            Bêta roulant (NaN pour les `window - 1` premiers points).
        """
        betas: list[float] = []
        n = len(asset_returns)

        for i in range(n):
            if i < self.window - 1:
                betas.append(np.nan)
                continue

            y = asset_returns.iloc[i - self.window + 1 : i + 1].values
            x = benchmark_returns.iloc[i - self.window + 1 : i + 1].values

            # Supprimer les NaN éventuels dans la fenêtre
            mask = ~(np.isnan(y) | np.isnan(x))
            if mask.sum() < 10:  # Fenêtre trop courte après nettoyage
                betas.append(np.nan)
                continue

            try:
                X = sm.add_constant(x[mask], has_constant="add")
                model = sm.OLS(y[mask], X).fit()
                betas.append(float(model.params[1]))  # coefficient de pente = bêta
            except Exception as exc:
                logger.warning("OLS échoué à i=%d : %s", i, exc)
                betas.append(np.nan)

        return pd.Series(betas, index=asset_returns.index, name="beta")

    def extract_alpha(
        self,
        data: dict[str, pd.DataFrame],
        market_returns: pd.Series,
        sector_map: dict[str, str],
    ) -> dict[str, pd.Series]:
        """
        Extrait l'Alpha résiduel pur pour chaque ticker en deux passes :
        1. Soustrait le mouvement de marché (β_marché × VTI)
        2. Soustrait le mouvement sectoriel sur les résidus (β_sectoriel × ETF)

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Dictionnaire {ticker: DataFrame} issu de data_engine.load_data.
        market_returns : pd.Series
            Rendements logarithmiques du marché de référence (VTI).
        sector_map : dict[str, str]
            Dictionnaire {ticker: etf_sectoriel}.

        Returns
        -------
        dict[str, pd.Series]
            Dictionnaire {ticker: pd.Series} des alphas résiduels purs.
            Les `window - 1` premiers points sont NaN (documenté).
        """
        alphas: dict[str, pd.Series] = {}

        for ticker, etf in sector_map.items():
            if ticker not in data:
                logger.warning("Ticker '%s' absent de data, ignoré.", ticker)
                continue
            if etf not in data:
                logger.warning("ETF sectoriel '%s' absent de data, ignoré.", etf)
                continue

            asset_lr = data[ticker]["log_return"]

            # --- Passe 1 : débruitage marché ---
            logger.info("Ticker '%s' — Passe 1 : bêta marché (VTI).", ticker)
            beta_market = self.fit_rolling_beta(asset_lr, market_returns)

            # Résidu 1 : rendement actif - β_marché × VTI
            residual_1 = asset_lr - beta_market * market_returns

            # --- Passe 2 : débruitage sectoriel sur les résidus ---
            logger.info("Ticker '%s' — Passe 2 : bêta sectoriel (%s).", ticker, etf)
            sector_returns = data[etf]["log_return"]
            beta_sector = self.fit_rolling_beta(residual_1, sector_returns)

            # Alpha pur : résidu 1 - β_sectoriel × ETF
            alpha = residual_1 - beta_sector * sector_returns
            alpha.name = f"alpha_{ticker}"

            nan_count = alpha.isna().sum()
            logger.info(
                "Ticker '%s' : alpha extrait. NaN=%d (attendus : %d premiers jours).",
                ticker,
                nan_count,
                self.window - 1,
            )

            alphas[ticker] = alpha

        return alphas


def plot_alpha_vs_returns(
    ticker: str,
    raw_returns: pd.Series,
    alpha: pd.Series,
    save_path: str | None = None,
) -> None:
    """
    Trace le rendement brut de l'actif vs son Alpha résiduel sur la même période.

    Parameters
    ----------
    ticker : str
        Nom de l'actif (pour le titre du graphique).
    raw_returns : pd.Series
        Rendements logarithmiques bruts de l'actif.
    alpha : pd.Series
        Alpha résiduel extrait par AlphaDeNoiser.
    save_path : str | None
        Chemin de sauvegarde PNG. Si None, utilise outputs/alpha_{ticker}.png.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, f"alpha_{ticker}.png")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Aligner les deux séries sur les mêmes dates (exclure NaN de l'alpha)
    valid = alpha.dropna()
    aligned_raw = raw_returns.loc[valid.index]

    ax.plot(aligned_raw.index, aligned_raw.values, label="Rendement brut", alpha=0.7, color="steelblue")
    ax.plot(valid.index, valid.values, label="Alpha résiduel", alpha=0.9, color="crimson", linewidth=0.9)

    std_raw = aligned_raw.std()
    std_alpha = valid.std()

    ax.set_title(
        f"{ticker} — Rendement brut vs Alpha résiduel\n"
        f"std(brut)={std_raw:.5f}  |  std(alpha)={std_alpha:.5f}  "
        f"({'OK : alpha < brut' if std_alpha < std_raw else 'ATTENTION : alpha >= brut'})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement logarithmique")
    ax.legend()
    ax.grid(True, alpha=0.3)

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

    print("\n=== Validation — alpha_denoiser.py ===\n")

    # Charger les données
    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=config.START_DATE,
        end=config.END_DATE,
    )

    market_returns = data[config.MARKET_TICKER]["log_return"]

    # Extraire les alphas
    denoiser = AlphaDeNoiser(window=config.ROLLING_WINDOW)
    alphas = denoiser.extract_alpha(data, market_returns, config.SECTOR_MAP)

    # Résumé
    print("\n--- Résumé des Alphas extraits ---")
    for ticker, alpha in alphas.items():
        valid = alpha.dropna()
        raw_std = data[ticker]["log_return"].loc[valid.index].std()
        alpha_std = valid.std()
        print(
            f"  {ticker:8s} | std brut={raw_std:.5f} | std alpha={alpha_std:.5f} "
            f"| reduction={100*(1 - alpha_std/raw_std):.1f}%"
        )

    # Graphique PEP
    print("\nGeneration du graphique pour PEP...")
    plot_alpha_vs_returns(
        ticker="PEP",
        raw_returns=data["PEP"]["log_return"],
        alpha=alphas["PEP"],
    )
    print(f"Graphique sauvegarde dans : {OUTPUTS_DIR}/alpha_PEP.png")

    # Validation : l'alpha de PEP doit avoir une variance plus faible
    pep_valid = alphas["PEP"].dropna()
    pep_raw_std = data["PEP"]["log_return"].loc[pep_valid.index].std()
    pep_alpha_std = pep_valid.std()
    assert pep_alpha_std < pep_raw_std, (
        f"ERREUR : std(alpha)={pep_alpha_std:.5f} >= std(brut)={pep_raw_std:.5f}"
    )
    print("\nValidation OK — std(alpha PEP) < std(rendement brut PEP).")
