"""
data_engine.py — Module 1 : Acquisition et Prétraitement des données
Pipeline AlphaPulse
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_data(
    tickers: list[str],
    market: str,
    sector_map: dict[str, str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """
    Télécharge et nettoie les données OHLCV journalières pour tous les tickers,
    le marché de référence et les ETF sectoriels.

    Parameters
    ----------
    tickers : list[str]
        Liste des tickers d'actifs à analyser (ex. ["PEP", "KO"]).
    market : str
        Ticker du marché de référence (ex. "VTI").
    sector_map : dict[str, str]
        Dictionnaire {ticker: etf_sectoriel} (ex. {"PEP": "XLP"}).
    start : str
        Date de début au format "YYYY-MM-DD".
    end : str
        Date de fin au format "YYYY-MM-DD".

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionnaire {ticker: DataFrame} avec colonnes
        [Open, High, Low, Close, Volume, log_return].
        Inclut également les clés du marché et des ETF sectoriels.
    """
    # Ensemble complet de tickers à télécharger
    sector_etfs = list(set(sector_map.values()))
    all_tickers = list(set(tickers + [market] + sector_etfs))
    logger.info("Téléchargement de %d tickers : %s", len(all_tickers), all_tickers)

    # -----------------------------------------------------------------------
    # 1. Téléchargement via yfinance
    # -----------------------------------------------------------------------
    try:
        raw: pd.DataFrame = yf.download(
            tickers=all_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("Erreur lors du téléchargement yfinance : %s", exc)
        raise

    # yfinance retourne un MultiIndex (colonne, ticker) quand >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        # Réorganiser en dict de DataFrames par ticker
        data: dict[str, pd.DataFrame] = {}
        for ticker in all_tickers:
            try:
                df = raw.xs(ticker, axis=1, level=1).copy()
                data[ticker] = df
            except KeyError:
                logger.warning("Ticker '%s' absent des données téléchargées.", ticker)
    else:
        # Un seul ticker téléchargé
        data = {all_tickers[0]: raw.copy()}

    # -----------------------------------------------------------------------
    # 2. Alignement sur un calendrier boursier commun
    # -----------------------------------------------------------------------
    # Trouver l'index commun (intersection de toutes les dates disponibles)
    common_index: pd.DatetimeIndex | None = None
    for ticker, df in data.items():
        df.index = pd.to_datetime(df.index)
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("Aucune date commune trouvée entre les tickers.")

    logger.info(
        "Calendrier commun : %d jours de bourse (%s → %s)",
        len(common_index),
        common_index[0].date(),
        common_index[-1].date(),
    )

    # -----------------------------------------------------------------------
    # 3. Nettoyage, forward fill et calcul des rendements logarithmiques
    # -----------------------------------------------------------------------
    cleaned: dict[str, pd.DataFrame] = {}

    for ticker, df in data.items():
        # Réindexer sur le calendrier commun
        df = df.reindex(common_index)

        # Forward fill (maximum 2 jours consécutifs)
        df = df.ffill(limit=2)

        # Vérifier les colonnes minimales requises
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        available = required_cols.intersection(df.columns)
        if available != required_cols:
            logger.warning(
                "Ticker '%s' : colonnes manquantes %s", ticker, required_cols - available
            )

        # Calcul des rendements logarithmiques sur le prix de clôture
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Vérification : pas de NaN dans log_return hors première ligne
        nan_count = df["log_return"].iloc[1:].isna().sum()
        if nan_count > 0:
            logger.warning(
                "Ticker '%s' : %d NaN dans log_return après nettoyage.", ticker, nan_count
            )
        else:
            logger.info("Ticker '%s' : log_return propre (0 NaN après ligne 1).", ticker)

        cleaned[ticker] = df

    logger.info("Chargement terminé. %d DataFrames disponibles.", len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Validation autonome (exécution directe du module)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Import de la configuration
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import config

    print("\n=== Validation — data_engine.py ===\n")

    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=config.START_DATE,
        end=config.END_DATE,
    )

    # Afficher les 5 premières lignes pour PEP
    print("--- 5 premières lignes de PEP ---")
    print(data["PEP"].head())
    print()

    # Vérifier l'absence de NaN dans log_return (hors première ligne)
    nan_in_lr = data["PEP"]["log_return"].iloc[1:].isna().sum()
    print(f"NaN dans log_return de PEP (hors ligne 0) : {nan_in_lr}")
    assert nan_in_lr == 0, "ERREUR : NaN détectés dans log_return !"
    print("\nValidation OK — aucun NaN dans log_return de PEP.")

    print(f"\nTickers chargés : {list(data.keys())}")
    print(f"Periode         : {data['PEP'].index[0].date()} -> {data['PEP'].index[-1].date()}")
    print(f"Nb jours        : {len(data['PEP'])}")
