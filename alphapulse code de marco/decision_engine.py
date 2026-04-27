"""
decision_engine.py — Module 6 : Moteur de Décision Z-Score
Pipeline AlphaPulse

Calcule le Z-Score roulant du spread entre deux alphas filtrés.
Émet un signal structuré quand le seuil ±2.0 est franchi.

Note : la validation IA (ai_validator.py) est temporairement désactivée
et déplacée dans _en_attente/. Le module fonctionne en autonome.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calcul du Z-Score roulant
# ---------------------------------------------------------------------------

def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Calcule le Z-Score roulant d'un spread.

    Z_t = (spread_t - μ_{t,window}) / σ_{t,window}

    Les `window - 1` premiers points sont NaN (fenêtre incomplète).

    Parameters
    ----------
    spread : pd.Series
        Série du spread entre les deux actifs.
    window : int
        Taille de la fenêtre roulante (jours de bourse, défaut : 60).

    Returns
    -------
    pd.Series
        Z-Score roulant (NaN pour les `window - 1` premiers points).
    """
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std  = spread.rolling(window=window, min_periods=window).std()

    # Éviter la division par zéro si l'écart-type est nul
    zscore = (spread - rolling_mean) / rolling_std.replace(0.0, np.nan)
    zscore.name = "zscore"

    valid = zscore.dropna()
    if len(valid) > 0:
        logger.info(
            "Z-Score calculé (fenêtre=%d) : min=%.2f, max=%.2f, NaN=%d",
            window, valid.min(), valid.max(), zscore.isna().sum(),
        )
    return zscore


# ---------------------------------------------------------------------------
# Calcul du spread avec hedge ratio roulant
# ---------------------------------------------------------------------------

def _compute_spread(
    alpha_a: pd.Series,
    alpha_b: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Calcule le spread entre deux alphas filtrés avec hedge ratio roulant OLS.

    spread_t = alpha_a_t - hedge_ratio_t × alpha_b_t

    Le hedge ratio est estimé par régression roulante (cov/var) pour s'adapter
    aux changements de corrélation dans le temps.

    Parameters
    ----------
    alpha_a : pd.Series
        Alpha filtré du premier actif.
    alpha_b : pd.Series
        Alpha filtré du deuxième actif.
    window : int
        Fenêtre roulante pour le hedge ratio (jours).

    Returns
    -------
    pd.Series
        Série du spread alignée sur l'index commun.
    """
    aligned = pd.concat([alpha_a, alpha_b], axis=1).dropna()
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]

    # Hedge ratio roulant = cov(a,b) / var(b)
    rolling_cov = a.rolling(window=window, min_periods=window).cov(b)
    rolling_var = b.rolling(window=window, min_periods=window).var()
    hedge_ratio  = (rolling_cov / rolling_var.replace(0.0, np.nan)).fillna(1.0)

    spread = a - hedge_ratio * b
    spread.name = "spread"
    return spread


# ---------------------------------------------------------------------------
# Moteur de décision principal
# ---------------------------------------------------------------------------

def run_decision(
    pairs_df: pd.DataFrame,
    alphas_filtered: dict[str, pd.Series],
    zscore_threshold: float = 2.0,
    rolling_window: int = 60,
) -> list[dict]:
    """
    Pour chaque paire valide : calcule le spread, son Z-Score roulant, et
    émet un signal structuré si le seuil est franchi sur la dernière observation.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame des paires détectées (issu de pairs_detector.detect_pairs).
        Colonnes attendues : Ticker_A, Ticker_B, Leader, Laggard, Lag_optimal.
    alphas_filtered : dict[str, pd.Series]
        Dictionnaire {ticker: alpha_filtré} issu de signal_filter.
    zscore_threshold : float
        Seuil absolu de Z-Score pour déclencher un signal (défaut : 2.0).
    rolling_window : int
        Fenêtre roulante pour le calcul du spread et du Z-Score (défaut : 60).

    Returns
    -------
    list[dict]
        Liste de signaux actifs. Chaque signal contient :
        {ticker_leader, ticker_laggard, ticker_a, ticker_b,
         signal, zscore, lag, spread_direction,
         confiance_ia, justification, timestamp}

    Note
    ----
    confiance_ia = 0.0 et justification = "Validation IA désactivée."
    tant que ai_validator.py est en _en_attente/.
    """
    if pairs_df.empty:
        logger.warning("pairs_df vide — aucun signal à générer.")
        return []

    signals: list[dict] = []
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    for _, row in pairs_df.iterrows():
        ticker_a: str = str(row["Ticker_A"])
        ticker_b: str = str(row["Ticker_B"])
        leader:   str = str(row["Leader"])
        laggard:  str = str(row["Laggard"])
        lag:      int = int(row["Lag_optimal"])

        # Vérifier la disponibilité des alphas
        if ticker_a not in alphas_filtered or ticker_b not in alphas_filtered:
            logger.warning(
                "Alphas manquants pour (%s, %s) — paire ignorée.", ticker_a, ticker_b
            )
            continue

        alpha_a = alphas_filtered[ticker_a].dropna()
        alpha_b = alphas_filtered[ticker_b].dropna()

        if len(alpha_a) < rolling_window + 10 or len(alpha_b) < rolling_window + 10:
            logger.warning(
                "Série trop courte pour la paire (%s, %s).", ticker_a, ticker_b
            )
            continue

        # 1. Spread avec hedge ratio roulant
        spread = _compute_spread(alpha_a, alpha_b, window=rolling_window)

        # 2. Z-Score roulant du spread
        zscore_series = compute_zscore(spread, window=rolling_window)
        zscore_clean  = zscore_series.dropna()

        if zscore_clean.empty:
            logger.warning("Z-Score entièrement NaN pour (%s, %s).", ticker_a, ticker_b)
            continue

        current_zscore = float(zscore_clean.iloc[-1])

        logger.info(
            "Paire (%s, %s) — Z-Score actuel : %.3f (seuil : ±%.1f)",
            ticker_a, ticker_b, current_zscore, zscore_threshold,
        )

        # 3. Seuil franchi ?
        if abs(current_zscore) < zscore_threshold:
            logger.info(
                "Paire (%s, %s) — pas de signal (|Z|=%.3f < %.1f).",
                ticker_a, ticker_b, abs(current_zscore), zscore_threshold,
            )
            continue

        # 4. Direction du signal
        # Z > 0 : alpha_a sur-performe → spread trop élevé → vendre A, acheter B
        # Z < 0 : alpha_b sur-performe → spread trop bas  → acheter A, vendre B
        if current_zscore > 0:
            spread_direction = f"LONG {ticker_b} / SHORT {ticker_a}"
        else:
            spread_direction = f"LONG {ticker_a} / SHORT {ticker_b}"

        logger.info(
            "Signal émis : (%s, %s) Z=%.3f → %s",
            ticker_a, ticker_b, current_zscore, spread_direction,
        )

        signals.append({
            "ticker_leader":    leader,
            "ticker_laggard":   laggard,
            "ticker_a":         ticker_a,
            "ticker_b":         ticker_b,
            "signal":           "SIGNAL_VALIDE",
            "zscore":           round(current_zscore, 4),
            "lag":              lag,
            "spread_direction": spread_direction,
            "confiance_ia":     0.0,
            "justification":    "Validation IA désactivée (ai_validator en _en_attente/).",
            "timestamp":        timestamp,
        })

    logger.info(
        "%d signal(s) actif(s) sur %d paire(s) évaluée(s).",
        len(signals), len(pairs_df),
    )
    return signals


# ---------------------------------------------------------------------------
# Validation autonome
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    sys.path.insert(0, os.path.dirname(__file__))
    import config
    from data_engine import load_data
    from alpha_denoiser import AlphaDeNoiser
    from signal_filter import apply_bandpass_filter
    from pairs_detector import detect_pairs

    print("\n=== Validation — decision_engine.py ===\n")

    # Charger les données
    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=config.START_DATE,
        end=config.END_DATE,
    )

    # Extraire et filtrer les alphas
    denoiser = AlphaDeNoiser(window=config.ROLLING_WINDOW)
    alphas_raw = denoiser.extract_alpha(
        data, data[config.MARKET_TICKER]["log_return"], config.SECTOR_MAP
    )
    alphas_filtered: dict[str, pd.Series] = {
        ticker: apply_bandpass_filter(
            alpha,
            low_days=config.FILTER_LOW_DAYS,
            high_days=config.FILTER_HIGH_DAYS,
        )
        for ticker, alpha in alphas_raw.items()
    }

    # Détecter les paires
    pairs_df = detect_pairs(
        alphas=alphas_filtered,
        sector_map=config.SECTOR_MAP,
        corr_threshold=config.CORRELATION_THRESHOLD,
        max_lag=config.LAG_MAX_DAYS,
        data=data,
        save_outputs=False,
    )

    if pairs_df.empty:
        print("Aucune paire avec seuil 0.6 — relance à 0.3...")
        pairs_df = detect_pairs(
            alphas=alphas_filtered,
            sector_map=config.SECTOR_MAP,
            corr_threshold=0.3,
            max_lag=config.LAG_MAX_DAYS,
            data=data,
            save_outputs=False,
        )

    print(f"Paires évaluées :\n{pairs_df.to_string(index=False)}\n")

    # Générer les signaux avec le seuil de config
    signals = run_decision(
        pairs_df=pairs_df,
        alphas_filtered=alphas_filtered,
        zscore_threshold=config.ZSCORE_THRESHOLD,
        rolling_window=config.ROLLING_WINDOW,
    )

    print(f"\n--- Signaux actifs (seuil Z=±{config.ZSCORE_THRESHOLD}) ---")
    if signals:
        for sig in signals:
            print(json.dumps(sig, indent=2, ensure_ascii=False))
    else:
        print("Aucun signal actif avec le seuil par défaut.")

    # Relance avec seuil bas pour forcer l'affichage d'un exemple de signal
    if not signals:
        print("\nRelance avec seuil Z=±0.5 pour afficher un exemple de signal structuré...")
        signals_demo = run_decision(
            pairs_df=pairs_df,
            alphas_filtered=alphas_filtered,
            zscore_threshold=0.5,
            rolling_window=config.ROLLING_WINDOW,
        )
        print(f"\n--- Signaux demo (seuil Z=±0.5) ---")
        for sig in signals_demo:
            print(json.dumps(sig, indent=2, ensure_ascii=False))

    # Validation structurelle
    all_signals = signals if signals else (signals_demo if not signals else [])
    for sig in all_signals:
        assert "ticker_leader"    in sig, "ERREUR : clé 'ticker_leader' manquante"
        assert "ticker_laggard"   in sig, "ERREUR : clé 'ticker_laggard' manquante"
        assert "signal"           in sig, "ERREUR : clé 'signal' manquante"
        assert "zscore"           in sig, "ERREUR : clé 'zscore' manquante"
        assert "spread_direction" in sig, "ERREUR : clé 'spread_direction' manquante"
        assert "timestamp"        in sig, "ERREUR : clé 'timestamp' manquante"

    print("\nValidation OK — structure des signaux conforme.")
