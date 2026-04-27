"""
pairs_detector.py — Module 4 : Détection de Paires et Analyse Lead-Lag
Pipeline AlphaPulse
"""

import logging
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PAIRS_DIR = os.path.join(OUTPUTS_DIR, "pairs")
os.makedirs(PAIRS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fonctions d'analyse
# ---------------------------------------------------------------------------

def filter_by_sector(
    tickers: list[str],
    sector_map: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Retourne toutes les paires intra-sectorielles possibles.

    Parameters
    ----------
    tickers : list[str]
        Liste des tickers à considérer.
    sector_map : dict[str, str]
        Dictionnaire {ticker: etf_sectoriel}.

    Returns
    -------
    list[tuple[str, str]]
        Liste de couples (ticker_a, ticker_b) appartenant au même secteur.
    """
    sector_groups: dict[str, list[str]] = {}
    for ticker in tickers:
        if ticker not in sector_map:
            logger.warning("Ticker '%s' absent de sector_map, ignoré.", ticker)
            continue
        etf = sector_map[ticker]
        sector_groups.setdefault(etf, []).append(ticker)

    pairs: list[tuple[str, str]] = []
    for etf, group in sector_groups.items():
        if len(group) < 2:
            logger.info("Secteur '%s' : un seul ticker (%s), pas de paire.", etf, group)
            continue
        for a, b in combinations(group, 2):
            pairs.append((a, b))
            logger.info("Paire intra-sectorielle identifiée : (%s, %s) [%s]", a, b, etf)

    logger.info("%d paires intra-sectorielles au total.", len(pairs))
    return pairs


def test_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
) -> float:
    """
    Teste la cointégration entre deux séries via le test d'Engle-Granger.

    Parameters
    ----------
    series_a : pd.Series
        Première série temporelle (prix ou alpha cumulé, sans NaN).
    series_b : pd.Series
        Deuxième série temporelle (même longueur, sans NaN).

    Returns
    -------
    float
        p-value du test (< 0.05 = cointégration détectée).
    """
    try:
        aligned = pd.concat([series_a, series_b], axis=1).dropna()
        if len(aligned) < 60:
            logger.warning("Série trop courte (%d points) pour le test.", len(aligned))
            return 1.0
        _, pvalue, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(pvalue)
    except Exception as exc:
        logger.warning("Erreur test cointégration : %s", exc)
        return 1.0


def compute_cross_correlation(
    alpha_a: pd.Series,
    alpha_b: pd.Series,
    max_lag: int = 15,
) -> dict:
    """
    Calcule la corrélation croisée entre deux séries d'alpha pour des lags
    allant de -max_lag à +max_lag jours.

    Parameters
    ----------
    alpha_a : pd.Series
        Alpha résiduel du premier actif.
    alpha_b : pd.Series
        Alpha résiduel du deuxième actif.
    max_lag : int
        Lag maximal à tester (jours).

    Returns
    -------
    dict avec clés :
        - "optimal_lag"  : lag maximisant la corrélation absolue
                           (positif = A précède B, négatif = B précède A)
        - "max_corr"     : corrélation maximale (valeur absolue)
        - "leader"       : ticker dont le mouvement précède l'autre
        - "laggard"      : ticker qui suit
        - "all_lags"     : dict {lag: corrélation} pour tous les lags testés
    """
    aligned = pd.concat([alpha_a, alpha_b], axis=1).dropna()
    if len(aligned) < 2 * max_lag + 10:
        logger.warning("Série trop courte pour la corrélation croisée.")
        return {
            "optimal_lag": 0,
            "max_corr": 0.0,
            "leader": alpha_a.name or "A",
            "laggard": alpha_b.name or "B",
            "all_lags": {},
        }

    a = aligned.iloc[:, 0].values
    b = aligned.iloc[:, 1].values
    name_a = alpha_a.name or "A"
    name_b = alpha_b.name or "B"

    all_lags: dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = float(np.corrcoef(a[-lag:], b[:lag if lag != 0 else len(b)])[0, 1])
        elif lag > 0:
            corr = float(np.corrcoef(a[:len(a) - lag], b[lag:])[0, 1])
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        all_lags[lag] = 0.0 if np.isnan(corr) else corr

    optimal_lag = max(all_lags, key=lambda k: abs(all_lags[k]))
    max_corr = abs(all_lags[optimal_lag])

    if optimal_lag > 0:
        leader, laggard = name_a, name_b
    elif optimal_lag < 0:
        leader, laggard = name_b, name_a
    else:
        leader, laggard = name_a, name_b

    return {
        "optimal_lag": optimal_lag,
        "max_corr": max_corr,
        "leader": leader,
        "laggard": laggard,
        "all_lags": all_lags,
    }


# ---------------------------------------------------------------------------
# Visualisation d'une paire
# ---------------------------------------------------------------------------

def plot_pair_analysis(
    ticker_a: str,
    ticker_b: str,
    data: dict[str, pd.DataFrame],
    alphas: dict[str, pd.Series],
    pvalue: float,
    correlation: float,
    lag: int,
    leader: str,
    save_dir: str | None = None,
) -> str:
    """
    Génère et sauvegarde une visualisation en 2 panneaux pour une paire :
    - Panneau 1 : historique des prix normalisés des deux tickers superposés
    - Panneau 2 : spread entre les deux actifs (alpha cumulé A - alpha cumulé B)
                  avec relation de cointégration mise en évidence (moyenne ± 2σ)

    Parameters
    ----------
    ticker_a, ticker_b : str
        Tickers de la paire.
    data : dict[str, pd.DataFrame]
        Données brutes (prix OHLCV) issues de data_engine.
    alphas : dict[str, pd.Series]
        Alphas résiduels filtrés.
    pvalue : float
        p-value du test de cointégration.
    correlation : float
        Corrélation croisée maximale.
    lag : int
        Lag optimal (jours).
    leader : str
        Ticker leader.
    save_dir : str | None
        Répertoire de sauvegarde (défaut : outputs/pairs/).

    Returns
    -------
    str
        Chemin du fichier PNG sauvegardé.
    """
    if save_dir is None:
        save_dir = PAIRS_DIR
    os.makedirs(save_dir, exist_ok=True)

    safe_a = ticker_a.replace(".", "_")
    safe_b = ticker_b.replace(".", "_")
    save_path = os.path.join(save_dir, f"pair_{safe_a}_{safe_b}.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # --- Panneau 1 : Prix normalisés ---
    ax1 = axes[0]

    close_a = data[ticker_a]["Close"].dropna()
    close_b = data[ticker_b]["Close"].dropna()

    # Aligner sur l'index commun
    common_idx = close_a.index.intersection(close_b.index)
    close_a = close_a.loc[common_idx]
    close_b = close_b.loc[common_idx]

    # Normalisation à base 100 sur le premier point commun
    norm_a = close_a / close_a.iloc[0] * 100
    norm_b = close_b / close_b.iloc[0] * 100

    ax1.plot(norm_a.index, norm_a.values, label=ticker_a, color="steelblue", linewidth=1.2)
    ax1.plot(norm_b.index, norm_b.values, label=ticker_b, color="tomato", linewidth=1.2)
    ax1.set_title(
        f"{ticker_a} / {ticker_b} — Prix normalisés (base 100)\n"
        f"Cointégration p-value={pvalue:.4f} | Corrélation croisée={correlation:.4f} | "
        f"Lag optimal={lag}j | Leader={leader}"
    )
    ax1.set_ylabel("Indice (base 100)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panneau 2 : Spread et relation de cointégration ---
    ax2 = axes[1]

    alpha_a = alphas[ticker_a].dropna()
    alpha_b = alphas[ticker_b].dropna()
    common_alpha_idx = alpha_a.index.intersection(alpha_b.index)
    alpha_a = alpha_a.loc[common_alpha_idx]
    alpha_b = alpha_b.loc[common_alpha_idx]

    # Spread = différence des alphas cumulés (proxy du spread de prix débruité)
    cumsum_a = alpha_a.cumsum()
    cumsum_b = alpha_b.cumsum()

    # Hedge ratio OLS : cumsum_a = hedge_ratio * cumsum_b + intercept
    x = cumsum_b.values
    y = cumsum_a.values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 10:
        X = np.column_stack([x[mask], np.ones(mask.sum())])
        coeffs, _, _, _ = np.linalg.lstsq(X, y[mask], rcond=None)
        hedge_ratio = coeffs[0]
    else:
        hedge_ratio = 1.0

    spread = cumsum_a - hedge_ratio * cumsum_b
    spread_mean = spread.mean()
    spread_std = spread.std()

    ax2.plot(spread.index, spread.values, color="darkgreen", linewidth=1.0,
             label=f"Spread (hedge ratio={hedge_ratio:.3f})")
    ax2.axhline(spread_mean, color="black", linestyle="--", linewidth=1.0, label="Moyenne")
    ax2.axhline(spread_mean + 2 * spread_std, color="red", linestyle=":",
                linewidth=1.2, label="+2σ")
    ax2.axhline(spread_mean - 2 * spread_std, color="red", linestyle=":",
                linewidth=1.2, label="-2σ")
    ax2.axhline(spread_mean + spread_std, color="orange", linestyle=":",
                linewidth=0.8, alpha=0.7, label="+1σ")
    ax2.axhline(spread_mean - spread_std, color="orange", linestyle=":",
                linewidth=0.8, alpha=0.7, label="-1σ")

    # Zones de dépassement ±2σ en surbrillance
    ax2.fill_between(
        spread.index, spread_mean + 2 * spread_std, spread.values,
        where=(spread.values > spread_mean + 2 * spread_std),
        alpha=0.25, color="red", label="Anomalie >+2σ",
    )
    ax2.fill_between(
        spread.index, spread_mean - 2 * spread_std, spread.values,
        where=(spread.values < spread_mean - 2 * spread_std),
        alpha=0.25, color="blue", label="Anomalie <-2σ",
    )

    ax2.set_title(
        f"Spread cointégré {ticker_a} - {hedge_ratio:.3f}×{ticker_b} | "
        f"std={spread_std:.5f}"
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Spread (alpha cumulé)")
    ax2.legend(fontsize=7, ncol=4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Graphique de paire sauvegardé : %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def save_pairs_csv(
    pairs_df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """
    Sauvegarde le DataFrame des paires détectées en CSV.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame produit par detect_pairs().
    output_path : str | None
        Chemin complet du fichier CSV. Défaut : outputs/pairs_results.csv.

    Returns
    -------
    str
        Chemin du fichier CSV sauvegardé.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, "pairs_results.csv")

    pairs_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("CSV des paires sauvegardé : %s (%d lignes)", output_path, len(pairs_df))
    return output_path


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def detect_pairs(
    alphas: dict[str, pd.Series],
    sector_map: dict[str, str],
    corr_threshold: float = 0.6,
    coint_threshold: float = 0.05,
    max_lag: int = 15,
    data: dict[str, pd.DataFrame] | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Pipeline complet de détection de paires :
    1. Filtre par secteur
    2. Test de cointégration (Engle-Granger)
    3. Corrélation croisée avec identification du lag et du leader/laggard
    4. (Optionnel) Sauvegarde CSV + graphiques par paire

    Parameters
    ----------
    alphas : dict[str, pd.Series]
        Dictionnaire {ticker: alpha_series} issu d'AlphaDeNoiser + signal_filter.
    sector_map : dict[str, str]
        Dictionnaire {ticker: etf_sectoriel}.
    corr_threshold : float
        Seuil minimal de corrélation croisée pour retenir une paire.
    coint_threshold : float
        Seuil maximal de p-value pour la cointégration (défaut 0.05).
    max_lag : int
        Lag maximal pour la corrélation croisée.
    data : dict[str, pd.DataFrame] | None
        Données brutes OHLCV (nécessaires pour les graphiques de prix normalisés).
        Si None, les graphiques ne seront pas générés.
    save_outputs : bool
        Si True, sauvegarde le CSV et les graphiques PNG.

    Returns
    -------
    pd.DataFrame
        DataFrame des paires valides avec colonnes :
        [Ticker_A, Ticker_B, Secteur, P_value, Correlation, Lag_optimal, Leader, Laggard]
        Triées par corrélation décroissante.
    """
    tickers_available = list(alphas.keys())
    sector_pairs = filter_by_sector(tickers_available, sector_map)

    records: list[dict] = []

    for ticker_a, ticker_b in sector_pairs:
        alpha_a = alphas[ticker_a].dropna()
        alpha_b = alphas[ticker_b].dropna()

        # Test de cointégration sur les alphas cumulés (proxy prix débruités)
        price_a = alpha_a.cumsum()
        price_b = alpha_b.cumsum()
        pvalue = test_cointegration(price_a, price_b)
        logger.info("(%s, %s) — p-value cointégration : %.4f", ticker_a, ticker_b, pvalue)

        if pvalue > coint_threshold:
            logger.info(
                "Paire (%s, %s) rejetée : p-value=%.4f > %.2f",
                ticker_a, ticker_b, pvalue, coint_threshold,
            )
            continue

        # Corrélation croisée
        cross = compute_cross_correlation(alpha_a, alpha_b, max_lag=max_lag)
        logger.info(
            "(%s, %s) — corr=%.4f, lag=%d j, leader=%s",
            ticker_a, ticker_b, cross["max_corr"], cross["optimal_lag"], cross["leader"],
        )

        if cross["max_corr"] < corr_threshold:
            logger.info(
                "Paire (%s, %s) rejetée : corr=%.4f < %.2f",
                ticker_a, ticker_b, cross["max_corr"], corr_threshold,
            )
            continue

        secteur = sector_map.get(ticker_a, "N/A")
        records.append({
            "Ticker_A": ticker_a,
            "Ticker_B": ticker_b,
            "Secteur": secteur,
            "P_value": round(pvalue, 6),
            "Correlation": round(cross["max_corr"], 4),
            "Lag_optimal": cross["optimal_lag"],
            "Leader": cross["leader"],
            "Laggard": cross["laggard"],
        })

        # Graphique de la paire (si données brutes disponibles)
        if save_outputs and data is not None:
            try:
                plot_pair_analysis(
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    data=data,
                    alphas=alphas,
                    pvalue=pvalue,
                    correlation=cross["max_corr"],
                    lag=cross["optimal_lag"],
                    leader=cross["leader"],
                )
            except Exception as exc:
                logger.warning("Impossible de générer le graphique (%s, %s) : %s",
                               ticker_a, ticker_b, exc)

    if not records:
        logger.warning("Aucune paire valide détectée avec les seuils actuels.")
        empty_df = pd.DataFrame(
            columns=["Ticker_A", "Ticker_B", "Secteur", "P_value",
                     "Correlation", "Lag_optimal", "Leader", "Laggard"]
        )
        if save_outputs:
            save_pairs_csv(empty_df)
        return empty_df

    df = pd.DataFrame(records).sort_values("Correlation", ascending=False).reset_index(drop=True)
    logger.info("%d paire(s) valide(s) détectée(s).", len(df))

    if save_outputs:
        save_pairs_csv(df)

    return df


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
    from signal_filter import apply_bandpass_filter

    print("\n=== Validation — pairs_detector.py ===\n")

    # Charger données
    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=config.START_DATE,
        end=config.END_DATE,
    )

    # Extraire alphas
    denoiser = AlphaDeNoiser(window=config.ROLLING_WINDOW)
    alphas_raw = denoiser.extract_alpha(
        data, data[config.MARKET_TICKER]["log_return"], config.SECTOR_MAP
    )

    # Appliquer le filtre
    alphas_filtered: dict[str, pd.Series] = {}
    for ticker, alpha in alphas_raw.items():
        alphas_filtered[ticker] = apply_bandpass_filter(
            alpha,
            low_days=config.FILTER_LOW_DAYS,
            high_days=config.FILTER_HIGH_DAYS,
        )

    # Détecter les paires avec le seuil par défaut
    pairs_df = detect_pairs(
        alphas=alphas_filtered,
        sector_map=config.SECTOR_MAP,
        corr_threshold=config.CORRELATION_THRESHOLD,
        max_lag=config.LAG_MAX_DAYS,
        data=data,
        save_outputs=True,
    )

    if pairs_df.empty:
        print("Aucune paire avec corr >= 0.6 — relance avec seuil 0.3 pour la validation...")
        pairs_df = detect_pairs(
            alphas=alphas_filtered,
            sector_map=config.SECTOR_MAP,
            corr_threshold=0.3,
            max_lag=config.LAG_MAX_DAYS,
            data=data,
            save_outputs=True,
        )

    print("\n--- Paires détectées ---")
    print(pairs_df.to_string(index=False))

    # Vérifier le CSV
    csv_path = os.path.join(OUTPUTS_DIR, "pairs_results.csv")
    assert os.path.exists(csv_path), f"ERREUR : CSV non trouvé à {csv_path}"
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == len(pairs_df), "ERREUR : nombre de lignes CSV incohérent"
    print(f"\nCSV sauvegarde : {csv_path} ({len(loaded)} lignes)")

    # Vérifier les graphiques
    png_files = [f for f in os.listdir(PAIRS_DIR) if f.endswith(".png")]
    print(f"Graphiques PNG generes dans {PAIRS_DIR} : {png_files}")
    assert len(png_files) >= len(pairs_df), "ERREUR : nombre de PNG insuffisant"

    # Vérifier la présence des paires attendues
    expected = [("PEP", "KO"), ("XOM", "CVX")]
    for a, b in expected:
        found = (
            ((pairs_df["Ticker_A"] == a) & (pairs_df["Ticker_B"] == b)) |
            ((pairs_df["Ticker_A"] == b) & (pairs_df["Ticker_B"] == a))
        ).any()
        print(f"Paire ({a}/{b}) : {'OK' if found else 'ABSENT'}")

    print("\nValidation OK — CSV et graphiques generes avec succes.")
