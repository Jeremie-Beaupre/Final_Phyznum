"""
backtester.py — Module 7 : Backtesting et Optimisation des Paramètres
Pipeline AlphaPulse

Rejoue la stratégie d'arbitrage sur données historiques journalières.
Calcule les métriques de performance et optimise les paramètres via grid search.

Note : n'appelle PAS ai_validator (trop coûteux sur l'historique).
"""

import logging
import os
from itertools import product

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decision_engine import compute_zscore, _compute_spread
from signal_filter import apply_bandpass_filter

logger = logging.getLogger(__name__)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Génération des positions (signal d'entrée/sortie)
# ---------------------------------------------------------------------------

def _generate_positions(zscore: pd.Series, threshold: float) -> pd.Series:
    """
    Génère les positions (+1, -1, 0) à partir du Z-Score.

    Règles :
    - Z > +threshold → SHORT spread (-1) : spread trop élevé, reviendra vers 0
    - Z < -threshold → LONG spread (+1)  : spread trop bas, reviendra vers 0
    - Z repasse par 0 → fermer la position (sortie mean-reversion)

    Parameters
    ----------
    zscore : pd.Series
        Z-Score roulant du spread.
    threshold : float
        Seuil d'entrée (ex. 2.0).

    Returns
    -------
    pd.Series
        Positions journalières (+1, -1 ou 0).
    """
    positions: list[int] = []
    current_pos = 0

    for z in zscore:
        if pd.isna(z):
            positions.append(0)
            current_pos = 0
            continue

        if current_pos == 0:
            if z > threshold:
                current_pos = -1   # Spread trop élevé → SHORT
            elif z < -threshold:
                current_pos = 1    # Spread trop bas  → LONG
        else:
            # Sortie quand le Z-Score repasse par 0 (mean reversion complète)
            if current_pos == 1 and z >= 0.0:
                current_pos = 0
            elif current_pos == -1 and z <= 0.0:
                current_pos = 0

        positions.append(current_pos)

    return pd.Series(positions, index=zscore.index, name="position", dtype=int)


# ---------------------------------------------------------------------------
# Extraction des trades individuels
# ---------------------------------------------------------------------------

def _extract_trades(
    position: pd.Series,
    daily_pnl: pd.Series,
    zscore: pd.Series,
) -> pd.DataFrame:
    """
    Extrait le tableau des trades individuels à partir des positions.

    Returns
    -------
    pd.DataFrame
        Colonnes : [entry_date, exit_date, direction, entry_zscore,
                    exit_zscore, pnl, duration_days]
    """
    trades: list[dict] = []
    in_trade = False
    entry_date = None
    entry_z = None
    trade_dir = 0
    trade_pnl = 0.0

    pos_values = position.values
    dates = position.index

    for i in range(1, len(position)):
        prev = int(pos_values[i - 1])
        curr = int(pos_values[i])

        # Entrée en trade
        if prev == 0 and curr != 0:
            in_trade = True
            entry_date = dates[i]
            entry_z = float(zscore.iloc[i]) if not pd.isna(zscore.iloc[i]) else np.nan
            trade_dir = curr
            trade_pnl = 0.0

        # En trade : accumuler P&L
        if in_trade and not pd.isna(daily_pnl.iloc[i]):
            trade_pnl += float(daily_pnl.iloc[i])

        # Sortie du trade
        if in_trade and curr == 0 and prev != 0:
            exit_date = dates[i]
            exit_z = float(zscore.iloc[i]) if not pd.isna(zscore.iloc[i]) else np.nan
            duration = (exit_date - entry_date).days if entry_date else 0
            trades.append({
                "entry_date":   entry_date,
                "exit_date":    exit_date,
                "direction":    "LONG" if trade_dir == 1 else "SHORT",
                "entry_zscore": round(entry_z, 3) if entry_z and not np.isnan(entry_z) else np.nan,
                "exit_zscore":  round(exit_z, 3) if exit_z and not np.isnan(exit_z) else np.nan,
                "pnl":          round(trade_pnl, 6),
                "duration_days": duration,
            })
            in_trade = False
            trade_pnl = 0.0

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Calcul des métriques de performance
# ---------------------------------------------------------------------------

def _compute_metrics(
    daily_pnl: pd.Series,
    trades_df: pd.DataFrame,
) -> dict:
    """
    Calcule les métriques de performance de la stratégie.

    Parameters
    ----------
    daily_pnl : pd.Series
        P&L journalier de la stratégie.
    trades_df : pd.DataFrame
        Tableau des trades individuels.

    Returns
    -------
    dict
        Clés : sharpe_ratio, sortino_ratio, max_drawdown,
                winrate, annual_return, n_trades, total_pnl
    """
    pnl = daily_pnl.dropna()
    pnl_in_trade = pnl[pnl != 0.0]  # Jours avec une position ouverte

    # Rendement annualisé (en supposant 252 jours de bourse/an)
    n_days = len(pnl_in_trade)
    total_pnl = float(pnl_in_trade.sum())

    if n_days > 0:
        mean_daily = float(pnl_in_trade.mean())
        std_daily  = float(pnl_in_trade.std())
        annual_return = mean_daily * 252

        # Sharpe Ratio annualisé (sans taux sans risque pour simplification académique)
        sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

        # Sortino : std des rendements négatifs seulement
        neg_returns = pnl_in_trade[pnl_in_trade < 0]
        sortino_std = float(neg_returns.std()) if len(neg_returns) > 1 else std_daily
        sortino = (mean_daily / sortino_std * np.sqrt(252)) if sortino_std > 0 else 0.0
    else:
        annual_return = 0.0
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown sur la courbe de richesse
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_drawdown = float(drawdown.min())

    # Win rate
    if not trades_df.empty:
        n_trades = len(trades_df)
        winning = int((trades_df["pnl"] > 0).sum())
        winrate = winning / n_trades if n_trades > 0 else 0.0
    else:
        n_trades = 0
        winrate = 0.0

    return {
        "sharpe_ratio":  round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown":  round(max_drawdown, 6),
        "winrate":       round(winrate, 3),
        "annual_return": round(annual_return, 6),
        "n_trades":      n_trades,
        "total_pnl":     round(total_pnl, 6),
    }


# ---------------------------------------------------------------------------
# Backtest principal d'une paire
# ---------------------------------------------------------------------------

def run_backtest(
    pair: tuple[str, str],
    alphas: dict[str, pd.Series],
    zscore_threshold: float = 2.0,
    rolling_window: int = 60,
    save_outputs: bool = True,
) -> dict:
    """
    Simule la stratégie de paires sur l'historique complet.

    Parameters
    ----------
    pair : tuple[str, str]
        Couple (ticker_a, ticker_b).
    alphas : dict[str, pd.Series]
        Dictionnaire {ticker: alpha_filtré}.
    zscore_threshold : float
        Seuil d'entrée en position.
    rolling_window : int
        Fenêtre roulante pour le spread et le Z-Score.
    save_outputs : bool
        Si True, sauvegarde les graphiques PNG.

    Returns
    -------
    dict
        Clés : metrics (dict), trades (DataFrame), wealth_curve (Series),
               spread (Series), zscore (Series), position (Series)
    """
    ticker_a, ticker_b = pair

    if ticker_a not in alphas or ticker_b not in alphas:
        raise ValueError(f"Alpha manquant pour {ticker_a} ou {ticker_b}.")

    alpha_a = alphas[ticker_a].dropna()
    alpha_b = alphas[ticker_b].dropna()

    # 1. Spread et Z-Score
    spread  = _compute_spread(alpha_a, alpha_b, window=rolling_window)
    zscore  = compute_zscore(spread, window=rolling_window)

    # 2. Positions
    position = _generate_positions(zscore, threshold=zscore_threshold)

    # 3. P&L journalier
    # Le P&L d'un jour = position d'hier × variation du spread aujourd'hui
    # (évite le look-ahead bias)
    spread_change = spread.diff()
    daily_pnl = position.shift(1) * spread_change
    daily_pnl.name = "daily_pnl"

    # 4. Courbe de richesse (P&L cumulé)
    wealth_curve = daily_pnl.fillna(0).cumsum()
    wealth_curve.name = "wealth"

    # 5. Trades individuels
    trades_df = _extract_trades(position, daily_pnl, zscore)

    # 6. Métriques
    metrics = _compute_metrics(daily_pnl, trades_df)

    logger.info(
        "Backtest (%s, %s) | Sharpe=%.2f | Sortino=%.2f | MaxDD=%.4f | "
        "WinRate=%.1f%% | Trades=%d | Return=%.4f",
        ticker_a, ticker_b,
        metrics["sharpe_ratio"], metrics["sortino_ratio"],
        metrics["max_drawdown"], metrics["winrate"] * 100,
        metrics["n_trades"], metrics["annual_return"],
    )

    # 7. Graphiques
    if save_outputs:
        _plot_backtest(
            ticker_a=ticker_a, ticker_b=ticker_b,
            spread=spread, zscore=zscore,
            position=position, wealth_curve=wealth_curve,
            trades_df=trades_df, metrics=metrics,
            zscore_threshold=zscore_threshold,
        )

    return {
        "metrics":       metrics,
        "trades":        trades_df,
        "wealth_curve":  wealth_curve,
        "spread":        spread,
        "zscore":        zscore,
        "position":      position,
    }


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def grid_search(
    pairs_df: pd.DataFrame,
    alphas_raw: dict[str, pd.Series],
    zscore_range: list[float] | None = None,
    filter_range: list[tuple[float, float]] | None = None,
    rolling_window: int = 60,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Grid search sur le seuil Z-Score et les bornes du filtre fréquentiel.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Paires détectées (issu de pairs_detector.detect_pairs).
    alphas_raw : dict[str, pd.Series]
        Alphas NON filtrés (issus de AlphaDeNoiser uniquement).
        Le filtre sera réappliqué à chaque point de la grille.
    zscore_range : list[float]
        Liste de seuils Z-Score à tester. Défaut : [1.5, 2.0, 2.5, 3.0].
    filter_range : list[tuple[float, float]]
        Liste de (low_days, high_days) à tester.
        Défaut : [(3, 126), (3, 252), (5, 252)].
    rolling_window : int
        Fenêtre roulante pour le Z-Score.
    save_outputs : bool
        Si True, sauvegarde la heatmap PNG.

    Returns
    -------
    pd.DataFrame
        Résultats du grid search avec colonnes :
        [ticker_a, ticker_b, zscore_threshold, low_days, high_days,
         sharpe_ratio, sortino_ratio, winrate, n_trades, annual_return]
        Triés par Sharpe décroissant.
    """
    if zscore_range is None:
        zscore_range = [1.5, 2.0, 2.5, 3.0]
    if filter_range is None:
        filter_range = [(3.0, 126.0), (3.0, 252.0), (5.0, 252.0)]

    if pairs_df.empty:
        logger.warning("pairs_df vide — grid search annulé.")
        return pd.DataFrame()

    records: list[dict] = []
    total = len(pairs_df) * len(zscore_range) * len(filter_range)
    done = 0

    for _, row in pairs_df.iterrows():
        ticker_a = str(row["Ticker_A"])
        ticker_b = str(row["Ticker_B"])

        if ticker_a not in alphas_raw or ticker_b not in alphas_raw:
            logger.warning("Alpha manquant pour (%s, %s).", ticker_a, ticker_b)
            continue

        for (low_days, high_days), zscore_thresh in product(filter_range, zscore_range):
            # Réappliquer le filtre avec les paramètres de la grille
            alpha_a_f = apply_bandpass_filter(
                alphas_raw[ticker_a], low_days=low_days, high_days=high_days
            )
            alpha_b_f = apply_bandpass_filter(
                alphas_raw[ticker_b], low_days=low_days, high_days=high_days
            )

            alphas_grid = {ticker_a: alpha_a_f, ticker_b: alpha_b_f}

            try:
                result = run_backtest(
                    pair=(ticker_a, ticker_b),
                    alphas=alphas_grid,
                    zscore_threshold=zscore_thresh,
                    rolling_window=rolling_window,
                    save_outputs=False,
                )
                m = result["metrics"]
                records.append({
                    "ticker_a":        ticker_a,
                    "ticker_b":        ticker_b,
                    "zscore_threshold": zscore_thresh,
                    "low_days":        low_days,
                    "high_days":       high_days,
                    "sharpe_ratio":    m["sharpe_ratio"],
                    "sortino_ratio":   m["sortino_ratio"],
                    "winrate":         m["winrate"],
                    "n_trades":        m["n_trades"],
                    "annual_return":   m["annual_return"],
                })
            except Exception as exc:
                logger.warning(
                    "Grid search (%s, %s) z=%.1f f=[%.0f,%.0f] : %s",
                    ticker_a, ticker_b, zscore_thresh, low_days, high_days, exc,
                )

            done += 1
            logger.info("Grid search : %d/%d terminés.", done, total)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    if save_outputs:
        _plot_grid_search_heatmap(df)

    return df


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _plot_backtest(
    ticker_a: str,
    ticker_b: str,
    spread: pd.Series,
    zscore: pd.Series,
    position: pd.Series,
    wealth_curve: pd.Series,
    trades_df: pd.DataFrame,
    metrics: dict,
    zscore_threshold: float,
    save_dir: str | None = None,
) -> str:
    """
    Génère 3 panneaux pour un backtest :
    1. Courbe de richesse (P&L cumulé)
    2. Z-Score avec zones d'entrée/sortie et trades
    3. Distribution des P&L par trade (histogram)
    """
    if save_dir is None:
        save_dir = OUTPUTS_DIR

    safe_a = ticker_a.replace(".", "_")
    safe_b = ticker_b.replace(".", "_")
    save_path = os.path.join(save_dir, f"backtest_{safe_a}_{safe_b}.png")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # --- Panneau 1 : Courbe de richesse ---
    ax1 = axes[0]
    ax1.plot(wealth_curve.index, wealth_curve.values, color="darkgreen", linewidth=1.2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.fill_between(
        wealth_curve.index, 0, wealth_curve.values,
        where=(wealth_curve.values >= 0), alpha=0.15, color="green",
    )
    ax1.fill_between(
        wealth_curve.index, 0, wealth_curve.values,
        where=(wealth_curve.values < 0), alpha=0.15, color="red",
    )
    ax1.set_title(
        f"{ticker_a}/{ticker_b} — Courbe de richesse (P&L cumulé)\n"
        f"Sharpe={metrics['sharpe_ratio']:.2f} | Sortino={metrics['sortino_ratio']:.2f} | "
        f"MaxDD={metrics['max_drawdown']:.4f} | WinRate={metrics['winrate']*100:.1f}% | "
        f"Trades={metrics['n_trades']}"
    )
    ax1.set_ylabel("P&L cumulé (unités de spread)")
    ax1.grid(True, alpha=0.3)

    # --- Panneau 2 : Z-Score + zones + trades ---
    ax2 = axes[1]
    z_clean = zscore.dropna()
    ax2.plot(z_clean.index, z_clean.values, color="navy", linewidth=0.8, label="Z-Score")
    ax2.axhline(0,                color="black", linestyle="--", linewidth=0.8)
    ax2.axhline(+zscore_threshold, color="red",   linestyle="--", linewidth=1.0,
                label=f"+{zscore_threshold}σ (SHORT)")
    ax2.axhline(-zscore_threshold, color="blue",  linestyle="--", linewidth=1.0,
                label=f"-{zscore_threshold}σ (LONG)")
    ax2.axhline(+2 * zscore_threshold, color="red",  linestyle=":", linewidth=0.7, alpha=0.5)
    ax2.axhline(-2 * zscore_threshold, color="blue", linestyle=":", linewidth=0.7, alpha=0.5)

    # Zones de position
    pos_clean = position.reindex(z_clean.index).fillna(0)
    ax2.fill_between(
        z_clean.index, -6, 6,
        where=(pos_clean.values == 1),  alpha=0.08, color="blue",  label="LONG spread"
    )
    ax2.fill_between(
        z_clean.index, -6, 6,
        where=(pos_clean.values == -1), alpha=0.08, color="red",   label="SHORT spread"
    )

    # Marqueurs d'entrée sur les trades
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            color = "blue" if trade["direction"] == "LONG" else "red"
            ax2.axvline(trade["entry_date"], color=color, alpha=0.3, linewidth=0.6)

    ax2.set_ylim(-max(4, zscore_threshold * 2.5), max(4, zscore_threshold * 2.5))
    ax2.set_title(f"Z-Score du spread | seuil=±{zscore_threshold}")
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=7, ncol=4)
    ax2.grid(True, alpha=0.3)

    # --- Panneau 3 : Distribution des P&L par trade ---
    ax3 = axes[2]
    if not trades_df.empty and len(trades_df) > 1:
        pnl_vals = trades_df["pnl"].values
        colors = ["green" if p > 0 else "red" for p in pnl_vals]
        ax3.bar(range(len(pnl_vals)), pnl_vals, color=colors, alpha=0.7)
        ax3.axhline(0, color="black", linewidth=0.8)
        ax3.set_title(
            f"P&L par trade | "
            f"Gagnants={int((trades_df['pnl']>0).sum())} / "
            f"Perdants={int((trades_df['pnl']<=0).sum())} | "
            f"Durée moy.={trades_df['duration_days'].mean():.0f}j"
        )
        ax3.set_xlabel("Numéro du trade")
        ax3.set_ylabel("P&L (unités de spread)")
    else:
        ax3.text(0.5, 0.5, "Aucun trade généré", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Distribution des trades")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Graphique backtest sauvegardé : %s", save_path)
    return save_path


def _plot_grid_search_heatmap(
    grid_df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Génère une heatmap du Sharpe Ratio pour le grid search.
    Axes : zscore_threshold (x) × filtre low_days-high_days (y).
    Une heatmap par paire.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "grid_search_heatmap.png")

    pairs = grid_df[["ticker_a", "ticker_b"]].drop_duplicates().values.tolist()
    n_pairs = len(pairs)

    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5), squeeze=False)

    for idx, (ta, tb) in enumerate(pairs):
        ax = axes[0][idx]
        sub = grid_df[(grid_df["ticker_a"] == ta) & (grid_df["ticker_b"] == tb)].copy()
        sub["filter_label"] = sub.apply(
            lambda r: f"{int(r['low_days'])}j–{int(r['high_days'])}j", axis=1
        )

        pivot = sub.pivot_table(
            index="filter_label",
            columns="zscore_threshold",
            values="sharpe_ratio",
            aggfunc="mean",
        )

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=min(-0.5, pivot.values.min()),
                       vmax=max(0.5, pivot.values.max()))

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"Z={v}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f"{ta}/{tb} — Sharpe Ratio\n(grid search)")
        ax.set_xlabel("Seuil Z-Score")
        ax.set_ylabel("Filtre [low, high]")

        # Annoter les cellules
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

        plt.colorbar(im, ax=ax, label="Sharpe Ratio")

    plt.suptitle("Grid Search — Optimisation des paramètres AlphaPulse", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Heatmap grid search sauvegardée : %s", save_path)


# ---------------------------------------------------------------------------
# Simulateur de budget en dollars réels
# ---------------------------------------------------------------------------

def simulate_budget(
    ticker_a: str,
    ticker_b: str,
    data: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    initial_budget: float = 10_000.0,
    save_dir: str | None = None,
) -> dict:
    """
    Simule le P&L réel en dollars d'une stratégie de paires sur un budget initial.

    Pour chaque trade :
    - On investit `budget_courant / 2` sur chaque jambe (long + short)
    - Le P&L en $ est calculé à partir des vrais prix de clôture d'entrée/sortie
    - Le budget est mis à jour après chaque trade (capitalisation)

    Convention :
    - direction="LONG"  -> Achat A / Vente B (on profite si A monte ou B baisse)
    - direction="SHORT" -> Vente A / Achat B (on profite si A baisse ou B monte)

    Parameters
    ----------
    ticker_a, ticker_b : str
        Tickers de la paire.
    data : dict[str, pd.DataFrame]
        Données de prix réels (clé "Close") issues de data_engine.
    trades_df : pd.DataFrame
        Tableau des trades issu de run_backtest() (colonnes : entry_date,
        exit_date, direction, pnl, ...).
    initial_budget : float
        Capital de départ en dollars (défaut : 10 000$).
    save_dir : str | None
        Répertoire de sauvegarde PNG (défaut : outputs/).

    Returns
    -------
    dict avec clés :
        - "summary"       : pd.DataFrame détaillé de chaque trade en $
        - "wealth_curve"  : pd.Series de la richesse cumulée en $
        - "total_profit"  : float — profit net total en $
        - "final_wealth"  : float — capital final en $
        - "roi_pct"       : float — rendement total en %
        - "best_trade"    : dict — meilleur trade en $
        - "worst_trade"   : dict — pire trade en $
        - "plot_path"     : str — chemin du PNG généré
    """
    if save_dir is None:
        save_dir = OUTPUTS_DIR

    close_a = data[ticker_a]["Close"]
    close_b = data[ticker_b]["Close"]

    if trades_df.empty:
        logger.warning("Aucun trade disponible pour la simulation de budget.")
        return {
            "summary":      pd.DataFrame(),
            "wealth_curve": pd.Series(dtype=float),
            "total_profit": 0.0,
            "final_wealth": initial_budget,
            "roi_pct":      0.0,
            "best_trade":   {},
            "worst_trade":  {},
            "plot_path":    "",
        }

    records: list[dict] = []
    wealth = initial_budget

    for i, (_, trade) in enumerate(trades_df.iterrows(), start=1):
        entry_date = trade["entry_date"]
        exit_date  = trade["exit_date"]
        direction  = trade["direction"]

        # Prix d'entrée et de sortie (clôture)
        if entry_date not in close_a.index or exit_date not in close_a.index:
            continue
        if entry_date not in close_b.index or exit_date not in close_b.index:
            continue

        pa_entry = float(close_a.loc[entry_date])
        pa_exit  = float(close_a.loc[exit_date])
        pb_entry = float(close_b.loc[entry_date])
        pb_exit  = float(close_b.loc[exit_date])

        # Investissement par jambe = moitié du capital courant
        per_leg = wealth / 2.0

        # Nombre d'actions achetées/vendues (fractions autorisées)
        shares_a = per_leg / pa_entry
        shares_b = per_leg / pb_entry

        # P&L en $ selon la direction
        if direction == "LONG":
            # Achat A + Vente à découvert B
            pnl_a = shares_a * (pa_exit - pa_entry)   # positif si A monte
            pnl_b = shares_b * (pb_entry - pb_exit)   # positif si B baisse
        else:  # "SHORT"
            # Vente à découvert A + Achat B
            pnl_a = shares_a * (pa_entry - pa_exit)   # positif si A baisse
            pnl_b = shares_b * (pb_exit - pb_entry)   # positif si B monte

        total_pnl_usd = pnl_a + pnl_b
        wealth += total_pnl_usd
        roi_trade = total_pnl_usd / initial_budget * 100

        records.append({
            "trade_n":      i,
            "entry_date":   entry_date.date() if hasattr(entry_date, "date") else entry_date,
            "exit_date":    exit_date.date()  if hasattr(exit_date,  "date") else exit_date,
            "direction":    direction,
            f"prix_entree_{ticker_a}": round(pa_entry, 2),
            f"prix_sortie_{ticker_a}": round(pa_exit,  2),
            f"prix_entree_{ticker_b}": round(pb_entry, 2),
            f"prix_sortie_{ticker_b}": round(pb_exit,  2),
            "pnl_a_usd":    round(pnl_a, 2),
            "pnl_b_usd":    round(pnl_b, 2),
            "pnl_total_usd": round(total_pnl_usd, 2),
            "roi_trade_pct": round(roi_trade, 3),
            "capital_apres": round(wealth, 2),
        })

    summary_df = pd.DataFrame(records)

    if summary_df.empty:
        return {
            "summary": summary_df, "wealth_curve": pd.Series(dtype=float),
            "total_profit": 0.0, "final_wealth": initial_budget,
            "roi_pct": 0.0, "best_trade": {}, "worst_trade": {}, "plot_path": "",
        }

    # Courbe de richesse indexée sur les dates de sortie
    wealth_curve = pd.Series(
        [initial_budget] + list(summary_df["capital_apres"].values),
        name="wealth_usd",
    )

    total_profit = float(summary_df["pnl_total_usd"].sum())
    final_wealth = initial_budget + total_profit
    roi_pct = total_profit / initial_budget * 100

    best_idx  = summary_df["pnl_total_usd"].idxmax()
    worst_idx = summary_df["pnl_total_usd"].idxmin()
    best_trade  = summary_df.loc[best_idx].to_dict()
    worst_trade = summary_df.loc[worst_idx].to_dict()

    logger.info(
        "Simulation budget (%s/%s) | Budget=%.0f$ | Final=%.0f$ | "
        "Profit=%.0f$ | ROI=%.1f%% | Trades=%d",
        ticker_a, ticker_b, initial_budget, final_wealth,
        total_profit, roi_pct, len(summary_df),
    )

    # --- Graphique ---
    safe_a = ticker_a.replace(".", "_")
    safe_b = ticker_b.replace(".", "_")
    save_path = os.path.join(save_dir, f"simulation_{safe_a}_{safe_b}.png")

    _plot_simulation(
        ticker_a=ticker_a, ticker_b=ticker_b,
        summary_df=summary_df, wealth_curve=wealth_curve,
        initial_budget=initial_budget, total_profit=total_profit,
        roi_pct=roi_pct, save_path=save_path,
    )

    return {
        "summary":      summary_df,
        "wealth_curve": wealth_curve,
        "total_profit": round(total_profit, 2),
        "final_wealth": round(final_wealth, 2),
        "roi_pct":      round(roi_pct, 2),
        "best_trade":   best_trade,
        "worst_trade":  worst_trade,
        "plot_path":    save_path,
    }


def _plot_simulation(
    ticker_a: str,
    ticker_b: str,
    summary_df: pd.DataFrame,
    wealth_curve: pd.Series,
    initial_budget: float,
    total_profit: float,
    roi_pct: float,
    save_path: str,
) -> None:
    """
    Génère un graphique en 3 panneaux pour la simulation de budget :
    1. Courbe de richesse en $ (avec ligne budget initial)
    2. P&L par trade en $ (barres vertes/rouges)
    3. Contribution de chaque jambe (ticker_A vs ticker_B) par trade
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    profit_color = "green" if total_profit >= 0 else "red"

    fig.suptitle(
        f"Simulation de budget -- {ticker_a} / {ticker_b}\n"
        f"Budget initial : {initial_budget:,.0f}$ | "
        f"Capital final : {initial_budget + total_profit:,.0f}$ | "
        f"Profit net : {total_profit:+,.0f}$ ({roi_pct:+.1f}%)",
        fontsize=12, fontweight="bold", color=profit_color,
    )

    # --- Panneau 1 : Courbe de richesse ---
    ax1 = axes[0]
    x = range(len(wealth_curve))
    ax1.plot(x, wealth_curve.values, color=profit_color,
             linewidth=1.8, marker="o", markersize=3, label="Capital ($)")
    ax1.axhline(initial_budget, color="gray", linestyle="--",
                linewidth=1.0, label=f"Budget initial : {initial_budget:,.0f}$")
    ax1.fill_between(
        x, initial_budget, wealth_curve.values,
        where=(np.array(wealth_curve.values) >= initial_budget),
        alpha=0.15, color="green",
    )
    ax1.fill_between(
        x, initial_budget, wealth_curve.values,
        where=(np.array(wealth_curve.values) < initial_budget),
        alpha=0.15, color="red",
    )

    # Annoter le point final
    ax1.annotate(
        f"  {wealth_curve.values[-1]:,.0f}$",
        xy=(len(wealth_curve) - 1, wealth_curve.values[-1]),
        fontsize=9, color=profit_color, fontweight="bold",
    )

    ax1.set_title("Courbe de richesse (capitalisation trade par trade)")
    ax1.set_xlabel("Numero du trade")
    ax1.set_ylabel("Capital ($)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}$")
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panneau 2 : P&L par trade en $ ---
    ax2 = axes[1]
    pnl_vals = summary_df["pnl_total_usd"].values
    trade_nums = summary_df["trade_n"].values
    colors = ["green" if p > 0 else "red" for p in pnl_vals]

    bars = ax2.bar(trade_nums, pnl_vals, color=colors, alpha=0.75, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)

    # Annoter les barres significatives (top/bottom 5)
    threshold_annot = np.percentile(np.abs(pnl_vals), 75)
    for bar, val in zip(bars, pnl_vals):
        if abs(val) >= threshold_annot:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + (3 if val > 0 else -3),
                f"{val:+.0f}$",
                ha="center", va="bottom" if val > 0 else "top",
                fontsize=6.5, color="black",
            )

    n_win  = int((summary_df["pnl_total_usd"] > 0).sum())
    n_loss = int((summary_df["pnl_total_usd"] <= 0).sum())
    avg_win  = summary_df.loc[summary_df["pnl_total_usd"] > 0,  "pnl_total_usd"].mean()
    avg_loss = summary_df.loc[summary_df["pnl_total_usd"] <= 0, "pnl_total_usd"].mean()

    ax2.set_title(
        f"P&L par trade ($) | {n_win} gagnants / {n_loss} perdants | "
        f"Gain moy.={avg_win:+.0f}$ | Perte moy.={avg_loss:+.0f}$"
        if not np.isnan(avg_win) else
        f"P&L par trade ($) | {n_win} gagnants / {n_loss} perdants"
    )
    ax2.set_xlabel("Numero du trade")
    ax2.set_ylabel("P&L ($)")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+,.0f}$")
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Panneau 3 : Contribution par jambe ---
    ax3 = axes[2]
    x_pos = np.arange(len(summary_df))
    width = 0.4

    pnl_a = summary_df["pnl_a_usd"].values
    pnl_b = summary_df["pnl_b_usd"].values

    ax3.bar(x_pos - width/2, pnl_a, width=width, alpha=0.75,
            color=["green" if v > 0 else "red" for v in pnl_a],
            label=f"{ticker_a} (jambe A)", edgecolor="white")
    ax3.bar(x_pos + width/2, pnl_b, width=width, alpha=0.75,
            color=["steelblue" if v > 0 else "orange" for v in pnl_b],
            label=f"{ticker_b} (jambe B)", edgecolor="white")
    ax3.axhline(0, color="black", linewidth=0.8)

    ax3.set_title(
        f"Contribution par jambe | "
        f"{ticker_a} total={pnl_a.sum():+,.0f}$ | "
        f"{ticker_b} total={pnl_b.sum():+,.0f}$"
    )
    ax3.set_xlabel("Numero du trade")
    ax3.set_ylabel("P&L par jambe ($)")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+,.0f}$")
    )
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Graphique simulation sauvegarde : %s", save_path)


# ---------------------------------------------------------------------------
# Graphique achats / ventes sur prix réels
# ---------------------------------------------------------------------------

def plot_trade_signals(
    ticker_a: str,
    ticker_b: str,
    data: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    zscore: pd.Series,
    zscore_threshold: float,
    save_dir: str | None = None,
) -> str:
    """
    Génère un graphique en 3 panneaux montrant les achats/ventes
    directement sur les prix réels des deux actifs.

    Panneau 1 : Prix de clôture de ticker_A + flèches d'entrée/sortie annotées
    Panneau 2 : Prix de clôture de ticker_B + flèches d'entrée/sortie annotées
    Panneau 3 : Z-Score avec P&L de chaque trade annoté

    Convention :
    - direction="LONG"  (long spread) → Achat A  / Vente B
    - direction="SHORT" (short spread) → Vente A  / Achat B

    Parameters
    ----------
    ticker_a : str
        Premier ticker de la paire.
    ticker_b : str
        Deuxième ticker de la paire.
    data : dict[str, pd.DataFrame]
        Données de prix réels (clé "Close") issues de data_engine.
    trades_df : pd.DataFrame
        Tableau des trades issu de run_backtest().
    zscore : pd.Series
        Z-Score roulant issu de run_backtest().
    zscore_threshold : float
        Seuil Z-Score utilisé pour la stratégie.
    save_dir : str | None
        Répertoire de sauvegarde (défaut : outputs/).

    Returns
    -------
    str
        Chemin du fichier PNG sauvegardé.
    """
    if save_dir is None:
        save_dir = OUTPUTS_DIR

    safe_a = ticker_a.replace(".", "_")
    safe_b = ticker_b.replace(".", "_")
    save_path = os.path.join(save_dir, f"trades_{safe_a}_{safe_b}.png")

    # Prix de clôture alignés
    close_a = data[ticker_a]["Close"].dropna()
    close_b = data[ticker_b]["Close"].dropna()
    common_idx = close_a.index.intersection(close_b.index)
    close_a = close_a.loc[common_idx]
    close_b = close_b.loc[common_idx]

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    fig.suptitle(
        f"Signaux de trading — {ticker_a} / {ticker_b} | seuil Z=±{zscore_threshold}",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, (ax, ticker, close, is_a) in enumerate(
        [(axes[0], ticker_a, close_a, True),
         (axes[1], ticker_b, close_b, False)]
    ):
        ax.plot(close.index, close.values, color="steelblue",
                linewidth=1.0, alpha=0.8, label=f"Prix {ticker}")

        if trades_df.empty:
            ax.set_title(f"{ticker} — Aucun trade généré")
            ax.set_ylabel("Prix ($)")
            ax.grid(True, alpha=0.3)
            continue

        for _, trade in trades_df.iterrows():
            entry_date = trade["entry_date"]
            exit_date  = trade["exit_date"]
            direction  = trade["direction"]   # "LONG" ou "SHORT"
            pnl        = trade["pnl"]

            # Pour ticker_A :
            #   LONG spread → on achète A à l'entrée, on vend A à la sortie
            #   SHORT spread → on vend A à l'entrée, on achète A à la sortie
            # Pour ticker_B : logique inverse
            if is_a:
                entry_is_buy = (direction == "LONG")
            else:
                entry_is_buy = (direction == "SHORT")

            # Récupérer les prix aux dates d'entrée/sortie
            entry_price = float(close.loc[entry_date]) if entry_date in close.index else None
            exit_price  = float(close.loc[exit_date])  if exit_date  in close.index else None

            if entry_price is None or exit_price is None:
                continue

            # Flèches d'entrée
            if entry_is_buy:
                ax.annotate(
                    f"  ACHAT\n  {entry_price:.2f}$",
                    xy=(entry_date, entry_price),
                    xytext=(entry_date, entry_price * 0.975),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                    fontsize=6.5, color="green", ha="center",
                )
            else:
                ax.annotate(
                    f"  VENTE\n  {entry_price:.2f}$",
                    xy=(entry_date, entry_price),
                    xytext=(entry_date, entry_price * 1.025),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    fontsize=6.5, color="red", ha="center",
                )

            # Flèches de sortie (sens inverse)
            if entry_is_buy:
                ax.annotate(
                    f"  VENTE\n  {exit_price:.2f}$",
                    xy=(exit_date, exit_price),
                    xytext=(exit_date, exit_price * 1.025),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    fontsize=6.5, color="red", ha="center",
                )
            else:
                ax.annotate(
                    f"  ACHAT\n  {exit_price:.2f}$",
                    xy=(exit_date, exit_price),
                    xytext=(exit_date, exit_price * 0.975),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                    fontsize=6.5, color="green", ha="center",
                )

            # Zone colorée entre entrée et sortie (vert si gagnant, rouge si perdant)
            region_idx = close.loc[entry_date:exit_date].index
            region_val = close.loc[region_idx]
            ax.fill_between(
                region_idx, region_val.min() * 0.995, region_val.max() * 1.005,
                alpha=0.06,
                color="green" if pnl > 0 else "red",
            )

        action_a = "LONG spread -> Achat A / Vente B"
        action_b = "SHORT spread -> Vente A / Achat B"
        ax.set_title(
            f"{ticker} — Prix reel avec signaux\n"
            f"(vert=achat, rouge=vente | {action_a if is_a else action_b})"
        )
        ax.set_ylabel("Prix de cloture ($)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Panneau 3 : Z-Score + P&L par trade annoté ---
    ax3 = axes[2]
    z_clean = zscore.dropna()
    ax3.plot(z_clean.index, z_clean.values, color="navy", linewidth=0.9, label="Z-Score")
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax3.axhline(+zscore_threshold, color="red",  linestyle="--", linewidth=1.0,
                label=f"+{zscore_threshold}s (SHORT spread)")
    ax3.axhline(-zscore_threshold, color="blue", linestyle="--", linewidth=1.0,
                label=f"-{zscore_threshold}s (LONG spread)")

    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            entry_date = trade["entry_date"]
            exit_date  = trade["exit_date"]
            pnl        = trade["pnl"]
            direction  = trade["direction"]
            entry_z    = trade["entry_zscore"]

            color = "green" if pnl > 0 else "red"

            # Ligne verticale à l'entrée
            ax3.axvline(entry_date, color=color, alpha=0.4, linewidth=0.8)

            # Annotation du P&L au niveau du Z-Score d'entrée
            if not pd.isna(entry_z):
                ax3.annotate(
                    f"P&L\n{'+' if pnl>0 else ''}{pnl:.4f}",
                    xy=(entry_date, entry_z),
                    fontsize=5.5, color=color,
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.7),
                )

    n_trades = len(trades_df) if not trades_df.empty else 0
    win_trades = int((trades_df["pnl"] > 0).sum()) if not trades_df.empty else 0
    ax3.set_title(
        f"Z-Score du spread | {n_trades} trades | "
        f"{win_trades} gagnants / {n_trades - win_trades} perdants"
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Z-Score")
    ax3.set_ylim(
        min(-zscore_threshold * 2.5, z_clean.min() * 1.1),
        max(+zscore_threshold * 2.5, z_clean.max() * 1.1),
    )
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Graphique trades sauvegarde : %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Validation autonome
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
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

    print("\n=== Validation — backtester.py ===\n")

    # Charger les données
    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=config.START_DATE,
        end=config.END_DATE,
    )

    # Alphas bruts (avant filtre) — nécessaires pour le grid search
    denoiser = AlphaDeNoiser(window=config.ROLLING_WINDOW)
    alphas_raw = denoiser.extract_alpha(
        data, data[config.MARKET_TICKER]["log_return"], config.SECTOR_MAP
    )

    # Alphas filtrés — pour le backtest principal
    alphas_filtered: dict[str, pd.Series] = {
        ticker: apply_bandpass_filter(
            alpha,
            low_days=config.FILTER_LOW_DAYS,
            high_days=config.FILTER_HIGH_DAYS,
        )
        for ticker, alpha in alphas_raw.items()
    }

    # Paires
    pairs_df = detect_pairs(
        alphas=alphas_filtered,
        sector_map=config.SECTOR_MAP,
        corr_threshold=0.3,   # Seuil bas pour inclure toutes les paires
        max_lag=config.LAG_MAX_DAYS,
        data=data,
        save_outputs=False,
    )

    print(f"Paires disponibles :\n{pairs_df[['Ticker_A','Ticker_B','Correlation']].to_string(index=False)}\n")

    # --- Backtest principal sur PEP/KO ---
    print("--- Backtest PEP/KO (paramètres par défaut) ---")
    result_pep_ko = run_backtest(
        pair=("PEP", "KO"),
        alphas=alphas_filtered,
        zscore_threshold=config.ZSCORE_THRESHOLD,
        rolling_window=config.ROLLING_WINDOW,
        save_outputs=True,
    )

    m = result_pep_ko["metrics"]
    print(f"\nMétriques PEP/KO :")
    print(f"  Sharpe Ratio     : {m['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio    : {m['sortino_ratio']:.3f}")
    print(f"  Max Drawdown     : {m['max_drawdown']:.4f}")
    print(f"  Win Rate         : {m['winrate']*100:.1f}%")
    print(f"  Rendement annuel : {m['annual_return']:.4f}")
    print(f"  Nombre de trades : {m['n_trades']}")
    print(f"  P&L total        : {m['total_pnl']:.4f}")

    if not result_pep_ko["trades"].empty:
        print(f"\nPremiers trades :")
        print(result_pep_ko["trades"].head(5).to_string(index=False))

    print(f"\nGraphique sauvegardé : {OUTPUTS_DIR}/backtest_PEP_KO.png")

    # --- Grid Search ---
    print("\n--- Grid Search (tous les paramètres) ---")
    grid_df = grid_search(
        pairs_df=pairs_df,
        alphas_raw=alphas_raw,
        zscore_range=[1.5, 2.0, 2.5, 3.0],
        filter_range=[(3.0, 126.0), (3.0, 252.0), (5.0, 252.0)],
        rolling_window=config.ROLLING_WINDOW,
        save_outputs=True,
    )

    if not grid_df.empty:
        print(f"\nTop 5 combinaisons (Sharpe) :")
        print(grid_df.head(5)[
            ["ticker_a", "ticker_b", "zscore_threshold",
             "low_days", "high_days", "sharpe_ratio", "winrate", "n_trades"]
        ].to_string(index=False))
        print(f"\nHeatmap sauvegardée : {OUTPUTS_DIR}/grid_search_heatmap.png")

    # Validation structurelle
    assert "metrics" in result_pep_ko,     "ERREUR : clé 'metrics' manquante"
    assert "trades" in result_pep_ko,      "ERREUR : clé 'trades' manquante"
    assert "wealth_curve" in result_pep_ko,"ERREUR : clé 'wealth_curve' manquante"
    assert "sharpe_ratio" in m,            "ERREUR : clé 'sharpe_ratio' manquante"
    assert "winrate" in m,                 "ERREUR : clé 'winrate' manquante"

    print("\nValidation OK — backtest et grid search fonctionnels.")
