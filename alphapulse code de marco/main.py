"""
main.py -- Orchestrateur du Pipeline AlphaPulse
================================================

Lance le pipeline complet de bout en bout :

    load_data -> extract_alpha -> apply_filter -> detect_pairs
              -> run_decision -> [optionnel: run_backtest + plot_trade_signals]

Usage
-----
    python -X utf8 main.py                   # Pipeline interactif (sans backtest)
    python -X utf8 main.py --backtest        # Pipeline interactif + backtest complet
    python -X utf8 main.py --no-interactive  # Utilise les valeurs de config.py directement
    python -X utf8 main.py --backtest --no-interactive

Note : aucun ordre reel n'est execute. Ce pipeline est exclusivement
       en mode simulation/analyse academique.
"""

import argparse
import io
import logging
import os
import sys
import time

# Forcer UTF-8 sur stdout/stderr (encodage Windows cp1252 incompatible)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("alphapulse.main")

# ---------------------------------------------------------------------------
# Banniere
# ---------------------------------------------------------------------------
BANNER = """
+----------------------------------------------------------+
|         AlphaPulse -- Pipeline d'Arbitrage               |
|      Statistique & Fondamental  |  v1.2                  |
|  Mode : Simulation academique -- AUCUN ORDRE REEL        |
+----------------------------------------------------------+
"""

SEP  = "-" * 62
SEP2 = "=" * 62


# ---------------------------------------------------------------------------
# Parsing CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaPulse -- Pipeline d'arbitrage statistique (simulation uniquement).",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Active le mode backtesting + grid search.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Saute la configuration interactive et utilise config.py directement.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactive la sauvegarde des graphiques PNG.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration interactive
# ---------------------------------------------------------------------------

def _ask(prompt: str, default, cast):
    """Pose une question avec une valeur par defaut. Retourne la valeur castee."""
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"  [!] Valeur invalide, conserve la valeur par defaut : {default}")
        return default


def configure_pipeline(config) -> dict:
    """
    Affiche un menu interactif pour parametrer le pipeline.
    Chaque question propose la valeur de config.py comme defaut.

    Retourne un dictionnaire de parametres actifs.
    """
    print(f"\n{SEP}")
    print("  CONFIGURATION DU PIPELINE")
    print(f"  (Appuyez sur Entree pour garder la valeur par defaut)")
    print(SEP)

    print("\n  -- Periode historique --")
    start = _ask(
        f"  Date de debut  [{config.START_DATE}] : ",
        config.START_DATE, str,
    )
    end = _ask(
        f"  Date de fin    [{config.END_DATE}] : ",
        config.END_DATE, str,
    )

    print("\n  -- Filtre frequentiel (Butterworth passe-bande) --")
    print("  Astuce overfitting : changer high de 252 a 126 jours reduit le look-back")
    filter_low = _ask(
        f"  Coupure basse  [{config.FILTER_LOW_DAYS:.0f} jours] : ",
        config.FILTER_LOW_DAYS, float,
    )
    filter_high = _ask(
        f"  Coupure haute  [{config.FILTER_HIGH_DAYS:.0f} jours] : ",
        config.FILTER_HIGH_DAYS, float,
    )

    print("\n  -- Parametres de decision --")
    print("  Astuce overfitting : tester Z=1.5 vs 2.0 vs 2.5 sur des periodes differentes")
    zscore_threshold = _ask(
        f"  Seuil Z-Score  [{config.ZSCORE_THRESHOLD}] : ",
        config.ZSCORE_THRESHOLD, float,
    )
    rolling_window = _ask(
        f"  Fenetre roulante [{config.ROLLING_WINDOW} jours] : ",
        config.ROLLING_WINDOW, int,
    )

    print("\n  -- Detection des paires --")
    print("  Astuce overfitting : si corr=0.3 donne de bons resultats mais 0.6 non -> suspect")
    corr_threshold = _ask(
        f"  Seuil correlation [{config.CORRELATION_THRESHOLD}] : ",
        config.CORRELATION_THRESHOLD, float,
    )

    print("\n  -- Simulateur de budget --")
    print("  Le simulateur calcule le profit reel en $ par paire avec les vrais prix.")
    budget = _ask(
        f"  Budget initial [$] [10000] : ",
        10_000.0, float,
    )

    params = {
        "start":            start,
        "end":              end,
        "filter_low":       filter_low,
        "filter_high":      filter_high,
        "zscore_threshold": zscore_threshold,
        "rolling_window":   rolling_window,
        "corr_threshold":   corr_threshold,
        "budget":           budget,
    }

    print(f"\n{SEP}")
    print("  Parametres retenus :")
    for k, v in params.items():
        print(f"    {k:20s} = {v}")
    print(SEP)
    confirm = input("\n  Confirmer ? [O/n] : ").strip().lower()
    if confirm == "n":
        print("  -> Relance de la configuration...\n")
        return configure_pipeline(config)

    return params


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestre l'ensemble des modules AlphaPulse dans l'ordre defini."""

    print(BANNER)
    t_start = time.time()

    module_dir = os.path.dirname(os.path.abspath(__file__))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    # -----------------------------------------------------------------------
    # Imports
    # -----------------------------------------------------------------------
    try:
        import config
        from data_engine import load_data
        from alpha_denoiser import AlphaDeNoiser
        from signal_filter import apply_bandpass_filter
        from pairs_detector import detect_pairs
        from decision_engine import run_decision
    except ImportError as exc:
        logger.error("Erreur d'import : %s", exc)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    if args.no_interactive:
        params = {
            "start":            config.START_DATE,
            "end":              config.END_DATE,
            "filter_low":       config.FILTER_LOW_DAYS,
            "filter_high":      config.FILTER_HIGH_DAYS,
            "zscore_threshold": config.ZSCORE_THRESHOLD,
            "rolling_window":   config.ROLLING_WINDOW,
            "corr_threshold":   config.CORRELATION_THRESHOLD,
            "budget":           10_000.0,
        }
        print("  [Mode non-interactif] Parametres lus depuis config.py")
    else:
        params = configure_pipeline(config)

    save_outputs = not args.no_plots

    # ===================================================================
    # ETAPE 1 -- Acquisition
    # ===================================================================
    print(f"\n{SEP}")
    print("ETAPE 1/5 -- Acquisition des donnees (yfinance)")
    print(SEP)

    data = load_data(
        tickers=config.TICKERS,
        market=config.MARKET_TICKER,
        sector_map=config.SECTOR_MAP,
        start=params["start"],
        end=params["end"],
    )

    n_days = len(data[config.TICKERS[0]])
    print(f"  [OK] {len(data)} actifs | {n_days} jours "
          f"({params['start']} -> {params['end']})")

    # ===================================================================
    # ETAPE 2 -- Alpha DeNoiser
    # ===================================================================
    print(f"\n{SEP}")
    print("ETAPE 2/5 -- Isolation de l'Alpha (OLS roulant)")
    print(SEP)

    market_returns = data[config.MARKET_TICKER]["log_return"]
    denoiser = AlphaDeNoiser(window=params["rolling_window"])
    alphas_raw = denoiser.extract_alpha(data, market_returns, config.SECTOR_MAP)

    for ticker, alpha in alphas_raw.items():
        valid = alpha.dropna()
        raw_std   = data[ticker]["log_return"].loc[valid.index].std()
        alpha_std = valid.std()
        reduction = 100 * (1 - alpha_std / raw_std) if raw_std > 0 else 0.0
        print(f"  [OK] {ticker:8s} | reduction variance : {reduction:.1f}%")

    # ===================================================================
    # ETAPE 3 -- Filtre frequentiel
    # ===================================================================
    print(f"\n{SEP}")
    print(f"ETAPE 3/5 -- Filtrage frequentiel "
          f"[{params['filter_low']:.0f}j - {params['filter_high']:.0f}j]")
    print(SEP)

    alphas_filtered: dict = {}
    for ticker, alpha in alphas_raw.items():
        alphas_filtered[ticker] = apply_bandpass_filter(
            alpha,
            low_days=params["filter_low"],
            high_days=params["filter_high"],
        )
        n_valid = alphas_filtered[ticker].dropna().shape[0]
        print(f"  [OK] {ticker:8s} | {n_valid} points filtres")

    # ===================================================================
    # ETAPE 4 -- Detection des paires
    # ===================================================================
    print(f"\n{SEP}")
    print(f"ETAPE 4/5 -- Detection des paires (correlation >= {params['corr_threshold']})")
    print(SEP)

    pairs_df = detect_pairs(
        alphas=alphas_filtered,
        sector_map=config.SECTOR_MAP,
        corr_threshold=params["corr_threshold"],
        max_lag=config.LAG_MAX_DAYS,
        data=data,
        save_outputs=save_outputs,
    )

    if pairs_df.empty:
        print(f"  [!] Aucune paire avec correlation >= {params['corr_threshold']}.")
        print(f"  -> Relance avec seuil abaisse a 0.3...")
        pairs_df = detect_pairs(
            alphas=alphas_filtered,
            sector_map=config.SECTOR_MAP,
            corr_threshold=0.3,
            max_lag=config.LAG_MAX_DAYS,
            data=data,
            save_outputs=save_outputs,
        )

    if pairs_df.empty:
        print("  [X] Aucune paire detectee. Fin du pipeline.")
        return

    print(f"\n  {len(pairs_df)} paire(s) retenue(s) :")
    for _, row in pairs_df.iterrows():
        print(f"    * {row['Ticker_A']:8s} / {row['Ticker_B']:8s} | "
              f"corr={row['Correlation']:.3f} | p-value={row['P_value']:.4f} | "
              f"lag={row['Lag_optimal']}j | leader={row['Leader']}")

    # ===================================================================
    # ETAPE 5 -- Moteur de decision
    # ===================================================================
    print(f"\n{SEP}")
    print(f"ETAPE 5/5 -- Signaux Z-Score (seuil +/-{params['zscore_threshold']})")
    print(SEP)
    print("  [Validation IA desactivee -- ai_validator en _en_attente/]")

    signals = run_decision(
        pairs_df=pairs_df,
        alphas_filtered=alphas_filtered,
        zscore_threshold=params["zscore_threshold"],
        rolling_window=params["rolling_window"],
    )

    if signals:
        print(f"\n  {len(signals)} signal(s) actif(s) :\n")
        for sig in signals:
            print(f"    +- Paire     : {sig['ticker_a']} / {sig['ticker_b']}")
            print(f"    |  Leader    : {sig['ticker_leader']}  ->  Laggard : {sig['ticker_laggard']}")
            print(f"    |  Signal    : {sig['signal']}")
            print(f"    |  Direction : {sig['spread_direction']}")
            print(f"    |  Z-Score   : {sig['zscore']:.3f}")
            print(f"    |  Lag       : {sig['lag']} jours")
            print(f"    +- Horodatage: {sig['timestamp']}")
            print()
    else:
        print(f"\n  Aucun signal actif (spreads dans la normale +/-{params['zscore_threshold']}s).")
        print("  -> Comportement attendu quand le marche est en equilibre.")

    # ===================================================================
    # [OPTIONNEL] BACKTEST + GRAPHIQUES ACHATS/VENTES
    # ===================================================================
    if args.backtest:
        try:
            from backtester import run_backtest, grid_search, plot_trade_signals, simulate_budget
        except ImportError as exc:
            logger.error("backtester.py introuvable : %s", exc)
            return

        print(f"\n{SEP2}")
        print("BACKTEST -- Validation historique")
        print(f"  Periode : {params['start']} -> {params['end']}")
        print(f"  Z-Score : +/-{params['zscore_threshold']} | "
              f"Filtre : [{params['filter_low']:.0f}j-{params['filter_high']:.0f}j] | "
              f"Fenetre : {params['rolling_window']}j")
        print("  [Simulation pure -- aucun appel IA]")
        print(SEP2)

        all_metrics: list[dict] = []

        for _, row in pairs_df.iterrows():
            ta, tb = str(row["Ticker_A"]), str(row["Ticker_B"])
            print(f"\n  Backtest {ta}/{tb}...")

            result = run_backtest(
                pair=(ta, tb),
                alphas=alphas_filtered,
                zscore_threshold=params["zscore_threshold"],
                rolling_window=params["rolling_window"],
                save_outputs=save_outputs,
            )
            m = result["metrics"]
            all_metrics.append({"pair": f"{ta}/{tb}", **m})

            print(f"    Sharpe={m['sharpe_ratio']:.2f} | "
                  f"Sortino={m['sortino_ratio']:.2f} | "
                  f"MaxDD={m['max_drawdown']:.4f} | "
                  f"WinRate={m['winrate']*100:.1f}% | "
                  f"Trades={m['n_trades']} | "
                  f"ReturnAnnuel={m['annual_return']:.3f}")

            # ----- Graphique achats/ventes sur prix reels -----
            if save_outputs and not result["trades"].empty:
                print(f"    Generation du graphique achats/ventes...")
                trades_path = plot_trade_signals(
                    ticker_a=ta,
                    ticker_b=tb,
                    data=data,
                    trades_df=result["trades"],
                    zscore=result["zscore"],
                    zscore_threshold=params["zscore_threshold"],
                )
                print(f"    [OK] Graphique sauvegarde : {os.path.basename(trades_path)}")

            # ----- Simulateur de budget en dollars -----
            print(f"    Simulation avec budget initial : {params['budget']:,.0f}$")
            sim = simulate_budget(
                ticker_a=ta,
                ticker_b=tb,
                data=data,
                trades_df=result["trades"],
                initial_budget=params["budget"],
                save_dir=os.path.join(module_dir, "outputs"),
            )

            if not sim["summary"].empty:
                print(f"\n    +{'─'*58}+")
                print(f"    | SIMULATION {ta}/{tb:<44}|")
                print(f"    +{'─'*58}+")
                print(f"    | Budget initial  : {params['budget']:>10,.2f}$"
                      f"{'':>28}|")
                print(f"    | Capital final   : {sim['final_wealth']:>10,.2f}$"
                      f"{'':>28}|")
                print(f"    | Profit net      : {sim['total_profit']:>+10,.2f}$"
                      f"{'':>28}|")
                print(f"    | ROI total       : {sim['roi_pct']:>+10.1f}%"
                      f"{'':>28}|")
                print(f"    | Nb trades       : {len(sim['summary']):>10d}"
                      f"{'':>28}|")
                if sim["best_trade"]:
                    print(f"    | Meilleur trade  : {sim['best_trade']['pnl_total_usd']:>+10,.2f}$"
                          f" (trade #{int(sim['best_trade']['trade_n'])})"
                          f"{'':>14}|")
                if sim["worst_trade"]:
                    print(f"    | Pire trade      : {sim['worst_trade']['pnl_total_usd']:>+10,.2f}$"
                          f" (trade #{int(sim['worst_trade']['trade_n'])})"
                          f"{'':>14}|")
                print(f"    +{'─'*58}+")

                # Tableau des 5 premiers trades
                print(f"\n    Premiers trades (en $) :")
                cols_show = ["trade_n", "entry_date", "exit_date", "direction",
                             "pnl_total_usd", "roi_trade_pct", "capital_apres"]
                print("    " + sim["summary"][cols_show].head(8).to_string(
                    index=False,
                    formatters={
                        "pnl_total_usd":  lambda v: f"{v:+.2f}$",
                        "roi_trade_pct":  lambda v: f"{v:+.3f}%",
                        "capital_apres":  lambda v: f"{v:,.2f}$",
                    }
                ).replace("\n", "\n    "))

                if save_outputs:
                    print(f"\n    [OK] Graphique simulation : "
                          f"{os.path.basename(sim['plot_path'])}")

        # --- Tableau recapitulatif ---
        print(f"\n{SEP}")
        print("  Recapitulatif backtests :")
        print(f"  {'Paire':20s} {'Sharpe':>8} {'WinRate':>9} "
              f"{'Trades':>7} {'ReturnAnn.':>11}")
        print("  " + "-" * 58)
        for m in all_metrics:
            print(f"  {m['pair']:20s} {m['sharpe_ratio']:>8.2f} "
                  f"{m['winrate']*100:>8.1f}% {m['n_trades']:>7d} "
                  f"{m['annual_return']:>11.3f}")

        # --- Grid Search ---
        print(f"\n{SEP}")
        print("GRID SEARCH -- Optimisation des parametres")
        print(f"  (Outil de detection d'overfitting : des Sharpe tres variables")
        print(f"   entre les combinaisons indiquent une sensibilite aux parametres)")
        print(SEP)

        grid_df = grid_search(
            pairs_df=pairs_df,
            alphas_raw=alphas_raw,
            zscore_range=[1.5, 2.0, 2.5, 3.0],
            filter_range=[
                (params["filter_low"], params["filter_high"] / 2),
                (params["filter_low"], params["filter_high"]),
                (params["filter_low"] + 2, params["filter_high"]),
            ],
            rolling_window=params["rolling_window"],
            save_outputs=save_outputs,
        )

        if not grid_df.empty:
            sharpe_std = grid_df["sharpe_ratio"].std()
            sharpe_min = grid_df["sharpe_ratio"].min()
            sharpe_max = grid_df["sharpe_ratio"].max()

            print(f"\n  Variabilite du Sharpe sur la grille :")
            print(f"    Min={sharpe_min:.2f} | Max={sharpe_max:.2f} | "
                  f"Ecart-type={sharpe_std:.2f}")
            if sharpe_std > 5:
                print("  [!] Forte variabilite -> risque d'overfitting detecte.")
            else:
                print("  [OK] Faible variabilite -> resultats relativement stables.")

            print(f"\n  Top 5 combinaisons (Sharpe) :")
            print(f"  {'Paire':15s} {'Z':>5} {'Filtre':>14} "
                  f"{'Sharpe':>8} {'WinRate':>9} {'Trades':>7}")
            print("  " + "-" * 62)
            for _, r in grid_df.head(5).iterrows():
                print(f"  {r['ticker_a']}/{r['ticker_b']:10s} "
                      f"{r['zscore_threshold']:>5.1f} "
                      f"[{int(r['low_days'])}j-{int(r['high_days'])}j]{'':<5} "
                      f"{r['sharpe_ratio']:>8.2f} "
                      f"{r['winrate']*100:>8.1f}% "
                      f"{r['n_trades']:>7d}")

    # ===================================================================
    # Resume final
    # ===================================================================
    elapsed = time.time() - t_start
    print(f"\n{SEP2}")
    print(f"  Pipeline termine en {elapsed:.1f}s")
    print(f"  Actifs analyses  : {len(config.TICKERS)}")
    print(f"  Paires detectees : {len(pairs_df)}")
    print(f"  Signaux actifs   : {len(signals)}")
    if save_outputs:
        outputs_dir = os.path.join(module_dir, "outputs")
        png_count = sum(1 for f in os.listdir(outputs_dir) if f.endswith(".png"))
        print(f"  Graphiques PNG   : {png_count} dans {outputs_dir}/")
    print(SEP2)
    print("\n  RAPPEL : Ce pipeline est en mode simulation academique.")
    print("           Aucun ordre reel n'a ete emis.")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
