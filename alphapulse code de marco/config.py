"""
config.py — Paramètres globaux configurables pour AlphaPulse
"""

import os

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

# Tickers de test pour la phase académique
TICKERS: list[str] = ["PEP", "KO", "CNR.TO", "CP.TO", "XOM", "CVX"]

# Marché de référence (Vanguard Total Stock Market ETF)
MARKET_TICKER: str = "VTI"

# ETF sectoriels par ticker
SECTOR_MAP: dict[str, str] = {
    "PEP": "XLP",    # Consumer Staples
    "KO": "XLP",
    "CNR.TO": "XLI", # Industrials
    "CP.TO": "XLI",
    "XOM": "XLE",    # Energy
    "CVX": "XLE",
}

# ---------------------------------------------------------------------------
# Période historique
# ---------------------------------------------------------------------------

START_DATE: str = "2019-01-01"
END_DATE: str = "2024-12-31"

# ---------------------------------------------------------------------------
# Paramètres du filtre passe-bande (en jours)
# ---------------------------------------------------------------------------

FILTER_LOW_DAYS: float = 3      # Coupure basse — retire le bruit court terme (<3 jours)
FILTER_HIGH_DAYS: float = 252   # Coupure haute — retire la tendance longue (>1 an)

# ---------------------------------------------------------------------------
# Paramètres de décision
# ---------------------------------------------------------------------------

ZSCORE_THRESHOLD: float = 2.0        # Seuil d'activation du signal
CORRELATION_THRESHOLD: float = 0.6   # Corrélation croisée minimale pour retenir une paire
LAG_MAX_DAYS: int = 15               # Décalage maximal pour l'analyse lead-lag (jours)

# Fenêtre roulante pour OLS (AlphaDeNoiser) et Z-Score
ROLLING_WINDOW: int = 60

# ---------------------------------------------------------------------------
# Clé API Gemini
# Priorité : variable d'environnement GEMINI_API_KEY > valeur ci-dessous
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "VOTRE_CLE_ICI")
