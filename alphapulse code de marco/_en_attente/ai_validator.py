"""
ai_validator.py — Module 5 : Validation Contextuelle par IA (Gemini + Search Grounding)
Pipeline AlphaPulse
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# Clé API chargée depuis config (ou variable d'environnement)
_GEMINI_API_KEY: str | None = None


def _get_api_key() -> str:
    """Retourne la clé API Gemini depuis la config ou l'environnement."""
    global _GEMINI_API_KEY
    if _GEMINI_API_KEY is not None:
        return _GEMINI_API_KEY

    # Priorité 1 : variable d'environnement
    key = os.environ.get("GEMINI_API_KEY", "")
    if key and key != "VOTRE_CLE_ICI":
        _GEMINI_API_KEY = key
        return key

    # Priorité 2 : config.py
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        import config as cfg
        if cfg.GEMINI_API_KEY and cfg.GEMINI_API_KEY != "VOTRE_CLE_ICI":
            _GEMINI_API_KEY = cfg.GEMINI_API_KEY
            return cfg.GEMINI_API_KEY
    except Exception:
        pass

    return ""


def _build_prompt(
    ticker_a: str,
    ticker_b: str,
    zscore: float,
    spread_direction: str,
) -> str:
    """Construit le prompt système pour Gemini."""
    return (
        f"Tu es un analyste financier. Analyse les nouvelles récentes concernant "
        f"{ticker_a} et {ticker_b}. "
        f"Le spread de prix entre ces deux actions a atteint un Z-Score de {zscore:.2f} "
        f"(anomalie statistique détectée, direction : {spread_direction}). "
        f"Identifie s'il existe un catalyseur fondamental (résultats trimestriels, "
        f"procès, changement de direction, événement géopolitique sectoriel) qui "
        f"justifie cet écart. "
        f"Si l'écart est une anomalie sans justification fondamentale, réponds SIGNAL_VALIDE. "
        f"Si l'écart est justifié par un événement grave, réponds SIGNAL_IGNORE avec une "
        f"explication courte. "
        f"Retourne UNIQUEMENT un JSON valide avec les clés : "
        f"signal (SIGNAL_VALIDE ou SIGNAL_IGNORE), justification (str), "
        f"confiance (float entre 0 et 1), sources (liste de strings)."
    )


def _parse_gemini_response(text: str) -> dict:
    """
    Extrait le JSON structuré de la réponse Gemini.
    Gère les cas où le modèle enveloppe le JSON dans du texte ou du markdown.
    """
    # Chercher un bloc JSON dans la réponse
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Tentative de parsing direct
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Fallback : construire une réponse à partir du texte
    signal = "SIGNAL_VALIDE" if "SIGNAL_VALIDE" in text.upper() else "SIGNAL_IGNORE"
    return {
        "signal": signal,
        "justification": text[:300].strip(),
        "confiance": 0.5,
        "sources": [],
    }


def validate_signal(
    ticker_a: str,
    ticker_b: str,
    zscore: float,
    spread_direction: str,
    max_retries: int = 2,
    timeout_seconds: int = 30,
) -> dict:
    """
    Valide un signal de trading via l'API Gemini avec Google Search Grounding.

    Parameters
    ----------
    ticker_a : str
        Premier ticker de la paire.
    ticker_b : str
        Deuxième ticker de la paire.
    zscore : float
        Z-Score du spread (valeur de l'anomalie statistique).
    spread_direction : str
        Direction du spread (ex. "LONG A / SHORT B" ou "LONG B / SHORT A").
    max_retries : int
        Nombre de tentatives en cas d'erreur API.
    timeout_seconds : int
        Timeout par tentative (non utilisé avec google-generativeai, indicatif).

    Returns
    -------
    dict avec clés :
        - "signal"        : "SIGNAL_VALIDE" | "SIGNAL_IGNORE"
        - "justification" : str
        - "confiance"     : float (0–1)
        - "sources"       : list[str]

    Note
    ----
    En cas d'échec API (timeout, quota, clé invalide), un fallback
    SIGNAL_VALIDE est retourné avec confiance=0.0 pour ne pas bloquer
    le pipeline. Ce comportement est documenté dans les logs.
    """
    api_key = _get_api_key()

    if not api_key:
        logger.warning(
            "Clé API Gemini non configurée. Fallback SIGNAL_VALIDE pour (%s, %s).",
            ticker_a, ticker_b,
        )
        return {
            "signal": "SIGNAL_VALIDE",
            "justification": "Clé API Gemini non configurée — validation IA désactivée.",
            "confiance": 0.0,
            "sources": [],
        }

    prompt = _build_prompt(ticker_a, ticker_b, zscore, spread_direction)

    for attempt in range(1, max_retries + 1):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # Modèle Gemini avec Google Search Grounding (nouveau SDK)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

            raw_text = response.text
            logger.info("Réponse Gemini brute (%s, %s) : %s", ticker_a, ticker_b, raw_text[:200])

            result = _parse_gemini_response(raw_text)

            # Extraire les sources depuis les grounding metadata si disponibles
            sources: list[str] = result.get("sources", [])
            try:
                if response.candidates:
                    candidate = response.candidates[0]
                    gm = getattr(candidate, "grounding_metadata", None)
                    if gm:
                        chunks = getattr(gm, "grounding_chunks", []) or []
                        for chunk in chunks:
                            web = getattr(chunk, "web", None)
                            uri = getattr(web, "uri", None) if web else None
                            if uri:
                                sources.append(uri)
            except Exception:
                pass

            result["sources"] = sources[:5]  # Limiter à 5 sources

            # Normaliser le signal
            signal_upper = result.get("signal", "").upper()
            if "VALIDE" in signal_upper:
                result["signal"] = "SIGNAL_VALIDE"
            elif "IGNORE" in signal_upper:
                result["signal"] = "SIGNAL_IGNORE"
            else:
                result["signal"] = "SIGNAL_VALIDE"  # Par défaut

            logger.info(
                "Validation IA (%s, %s) : signal=%s, confiance=%.2f",
                ticker_a, ticker_b, result["signal"], result.get("confiance", 0.0),
            )
            return result

        except Exception as exc:
            logger.warning(
                "Tentative %d/%d échouée pour (%s, %s) : %s",
                attempt, max_retries, ticker_a, ticker_b, exc,
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Backoff exponentiel

    # Fallback après tous les échecs
    logger.error(
        "Toutes les tentatives Gemini échouées pour (%s, %s). Fallback SIGNAL_VALIDE.",
        ticker_a, ticker_b,
    )
    return {
        "signal": "SIGNAL_VALIDE",
        "justification": f"Erreur API Gemini après {max_retries} tentatives — validation IA indisponible.",
        "confiance": 0.0,
        "sources": [],
    }


# ---------------------------------------------------------------------------
# Validation autonome
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("\n=== Validation — ai_validator.py ===\n")
    print("Test avec la paire PEP/KO, Z-Score fictif = 2.3")
    print("(Si la clé API n'est pas configurée, un fallback sera retourné)\n")

    result = validate_signal(
        ticker_a="PEP",
        ticker_b="KO",
        zscore=2.3,
        spread_direction="LONG PEP / SHORT KO",
    )

    print("--- Résultat JSON ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    assert "signal" in result, "ERREUR : clé 'signal' absente"
    assert result["signal"] in ("SIGNAL_VALIDE", "SIGNAL_IGNORE"), \
        f"ERREUR : valeur inattendue pour 'signal' : {result['signal']}"
    assert "justification" in result, "ERREUR : clé 'justification' absente"
    assert "confiance" in result, "ERREUR : clé 'confiance' absente"
    assert "sources" in result, "ERREUR : clé 'sources' absente"

    print("\nValidation OK — structure JSON conforme.")
