from stockdex import Ticker
import numpy
import matplotlib.pyplot as plt

us_companies = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "Alphabet Inc. (Class A)": "GOOGL",
    "Alphabet Inc. (Class C)": "GOOG",
    "Meta Platforms Inc.": "META",
    "Tesla Inc.": "TSLA",
    "NVIDIA Corporation": "NVDA",
    "Berkshire Hathaway Inc. (Class B)": "BRK.B",
    "Johnson & Johnson": "JNJ",
    "JPMorgan Chase & Co.": "JPM",
    "Visa Inc.": "V",
    "Procter & Gamble Co.": "PG",
    "UnitedHealth Group Incorporated": "UNH",
    "Mastercard Incorporated": "MA",
    "The Home Depot Inc.": "HD",
    "Chevron Corporation": "CVX",
    "Exxon Mobil Corporation": "XOM",
    "Pfizer Inc.": "PFE",
    "Ccao-Cola Company": "KO",
    "PepsiCo Inc.": "PEP",
    "Walmart Inc.": "WMT",
    "Intel Corporation": "INTC",
    "Cisco Systems Inc.": "CSCO",
    "Adobe Inc.": "ADBE",
    "Netflix Inc.": "NFLX",
    "Salesforce Inc.": "CRM",
    "Broadcom Inc.": "AVGO",
    "Costco Wholesale Corporation": "COST",
    "AbbVie Inc.": "ABBV",
    "Merck & Co. Inc.": "MRK",
    "McDonald's Corporation": "MCD",
    "Nike Inc.": "NKE",
    "Starbucks Corporation": "SBUX",
    "Goldman Sachs Group Inc.": "GS",
    "Morgan Stanley": "MS",
    "American Express Company": "AXP",
    "IBM": "IBM",
    "Oracle Corporation": "ORCL",
    "Texas Instruments Incorporated": "TXN",
    "Qualcomm Incorporated": "QCOM",
    "AMD (Advanced Micro Devices)": "AMD",
    "General Electric Company": "GE",
    "Ford Motor Company": "F",
    "General Motors Company": "GM",
    "Lockheed Martin Corporation": "LMT",
    "Boeing Company": "BA",
    "3M Company": "MMM",
    "Caterpillar Inc.": "CAT",
    "UPS (United Parcel Service)": "UPS"
}

# action = "AAPL"  #PEP KO AMZN AAPL MSFT
action = us_companies["Ccao-Cola Company"]

def simulate_trading_lagged(price, initial_cash=2000):
    cash = initial_cash
    shares = 0
    portfolio_values = []

    # On commence à i = 2 (il faut 2 jours d'historique)
    for i in range(len(price)):
        
        # Pas assez d'info → rien faire
        if i < 2:
            total_value = cash + shares * price[i]
            portfolio_values.append(total_value)
            continue

        today = price[i]
        yesterday = price[i-1]
        day_before = price[i-2]

        # Signal basé sur (i-2 → i-1)
        if yesterday > day_before:
            # BUY ALL
            if cash > 0:
                shares = cash / today
                cash = 0

        elif yesterday < day_before:
            # SELL ALL
            if shares > 0:
                cash = shares * today
                shares = 0

        # Valeur du portefeuille
        total_value = cash + shares * today
        portfolio_values.append(total_value)

    return numpy.array(portfolio_values)




ticker = Ticker(ticker=action)
result = ticker.yahoo_api_price(range='5y', dataGranularity='1d')
price = result["close"].to_numpy()


portfolio_strat = simulate_trading_lagged(price)

buy_hold = price / price[0] * 2000

plt.plot(portfolio_strat, label="Strategy (lagged)", color="green")
plt.plot(buy_hold, label="Buy & Hold", color="blue")
plt.title(action)
plt.legend()
plt.show()

