from stockdex import Ticker
import numpy as np
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
    "Berkshire Hathaway Inc. (Class B)": "BRK-B",
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
action = us_companies["Costco Wholesale Corporation"]


ticker = Ticker(ticker=action)
result = ticker.yahoo_api_price(range='5y', dataGranularity='1d')
price = result["close"].to_numpy()



price_fft = np.fft.fftshift(np.fft.fft(price))
freq = np.fft.fftshift(np.fft.fftfreq(len(price)))*len(price)
magnitude = np.abs(price_fft)
phase = np.angle(price_fft)

idx = np.argsort(freq)




plt.plot(price)
plt.show()


plt.figure()
plt.scatter(freq[:], magnitude[:], label="FFT", color="green")
plt.title(action)
plt.legend()


plt.show()