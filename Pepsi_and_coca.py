from stockdex import Ticker
import numpy
import matplotlib.pyplot as plt


ticker = Ticker(ticker="PEP")
result = ticker.yahoo_api_price(range='5y', dataGranularity='1d')
price_pepsi = result["close"].to_numpy()

ticker = Ticker(ticker="KO")
result = ticker.yahoo_api_price(range='5y', dataGranularity='1d')
price_coca = result["close"].to_numpy()

ticker = Ticker(ticker="AMZN")
result = ticker.yahoo_api_price(range='5y', dataGranularity='1d')
price_amazon = result["close"].to_numpy()




fig, ax1 = plt.subplots()

# Axe gauche (Pepsi)
line1, = ax1.plot(price_pepsi, color="blue", label="Pepsi")
ax1.set_ylabel("Pepsi Price", color="black")
ax1.tick_params(axis='y', labelcolor="black")

# Axe droit (Coca)
ax2 = ax1.twinx()
line2, = ax2.plot(price_coca, color="red", label="Coca-Cola")
ax2.set_ylabel("Coca-Cola Price", color="black")
ax2.tick_params(axis='y', labelcolor="black")

# Combine les deux légendes
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left")

plt.title("Pepsi vs Coca-Cola Stock Prices")
plt.show()