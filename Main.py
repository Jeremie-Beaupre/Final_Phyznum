from stockdex import Ticker
import numpy
import matplotlib.pyplot as plt


ticker = Ticker(ticker="AMZN")
result = ticker.yahoo_api_price(range='1y', dataGranularity='1d')



price = result["close"].to_numpy()
print(price)


plt.plot(price)
plt.show()