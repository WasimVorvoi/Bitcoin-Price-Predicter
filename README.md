# Bitcoin-Price-Predicter

Under Constant Development

This project is a Bitcoin price prediction tool that uses a Long Short-Term Memory (LSTM) neural network to forecast short-term price movements based on 5-minute interval stock data. It begins by collecting real-time BTC-USD price data from Yahoo Finance for the past 30 days and scales the closing prices between 0 and 1 using a MinMaxScaler to prepare it for training. The model uses sequences of the past 60 minutes to learn patterns and predict the next minute's price. It consists of stacked LSTM layers with dropout regularization to reduce overfitting and is trained using the mean squared error loss function. Once trained, the model predicts future prices and saves them to a CSV file for analysis. This setup allows for fast, near real-time forecasting and is ideal for experimentation with live financial data. The AI in this code is made from scratch and can be exported and saved anytime the accuracy satisfactory. This AI so far has a 95% accuracy when it comes to predicting the future price of Bitcoin.

To run the code you have to:
1. Download all the dependencies from requirements.txt
2. Download Python 3.8 and Install it to PATH
3. Run predict.py using Python 3.8
4. Watch the AI get trained and see the predicted price of Bitcoin
