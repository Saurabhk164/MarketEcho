# MarketEcho
Predicting Stock Price Movements Using Sentiment-Adjusted Metrics
Stock markets are inherently influenced by numerous factors, including economic indicators, company performance, and public sentiment. Among these, public sentiment—reflected in tweets, news, and discussions—plays a crucial role in short-term market movements. Recognizing this, our project integrates sentiment analysis with traditional stock price analysis to create innovative metrics that better capture market dynamics and predict stock movements.

Motivation for the Project
The motivation for this project stems from:

Gaps in Traditional Analysis: Conventional stock price analysis methods, such as moving averages, often overlook the psychological aspect of market behavior influenced by public sentiment.
Rise of Social Media: Platforms like Twitter have become hotbeds for investor opinions, with their sentiment significantly impacting stock prices, especially for high-profile companies like Apple (AAPL).
Improved Predictive Accuracy: By integrating sentiment into price analysis, we aimed to enhance prediction models, offering traders and investors a more nuanced understanding of market trends.
Key Features of the Project
Sentiment Analysis:

Collected tweets about the stock (e.g., AAPL).
Processed tweets using the VADER Sentiment Analyzer to derive scores for positivity, neutrality, negativity, and an overall compound sentiment score.
Sentiment-Adjusted Metrics:

Developed custom moving averages, SAMA(7) and SAMA(20), that dynamically adjust based on sentiment scores to capture sentiment-driven market trends.
Compared SAMA metrics with traditional moving averages (MA7 and MA20) to evaluate the impact of sentiment on price trends.
Predictive Modeling:

Built machine learning models to predict stock price movements (up or down) based on historical prices, sentiment scores, and sentiment-adjusted metrics.
Evaluated model performance using metrics such as accuracy, precision, recall, and F1 score.
Challenges Encountered and Solutions
Sentiment Analysis Data Availability:

Challenge: The stock tweets dataset needed cleaning and normalization for effective sentiment scoring.
Resolution: Leveraged Python’s unicodedata library to preprocess text and ensured accurate scoring using the VADER lexicon, which required downloading missing dependencies.
Incorporating Sentiment into Financial Metrics:

Challenge: Traditional moving averages did not account for sentiment dynamics, making it necessary to design a novel approach.
Resolution: Designed and implemented the SAMA function, which dynamically adjusts weights based on sentiment scores, providing a more refined view of stock trends.
Feature Engineering for Prediction:

Challenge: Identifying and preparing relevant features, such as sentiment scores, moving averages, and historical prices, for use in machine learning models.
Resolution: Created new features, ensured proper alignment of stock prices and sentiment data, and normalized inputs for better model performance.
Evaluating Model Performance:

Challenge: Balancing simplicity and accuracy while testing models like Logistic Regression and Random Forest.
Resolution: Used comprehensive metrics (accuracy, precision, recall, F1 score) and visualized feature importance to identify areas of improvement.

