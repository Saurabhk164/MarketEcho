# Stock Price Prediction and Sentiment Analysis

## **Introduction**
Predicting stock price movements is a complex yet fascinating task. With the rise of social media, public sentiment has emerged as a crucial factor in influencing stock prices. This project combines traditional financial analysis with sentiment analysis to predict stock movements more effectively.

The core innovation of this project is the **Sentiment-Adjusted Moving Average (SAMA)**, which integrates public sentiment data into the calculation of stock price trends. Using tools like **VADER sentiment analysis**, **customized moving averages**, and a **machine learning model**, this project explores how sentiment-driven strategies can improve the accuracy of stock predictions.

### **Key Features**
1. **Data Collection and Processing**:
   - Extracted stock-related tweets and analyzed them for sentiment scores using VADER.
   - Mapped sentiment data to historical stock prices from Yahoo Finance.

2. **Sentiment-Adjusted Moving Averages**:
   - Developed **SAMA(7)** (short-term) and **SAMA(20)** (medium-term) moving averages that account for sentiment.

3. **Predictive Modeling**:
   - Built a machine learning model to predict stock movements based on sentiment-adjusted metrics.

4. **Interactive Dashboard**:
   - Visualized stock prices, sentiment trends, and predictive outputs using Streamlit.

---

## **How to Run the Streamlit Application**

### **Prerequisites**
1. **Install Python**:
   Ensure Python 3.7 or higher is installed on your machine.

2. **Install Required Libraries**:
   Run the following command to install all required libraries:
   ```bash
   pip install -r requirements.txt

## **Steps to Run the Application**

### **1. Clone the Repository**
   Clone the project repository to your local machine:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
