import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import math
import statistics
import nltk
nltk.download('vader_lexicon')
import streamlit_option_menu
from streamlit_option_menu import option_menu
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib.dates import DateFormatter
from scipy.stats import f
import warnings
warnings.filterwarnings("ignore")



# Title and description
st.title("MarketEcho")
st.write("""
Analyze stock sentiments from social media.
""")
with st.sidebar:
    selected = option_menu(
    menu_title = "Main Menu",
    options = ["Preview Data","Sentiment Analysis","Predictions","About"],
    icons = ["gear","activity","alt","archive"],
    menu_icon = "cast",
    default_index = 0,
    #orientation = "horizontal",
)
# Load Data
all_tweets = pd.read_csv(r"D:\Projects\MarketEcho\stock_tweets.csv")
all_stock = pd.read_csv(r"D:\Projects\MarketEcho\stock_yfinance_data.csv")

if selected == "Preview Data":
    st.write("Tweet Data Sample:")
    st.write(all_tweets.head())

    st.write("Stock Data Sample:")
    st.write(all_stock.head())

if selected == "Sentiment Analysis":
    companies = ['TSLA', 'MSFT', 'META' ,'AMZN', 'GOOG', 'AAPL', 'NFLX']
    # Company Selector
    selected_company = st.selectbox("Select a company:", companies)
    # Filter tweets for a specific stock
    stk_name = selected_company
    appl_tweets = all_tweets[all_tweets["Stock Name"] == stk_name]

    # Sentiment Analysis
    st.header("Sentiment Analysis")
    appl_sent = appl_tweets.copy()
    appl_sent["sent_score"] = ''
    appl_sent["pos_score"] = ''
    appl_sent["neu_score"] = ''
    appl_sent["neg_score"] = ''

    sent_analyze = SentimentIntensityAnalyzer()

    for ind, row in appl_sent.T.items():
        sentence_i = unicodedata.normalize("NFKD", appl_sent.loc[ind, "Tweet"])
        sent_sent = sent_analyze.polarity_scores(sentence_i)
        appl_sent.at[ind, "sent_score"] = sent_sent["compound"]
        appl_sent.at[ind, "pos_score"] = sent_sent["pos"]
        appl_sent.at[ind, "neu_score"] = sent_sent["neu"]
        appl_sent.at[ind, "neg_score"] = sent_sent["neg"]

    # Process Date
    appl_sent["Date"] = pd.to_datetime(appl_sent["Date"]).dt.date
    sama_df = appl_sent.groupby("Date")["sent_score"].mean().reset_index()

    # Join sentiment with stock data
    appl_stock = all_stock[all_stock["Stock Name"] == stk_name]
    appl_stock["Date"] = pd.to_datetime(appl_stock["Date"]).dt.date
    sama_df = pd.merge(appl_stock, sama_df, on="Date", how="left")

    # Display data
    st.write("Merged Data Sample:")
    st.write(sama_df.head())

    # Plot Sentiment and Stock Price
    st.header("Visualize Sentiment and Stock Price")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(sama_df["Date"], sama_df["Close"], label="Stock Price")
    ax.set(xlabel="Date", ylabel="USD", title=f"{stk_name} Stock Price")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(sama_df["Date"], sama_df["sent_score"], label="Sentiment Score", color="orange")
    ax.set(xlabel="Date", ylabel="Sentiment Score", title=f"{stk_name}'s Public Sentiment")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    st.pyplot(fig)

if selected == "Predictions":
    
    tab1, tab2, tab3 = st.tabs(["SAMA", "F-test", "Predictions"])
    
    with tab1:
        companies = ['TSLA', 'MSFT', 'META' ,'AMZN', 'GOOG', 'AAPL', 'NFLX']
        # Company Selector
        selected_company = st.selectbox("Select a company:", companies)
        # Filter tweets for a specific stock
        stk_name = selected_company
        appl_tweets = all_tweets[all_tweets["Stock Name"] == stk_name]

        # Sentiment Analysis
        appl_sent = appl_tweets.copy()
        appl_sent["sent_score"] = ''
        appl_sent["pos_score"] = ''
        appl_sent["neu_score"] = ''
        appl_sent["neg_score"] = ''

        sent_analyze = SentimentIntensityAnalyzer()

        for ind, row in appl_sent.T.items():
            sentence_i = unicodedata.normalize("NFKD", appl_sent.loc[ind, "Tweet"])
            sent_sent = sent_analyze.polarity_scores(sentence_i)
            appl_sent.at[ind, "sent_score"] = sent_sent["compound"]
            appl_sent.at[ind, "pos_score"] = sent_sent["pos"]
            appl_sent.at[ind, "neu_score"] = sent_sent["neu"]
            appl_sent.at[ind, "neg_score"] = sent_sent["neg"]
        # Process Date
        appl_sent["Date"] = pd.to_datetime(appl_sent["Date"]).dt.date
        sama_df = appl_sent.groupby("Date")["sent_score"].mean().reset_index()

        # Join sentiment with stock data
        appl_stock = all_stock[all_stock["Stock Name"] == stk_name]
        appl_stock["Date"] = pd.to_datetime(appl_stock["Date"]).dt.date
        sama_df = pd.merge(appl_stock, sama_df, on="Date", how="left")
        # Compute Sentiment-Aware Moving Averages (SAMA)
        st.header("Sentiment-Aware Moving Averages (SAMA)")
        def SAMA(df, ma_days=5):
            def weight_multiplier(close, sent_score):
                len_close = len(close)
                interval = statistics.variance(close) if len_close > 1 else math.sqrt(close[0])
                max_var = interval
                weighted = sum(
                    close[i] + (2 * sent_score[i] * max_var if sent_score[i] < 0 else sent_score[i] * max_var)
                    for i in range(len_close)
                )
                return weighted / len_close

            samas = [df.loc[0, "Close"]]
            for i in range(1, len(df)):
                mini_df = df.iloc[max(0, i - ma_days + 1):i + 1]
                samas.append(weight_multiplier(mini_df["Close"].tolist(), mini_df["sent_score"].tolist()))
            return samas

        fin_df = sama_df[["Date", "Close", "sent_score"]].copy()
        fin_df["SAMA(7)"] = SAMA(fin_df, ma_days=7)
        fin_df["SAMA(20)"] = SAMA(fin_df, ma_days=20)

        # Plot Comparisons
        st.subheader("Compare Moving Averages")
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(fin_df["Date"], fin_df["SAMA(7)"], label="SAMA(7)", linestyle="--", color="r")
        ax.plot(fin_df["Date"], fin_df["Close"], label="Original", color="b")
        ax.plot(fin_df["Date"], fin_df["SAMA(20)"], label="SAMA(20)", linestyle="--", color="g")
        ax.set(xlabel="Date", ylabel="USD", title=f"{stk_name} Stock Price with SAMA")
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        plt.legend()
        st.pyplot(fig)

        # Traditional Moving Averages
        fin_df["MA7"] = fin_df["Close"].rolling(window=7).mean()
        fin_df["MA20"] = fin_df["Close"].rolling(window=20).mean()

        st.subheader("Compare SAMA and MA")
        for i in ("7", "20"):
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(fin_df["Date"], fin_df[f"SAMA({i})"], label=f"SAMA({i})", linestyle="-.", color="r")
            ax.plot(fin_df["Date"], fin_df["Close"], label="Original", color="b")
            ax.plot(fin_df["Date"], fin_df[f"MA{i}"], label=f"MA{i}", linestyle="--", color="g")
            ax.set(xlabel="Date", ylabel="USD", title=f"{stk_name} Stock Price Comparison")
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))
            plt.legend()
            st.pyplot(fig)
            
        st.write("* SAMA(7) and SAMA(20): This moving average adjusts the stock price with sentiment data derived from social media (tweets) or other sources. Uses the last 7 and 20 days of stock price")
        st.write("* MA(7) & MA(20): This moving average only considers the actual stock price.Uses the last 7 and 20 days of stock price")
        
    with tab2:
        # F-Test for Variance
        st.header("F-Test Results")
        var_sama7 = np.var(fin_df["SAMA(7)"], ddof=1)
        var_ma7 = np.var(fin_df["MA7"], ddof=1)
        f_value_7 = var_sama7 / var_ma7
        df = len(fin_df) - 1
        p_value = f.cdf(f_value_7, df, df)
        st.write(f"F-Value: {f_value_7}")
        
        if f_value_7 >= 1.5:
            st.write("* Sentiment-based Stock Price approach has more fluctuation than the Original Stock Price.")
            st.write("* People's Sentiment have less impact on this Stock price")
        else:
            st.write("* Sentiment-based Stock Price approach has less fluctuation than the Original Stock Price")
            st.write("*  People's Sentiment has an impact on this Stock price")
        
        st.write(f"P-Value: {p_value}")
        
        if p_value < 0.5:
            st.write("* Reject Null Hypothesis: The variance of the sentiment-based stock price is significantly less")
        else:
            st.write("* Fail to Reject Null Hypothesis: The variance of the sentiment-based stock price is significantly more")
            
        st.header("What is F-test?")
        st.write("* F-Value: It tells you how much more variable the sentiment-adjusted moving average is compared to the traditional moving average. A larger F-value suggests that sentiment data is adding significant variance (noise) to the stock price predictions, indicating that the sentiment-based approach has more fluctuation than the traditional price-based method.")
        st.write("* P-Value: If the p-value is low (e.g., less than 0.05), it means that the variances of SAMA(7) and MA7 are significantly different. This implies that the sentiment-adjusted moving average (SAMA) and the traditional moving average are not simply different by random chance. A high p-value (e.g., greater than 0.05) would suggest that there's no significant difference in the variances.")

        

        
    with tab3:    
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

        # Load the processed data
        fin_df = sama_df[["sent_score", "Open", "High", "Low", "Close"]].copy()

        # Create the target variable (binary classification)
        fin_df["Target"] = (fin_df["Close"].shift(-1) > fin_df["Close"]).astype(int)

        # Drop rows with NaN (due to shifting or moving averages)
        fin_df.dropna(inplace=True)

        # Define features and target
        X = fin_df[["sent_score", "Open", "High", "Low", "Close"]]
        y = fin_df["Target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model 1: Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)

        # Model 2: Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        # Evaluate both models
        def evaluate_model(y_true, y_pred, model_name):
            st.write(f"### {model_name} Evaluation")
            st.write("Accuracy:", accuracy_score(y_true, y_pred))
            st.write("Precision:", precision_score(y_true, y_pred))
            st.write("Recall:", recall_score(y_true, y_pred))
            st.write("F1 Score:", f1_score(y_true, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_true, y_pred))

        # Streamlit Integration
        import streamlit as st
        st.title("Stock Movement Prediction")
        
        if accuracy_score(y_test, y_pred_lr) > accuracy_score(y_test, y_pred_rf):
            st.subheader("The Logistic Regression model is better than the Random Forest model.")
        else:
            st.subheader("The Random Forest model is better than the Logistic Regression model.")

        st.header("Logistic Regression Results")
        evaluate_model(y_test, y_pred_lr, "Logistic Regression")

        st.header("Random Forest Results")
        evaluate_model(y_test, y_pred_rf, "Random Forest")

        # Display feature importance for Random Forest
        import matplotlib.pyplot as plt
        feature_importance = rf_model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(X.columns, feature_importance, color="skyblue")
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        st.pyplot(plt)
        
if selected == "About":
    st.title("About")
    
    st.header("Introduction: Predicting Stock Price Movements Using Sentiment-Adjusted Metrics")
    st.write("Stock markets are inherently influenced by numerous factors, including economic indicators, company performance, and public sentiment. Among these, public sentiment—reflected in tweets, news, and discussions—plays a crucial role in short-term market movements. Recognizing this, our project integrates sentiment analysis with traditional stock price analysis to create innovative metrics that better capture market dynamics and predict stock movements.")
    
    st.header("Motivation")
    st.write("The motivation for this project stems from:")
    st.write("* Gaps in Traditional Analysis: Conventional stock price analysis methods, such as moving averages, often overlook the psychological aspect of market behavior influenced by public sentiment.")
    st.write("* Rise of Social Media: Platforms like Twitter have become hotbeds for investor opinions, with their sentiment significantly impacting stock prices, especially for high-profile companies like Apple (AAPL).")
    st.write("* Improved Predictive Accuracy: By integrating sentiment into price analysis, we aimed to enhance prediction models, offering traders and investors a more nuanced understanding of market trends.")

    st.header("Key Features of the Project")
    st.write("* Sentiment Analysis: Collected tweets about the stock (e.g., AAPL). Processed tweets using the VADER Sentiment Analyzer to derive scores for positivity, neutrality, negativity, and an overall compound sentiment score.")
    st.write("* Sentiment-Adjusted Metrics: Developed custom moving averages, SAMA(7) and SAMA(20), that dynamically adjust based on sentiment scores to capture sentiment-driven market trends. Compared SAMA metrics with traditional moving averages (MA7 and MA20) to evaluate the impact of sentiment on price trends.")
    st.write("* Predictive Modeling: Built machine learning models to predict stock price movements (up or down) based on historical prices, sentiment scores, and sentiment-adjusted metrics. Evaluated model performance using metrics such as accuracy, precision, recall, and F1 score.")
