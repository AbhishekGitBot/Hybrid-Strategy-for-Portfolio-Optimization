from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)

# Define Nifty 50 companies (symbol: name)
NIFTY_50_COMPANIES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "HCLTECH.NS": "HCL Technologies",
    "MARUTI.NS": "Maruti Suzuki",
    "ASIANPAINT.NS": "Asian Paints",
    "INDUSINDBK.NS": "IndusInd Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
    "NESTLEIND.NS": "Nestle India",
    "JSWSTEEL.NS": "JSW Steel",
    "POWERGRID.NS": "Power Grid Corporation",
    "NTPC.NS": "NTPC Limited",
    "M&M.NS": "Mahindra & Mahindra",
    "TATASTEEL.NS": "Tata Steel",
    "ADANIGREEN.NS": "Adani Green Energy",
    "TATACONSUM.NS": "Tata Consumer Products",
    "ADANIPORTS.NS": "Adani Ports",
    "ONGC.NS": "Oil & Natural Gas Corporation",
    "DIVISLAB.NS": "Divi's Laboratories",
    "COALINDIA.NS": "Coal India",
    "BPCL.NS": "Bharat Petroleum",
    "GRASIM.NS": "Grasim Industries",
    "BRITANNIA.NS": "Britannia Industries",
    "TECHM.NS": "Tech Mahindra",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "CIPLA.NS": "Cipla",
    "SBILIFE.NS": "SBI Life Insurance",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "TATAMOTORS.NS": "Tata Motors",
    "DABUR.NS": "Dabur India",
    "VEDL.NS": "Vedanta",
    "UPL.NS": "UPL Ltd",
    "EICHERMOT.NS": "Eicher Motors",
    "SHREECEM.NS": "Shree Cement",
    "ADANIENT.NS": "Adani Enterprises"
}


def fetch_stock_data(symbol):
    """Fetch stock data for 2023 and calculate log returns."""
    df = yf.download(symbol, start="2023-01-01", end="2023-12-31")
    df['Log Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    return df

def plot_to_base64(fig):
    """Convert a Matplotlib plot to a base64 encoded PNG image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return image_base64

@app.route("/", methods=["GET", "POST"])
def index():
    selected_symbol = request.form.get("symbol", "RELIANCE.NS")
    df = fetch_stock_data(selected_symbol)

    # Time Series Plot
    fig1, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Adj Close'], color='cyan')
    ax.set_title(f"{NIFTY_50_COMPANIES[selected_symbol]} Adjusted Close Prices (2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    time_series_img = plot_to_base64(fig1)

    # Log Returns Plot
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Log Return'], color='orange')
    ax.set_title(f"{NIFTY_50_COMPANIES[selected_symbol]} Log Returns (2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    log_returns_img = plot_to_base64(fig2)

    # Correlation Matrix Heatmap
    returns_df = df[['Log Return']].dropna()
    corr_matrix = returns_df.corr()
    fig3, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    corr_matrix_img = plot_to_base64(fig3)

    # Distance Matrix Heatmap
    dist_matrix = pairwise_distances(returns_df, metric="euclidean")
    fig4, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(dist_matrix, cmap="viridis")
    ax.set_title("Distance Matrix Heatmap")
    distance_matrix_img = plot_to_base64(fig4)

    # Multidimensional Scaling (MDS)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(dist_matrix)
    fig5, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mds_coords[:, 0], mds_coords[:, 1], c='blue')
    ax.set_title("Multidimensional Scaling (MDS)")
    mds_img = plot_to_base64(fig5)

    # Elbow Method for Optimal Clusters
    inertia_values = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(returns_df)
        inertia_values.append(kmeans.inertia_)

    fig6, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inertia_values, marker="o", color="purple")
    ax.set_title("Elbow Method for Optimal Clusters")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    elbow_img = plot_to_base64(fig6)

    # KMeans Clustering Plot (using optimal number of clusters, e.g., k=3)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(returns_df)
    fig7, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(mds_coords[:, 0], mds_coords[:, 1], c=clusters, cmap="tab10")
    ax.set_title(f"KMeans Clustering (k={optimal_k})")
    kmeans_img = plot_to_base64(fig7)

    return render_template(
        "index.html",
        companies=NIFTY_50_COMPANIES,
        selected_symbol=selected_symbol,
        time_series_img=time_series_img,
        log_returns_img=log_returns_img,
        corr_matrix_img=corr_matrix_img,
        distance_matrix_img=distance_matrix_img,
        mds_img=mds_img,
        elbow_img=elbow_img,
        kmeans_img=kmeans_img
    )

if __name__ == "__main__":
    app.run(debug=True)
