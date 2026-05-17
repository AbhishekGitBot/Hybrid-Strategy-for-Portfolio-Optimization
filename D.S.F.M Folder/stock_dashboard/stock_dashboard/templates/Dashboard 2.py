import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf

# Streamlit setup
st.set_page_config(page_title="Nifty 50 Financial Dashboard (2023)", layout="wide")
st.title("Nifty 50 Financial Dashboard (2023)")

# Define stock tickers
tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", 
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "IOC.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", 
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "ZEEL.NS"
]

# Fetch data for 2023
@st.cache_data(ttl=86400)
def load_data():
    data = yf.download(tickers, start="2023-01-01", end="2023-12-31")['Adj Close']
    return data

@st.cache_data(ttl=86400)
def load_volume_and_ratios():
    volumes = {}
    pe_ratios = {}
    sectors = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            volumes[ticker] = stock.history(period='1y')['Volume']
            pe_ratios[ticker] = stock.info.get('forwardPE', None)
            sectors[ticker] = stock.info.get('sector', None)
        except Exception as e:
            st.warning(f"Data for {ticker} could not be fetched. Error: {e}")
    return volumes, pe_ratios, sectors

data = load_data()
volumes, pe_ratios, sectors = load_volume_and_ratios()

# Sidebar for sector selection
st.sidebar.title("Select Stocks by Sector")
unique_sectors = list(set(sectors.values()))
selected_sector = st.sidebar.selectbox("Choose Sector", ["All Sectors"] + unique_sectors)

# Filter stocks based on sector or individual selection
if selected_sector == "All Sectors":
    sector_stocks = tickers
else:
    sector_stocks = [ticker for ticker, sector in sectors.items() if sector == selected_sector]

selected_stocks = st.sidebar.multiselect("Select Specific Stocks", options=sector_stocks, default=sector_stocks)
filtered_data = data[selected_stocks] if selected_stocks else data[sector_stocks]

# Left sidebar for advanced visualizations
with st.sidebar:
    st.title("Advanced Visualizations")
    show_correlation_matrix = st.checkbox("Show Correlation Matrix Heatmap")
    show_distance_matrix = st.checkbox("Show Distance Matrix Heatmap")
    show_mds = st.checkbox("Show Multidimensional Scaling (MDS)")
    show_kmeans = st.checkbox("Show KMeans Clustering")

# Right sidebar for Nifty 50 closing prices
with st.sidebar.container():
    st.title("Nifty 50 Closing Prices")
    closing_prices = data.iloc[-1]
    for ticker, price in closing_prices.items():
        st.write(f"{ticker}: ₹{price:.2f}")

# Recommendation function based on P/E Ratio
def get_recommendation(pe_ratio):
    if pe_ratio is None:
        return "N/A"
    elif pe_ratio < 15:
        return "Buy"
    elif 15 <= pe_ratio <= 20:
        return "Hold"
    else:
        return "Sell"

# Main dashboard content
st.write("## Portfolio Analytics")
fig = go.Figure()
for ticker in selected_stocks:
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[ticker], mode='lines', name=ticker))
fig.update_layout(title="Time Series of Selected Nifty 50 Stocks", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# Distribution Analysis (Fat-Tail Analysis)
st.write("## Distribution Analysis of Log Returns")
filtered_log_returns = np.log(filtered_data / filtered_data.shift(1)).dropna()
for ticker in selected_stocks:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_log_returns[ticker].dropna(), kde=True, stat="density", ax=ax, label="Empirical")
    sns.kdeplot(np.random.normal(loc=filtered_log_returns[ticker].mean(), 
                                 scale=filtered_log_returns[ticker].std(), 
                                 size=1000), 
                ax=ax, color="red", linestyle="--", label="Normal Dist")
    ax.set_title(f"Distribution of Log Returns for {ticker}")
    ax.legend()
    st.pyplot(fig)

# Autocorrelation Analysis
st.write("## Autocorrelation Analysis of Stock Prices")
for ticker in selected_stocks:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(filtered_data[ticker].dropna(), ax=ax, title=f"Autocorrelation of {ticker} Stock Prices")
    st.pyplot(fig)

# Volume Data for selected stocks
st.write("## Volume Data")
volume_data = pd.DataFrame({ticker: volumes.get(ticker) for ticker in selected_stocks}).dropna()
if not volume_data.empty:
    fig = go.Figure()
    for ticker in selected_stocks:
        fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[ticker], mode='lines', name=ticker))
    fig.update_layout(title="Volume Data of Selected Stocks", xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(fig)

# P/E Ratios and Sectors for selected stocks
st.write("## P/E Ratios and Sectors")
pe_sector_data = pd.DataFrame({
    "Ticker": selected_stocks,
    "P/E Ratio": [pe_ratios.get(ticker) for ticker in selected_stocks],
    "Sector": [sectors.get(ticker) for ticker in selected_stocks]
})
pe_sector_data['Recommendation'] = pe_sector_data['P/E Ratio'].apply(get_recommendation)
st.write(pe_sector_data)

# Correlation Matrix Heatmap
if show_correlation_matrix:
    st.write("## Correlation Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(filtered_log_returns.corr(), cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)

# Distance Matrix Heatmap
if show_distance_matrix:
    st.write("## Distance Matrix Heatmap")
    distance_matrix = squareform(pdist(filtered_log_returns.T, 'euclidean'))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(distance_matrix, cmap="viridis", xticklabels=selected_stocks, yticklabels=selected_stocks, ax=ax)
    st.pyplot(fig)

# Multidimensional Scaling (MDS) and KMeans Clustering
if show_mds or show_kmeans:
    if len(selected_stocks) >= 2:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_results = mds.fit_transform(distance_matrix)

        # MDS Visualization
        if show_mds:
            st.write("## Multidimensional Scaling (MDS)")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(mds_results[:, 0], mds_results[:, 1], c='orange')
            for i, ticker in enumerate(selected_stocks):
                ax.text(mds_results[i, 0], mds_results[i, 1], ticker, fontsize=9)
            st.pyplot(fig)

        # KMeans Clustering
        if show_kmeans:
            st.write("## KMeans Clustering")
            optimal_k = st.sidebar.slider("Select optimal number of clusters", 2, min(10, len(selected_stocks)), 4)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(filtered_log_returns.T)
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(mds_results[:, 0], mds_results[:, 1], c=clusters, cmap="viridis")
            for i, ticker in enumerate(selected_stocks):
                ax.text(mds_results[i, 0], mds_results[i, 1], ticker, fontsize=9)
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

# Download options for data and figures
st.write("### Download Options")
csv_data = filtered_data.to_csv().encode('utf-8')
st.download_button(label="Download Closing Prices Data", data=csv_data, file_name="closing_prices.csv", mime='text/csv')

log_returns_csv = filtered_log_returns.to_csv().encode('utf-8')
st.download_button(label="Download Log Returns Data", data=log_returns_csv, file_name="log_returns.csv", mime='text/csv')

volume_data_csv = volume_data.to_csv().encode('utf-8')
st.download_button(label="Download Volume Data", data=volume_data_csv, file_name="volume_data.csv", mime='text/csv')