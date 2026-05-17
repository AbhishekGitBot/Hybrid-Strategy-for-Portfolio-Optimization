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

# Custom CSS to improve layout
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .metric-card {
        border: 1px solid #e6e6e6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Define stock tickers (same as before)
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

# Cache functions (same as before)
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

# Load data
data = load_data()
volumes, pe_ratios, sectors = load_volume_and_ratios()

# Header section with key metrics
st.title("Nifty 50 Financial Dashboard (2023)")

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")
    
    # Sector selection
    unique_sectors = list(set(sectors.values()))
    selected_sector = st.selectbox("Choose Sector", ["All Sectors"] + unique_sectors)
    
    # Filter stocks based on sector
    if selected_sector == "All Sectors":
        sector_stocks = tickers
    else:
        sector_stocks = [ticker for ticker, sector in sectors.items() if sector == selected_sector]
    
    selected_stocks = st.multiselect("Select Specific Stocks", options=sector_stocks, default=sector_stocks)
    
    # Advanced visualization options
    st.title("Advanced Visualizations")
    show_correlation_matrix = st.checkbox("Show Correlation Matrix Heatmap")
    show_distance_matrix = st.checkbox("Show Distance Matrix Heatmap")
    show_mds = st.checkbox("Show Multidimensional Scaling (MDS)")
    show_kmeans = st.checkbox("Show KMeans Clustering")
    if show_kmeans:
        optimal_k = st.slider("Select optimal number of clusters", 2, min(10, len(selected_stocks)), 4)

# Filter data based on selection
filtered_data = data[selected_stocks] if selected_stocks else data[sector_stocks]
filtered_log_returns = np.log(filtered_data / filtered_data.shift(1)).dropna()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Price Analysis", 
    "Volume Analysis", 
    "Distribution Analysis",
    "Advanced Analytics",
    "Download Center"
])

# Tab 1: Overview & Price Analysis
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Price Trends")
        fig = go.Figure()
        for ticker in selected_stocks:
            fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data[ticker], mode='lines', name=ticker))
        fig.update_layout(height=500, title="Time Series of Selected Stocks")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Latest Stock Prices")
        closing_prices = filtered_data.iloc[-1]
        for ticker, price in closing_prices.items():
            st.markdown(f"""
            <div class="metric-card">
                <h4>{ticker}</h4>
                <p>₹{price:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Volume Analysis
with tab2:
    st.subheader("Trading Volume Analysis")
    volume_data = pd.DataFrame({ticker: volumes.get(ticker) for ticker in selected_stocks}).dropna()
    if not volume_data.empty:
        fig = go.Figure()
        for ticker in selected_stocks:
            fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[ticker], mode='lines', name=ticker))
        fig.update_layout(height=600, title="Volume Data of Selected Stocks")
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Distribution Analysis
with tab3:
    st.subheader("Distribution Analysis of Log Returns")
    cols = st.columns(2)
    for idx, ticker in enumerate(selected_stocks):
        with cols[idx % 2]:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_log_returns[ticker].dropna(), kde=True, stat="density", ax=ax, label="Empirical")
            sns.kdeplot(np.random.normal(loc=filtered_log_returns[ticker].mean(), 
                                     scale=filtered_log_returns[ticker].std(), 
                                     size=1000), 
                    ax=ax, color="red", linestyle="--", label="Normal Dist")
            ax.set_title(f"Distribution of Log Returns for {ticker}")
            ax.legend()
            st.pyplot(fig)
            
            # Add autocorrelation plot
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_acf(filtered_data[ticker].dropna(), ax=ax, title=f"Autocorrelation of {ticker}")
            st.pyplot(fig)

# Tab 4: Advanced Analytics
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        # P/E Ratios and Sectors
        st.subheader("Fundamental Analysis")
        pe_sector_data = pd.DataFrame({
            "Ticker": selected_stocks,
            "P/E Ratio": [pe_ratios.get(ticker) for ticker in selected_stocks],
            "Sector": [sectors.get(ticker) for ticker in selected_stocks]
        })
        pe_sector_data['Recommendation'] = pe_sector_data['P/E Ratio'].apply(
            lambda x: "Buy" if x and x < 15 else "Hold" if x and 15 <= x <= 20 else "Sell" if x else "N/A"
        )
        st.dataframe(pe_sector_data)
    
    with col2:
        # Correlation Matrix
        if show_correlation_matrix:
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(filtered_log_returns.corr(), cmap="coolwarm", annot=True, ax=ax)
            st.pyplot(fig)
    
    # Distance Matrix and Clustering
    if show_distance_matrix or show_mds or show_kmeans:
        st.subheader("Advanced Statistical Analysis")
        distance_matrix = squareform(pdist(filtered_log_returns.T, 'euclidean'))
        
        if show_distance_matrix:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(distance_matrix, cmap="viridis", xticklabels=selected_stocks, 
                       yticklabels=selected_stocks, ax=ax)
            st.pyplot(fig)
        
        if len(selected_stocks) >= 2:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
            mds_results = mds.fit_transform(distance_matrix)
            
            if show_mds:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(mds_results[:, 0], mds_results[:, 1], c='orange')
                for i, ticker in enumerate(selected_stocks):
                    ax.text(mds_results[i, 0], mds_results[i, 1], ticker, fontsize=9)
                st.pyplot(fig)
            
            if show_kmeans:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(filtered_log_returns.T)
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(mds_results[:, 0], mds_results[:, 1], c=clusters, cmap="viridis")
                for i, ticker in enumerate(selected_stocks):
                    ax.text(mds_results[i, 0], mds_results[i, 1], ticker, fontsize=9)
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                st.pyplot(fig)

# Tab 5: Download Center
with tab5:
    st.subheader("Download Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = filtered_data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Closing Prices",
            data=csv_data,
            file_name="closing_prices.csv",
            mime='text/csv'
        )
    
    with col2:
        log_returns_csv = filtered_log_returns.to_csv().encode('utf-8')
        st.download_button(
            label="Download Log Returns",
            data=log_returns_csv,
            file_name="log_returns.csv",
            mime='text/csv'
        )
    
    with col3:
        volume_data_csv = volume_data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Volume Data",
            data=volume_data_csv,
            file_name="volume_data.csv",
            mime='text/csv'
        )