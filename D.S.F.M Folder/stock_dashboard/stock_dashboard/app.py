from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

def calculate_optimal_clusters(data, max_clusters=10):
    """Calculate optimal number of clusters using elbow method"""
    distortions = []
    K = range(1, max_clusters+1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    differences = np.diff(distortions)
    elbow_point = np.argmin(differences) + 1
    return elbow_point

def calculate_rolling_correlation_matrix(returns, window=30):
    """Calculate correlation matrix based on rolling windows"""
    corr_matrix = returns.rolling(window=window).corr()
    return corr_matrix.fillna(0)  # Fill NaN values with 0

@app.route('/')
def home():
    try:
        # Get Nifty 50 data
        nifty = yf.download('^NSEI', start='2023-01-01', end='2024-10-31')
        
        # Calculate log returns
        nifty['Log_Return'] = np.log(nifty['Close'] / nifty['Close'].shift(1))
        nifty = nifty.dropna()
        
        # Create figures list
        figures = []
        
        # 1. Time Series Plot
        fig1 = make_subplots(rows=2, cols=1,
                           subplot_titles=('Nifty 50 Price', 'Log Returns'),
                           vertical_spacing=0.2,
                           row_heights=[0.7, 0.3])

        fig1.add_trace(
            go.Candlestick(
                x=nifty.index,
                open=nifty['Open'],
                high=nifty['High'],
                low=nifty['Low'],
                close=nifty['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Scatter(
                x=nifty.index,
                y=nifty['Log_Return'],
                mode='lines',
                name='Log Returns'
            ),
            row=2, col=1
        )

        fig1.update_layout(
            title_text="Nifty 50 Analysis",
            template='plotly_dark',
            height=800
        )

        # 2. Correlation Matrix
        correlation = nifty[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        fig2 = go.Figure(data=go.Heatmap(
            z=correlation,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='Viridis'
        ))
        fig2.update_layout(
            title='Correlation Matrix',
            template='plotly_dark',
            height=600
        )

        # 3. MDS Plot
        scaled_data = StandardScaler().fit_transform(nifty[['Close', 'Volume', 'Log_Return']].fillna(0))
        mds = MDS(n_components=2, random_state=42)
        mds_coords = mds.fit_transform(scaled_data)

        fig3 = go.Figure(data=go.Scatter(
            x=mds_coords[:, 0],
            y=mds_coords[:, 1],
            mode='markers',
            marker=dict(
                color=nifty['Close'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig3.update_layout(
            title='MDS Visualization',
            template='plotly_dark',
            height=600
        )

        # 4. K-Means Clustering
        optimal_clusters = calculate_optimal_clusters(scaled_data)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        fig4 = go.Figure(data=go.Scatter(
            x=mds_coords[:, 0],
            y=mds_coords[:, 1],
            mode='markers',
            marker=dict(
                color=clusters,
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig4.update_layout(
            title=f'K-Means Clustering (k={optimal_clusters})',
            template='plotly_dark',
            height=600
        )

        # Convert figures to JSON
        plot1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        plot2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        plot3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        plot4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

        # Calculate statistics
        stats = {
            'mean_return': float(nifty['Log_Return'].mean()),
            'std_return': float(nifty['Log_Return'].std()),
            'min_return': float(nifty['Log_Return'].min()),
            'max_return': float(nifty['Log_Return'].max()),
            'optimal_clusters': int(optimal_clusters)
        }

        return render_template('dashboard.html',
                             plot1=plot1,
                             plot2=plot2,
                             plot3=plot3,
                             plot4=plot4,
                             stats=stats)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('dashboard.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)