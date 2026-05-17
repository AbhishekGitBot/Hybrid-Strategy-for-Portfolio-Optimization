#!/usr/bin/env python
# coding: utf-8

# ## 1. Download stock data from the Nikkei 225 index for 10 years based on the starting date mentioned in the DSFM groups: Nikkei 225 market

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the website to scrape
url = "https://indexes.nikkei.co.jp/en/nkave/index/component"

# Send a request to the webpage
response = requests.get(url)

# Parse the HTML content of the page using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Lists to store company codes, names, and sectors
company_codes = []
company_names = []
sectors = []

current_sector = None

# Loop through all h3 (sector headers) and tables (company data)
for element in soup.find_all(['h3', 'table']):
    if element.name == 'h3':  # Sector name
        current_sector = element.text.strip()  # Set the current sector
    elif element.name == 'table':  # Company table
        rows = element.find_all('tr')
        for row in rows[1:]:  # Skip header
            cols = row.find_all('td')
            if len(cols) >= 2:
                company_code = cols[0].text.strip()
                company_name = cols[1].text.strip()
                company_codes.append(company_code)
                company_names.append(company_name)
                sectors.append(current_sector)  # Assign the current sector to the company

# Create a pandas DataFrame
data = {
    "Company Code": company_codes,
    "Company Name": company_names,
    "Sector": sectors
}

df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = 'nikkei_companies_by_sectors.csv'
df.to_csv(output_file, index=False)

print(f"Data has been saved to {output_file}")


# In[ ]:





# ## 2. Construct one CSV file of all the stocks for close prices (similar to the attachment SP500_350.csv) and arrange them in sectors.

# In[2]:


import pandas as pd
import yfinance as yf

# Load the Excel sheet with company tickers
file_path = 'nikkei_companies_by_sectors.csv'
df = pd.read_csv(file_path)
df


# In[ ]:





# In[ ]:





# In[3]:


# Append '.T' to each company code in the 'Company Code' column
df['Company Code'] = df['Company Code'].astype(str) + '.T'

print(df)


# In[ ]:





# In[ ]:





# In[4]:


import pandas as pd
import yfinance as yf

# Assuming df is your DataFrame already defined
# Extract the company codes (tickers) and sectors
company_codes = df['Company Code'].unique()
sectors = df['Sector'].unique()

# Create a mapping for sectors to their short forms
sector_mapping = {sector: sector[:2].upper() for sector in sectors}  # Adjust this logic as needed

# Defining the start date and end date
start_date = "2010-04-01"
end_date = "2020-04-01"

stock_data = {}

# Downloading stock data
for ticker in company_codes:
    try:
        stock_df = yf.download(ticker, start=start_date, end=end_date)
        if not stock_df.empty:
            stock_data[ticker] = stock_df
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")


if stock_data:
    # Creating a DataFrame to store the closing prices
    closing_prices = pd.DataFrame()

    # Iterate through the stock data and extract the closing prices
    for ticker, stock_df in stock_data.items():
        # Find the corresponding sector for the ticker
        sector = df.loc[df['Company Code'] == ticker, 'Sector'].values[0]
        shortform = sector_mapping[sector]  # Get the short form of sectors
        column_name = f"{shortform}_{ticker}"

        # Add the closing prices to the DataFrame with the formatted column name
        closing_prices[column_name] = stock_df['Close']

    # Set the index (dates) as a column
    closing_prices['Date'] = stock_df.index
    closing_prices.set_index('Date', inplace=True)  # Set 'Date' as the index

    #save data in csv file
    output_file = 'closing_prices_dataset_nikkei.csv'
    closing_prices.to_csv(output_file)

    print(f"Closing prices dataset has been saved to {output_file}")
else:
    print("No stock data available to process.")


# In[ ]:





# In[ ]:





# In[5]:


closing_prices.head()


# In[ ]:





# ## 3. Plot all the closing time series in 3X3 subplots, i.e., 9 time series each, as discussed in class.

# In[6]:


import matplotlib.pyplot as plt
import pandas as pd
import math


# Number of stocks per subplot grid
group_size = 9

# Calculate the number of 3x3 grids needed
num_groups = math.ceil(len(closing_prices.columns) / group_size)

# Plotting the groups
for i in range(num_groups):
    group = closing_prices.iloc[:, i * group_size:(i + 1) * group_size]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot each stock's closing prices in the 3x3 grid
    for j, col in enumerate(group.columns):
        group[col].plot(ax=axes[j], title=col)
        axes[j].set_xlabel('Date')
        axes[j].set_ylabel('Closing Price')

    # Hide any unused subplots
    for k in range(j + 1, 9):
        axes[k].axis('off')
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[ ]:





# ## 4. Remove all the stocks with more than two consecutive days' NAN values.

# In[7]:


import pandas as pd

# Assuming 'closing_prices' is the DataFrame containing the closing prices
#finding the stocks with more than two consecutive days' NAN values.
def has_consecutive_nans(series, threshold=2):
    """
    Check if a series has more than a specified number of consecutive NaN values.
    """
    return series.isna().rolling(window=threshold + 1).sum().max() > threshold

# Identifying stocks to remove
stocks_to_remove = [col for col in closing_prices.columns if has_consecutive_nans(closing_prices[col])]

# list of stocks that will be removed
print(f"Stocks to be removed: {stocks_to_remove}")

# Remove the identified stocks from the DataFrame
cleaned_closing_prices = closing_prices.drop(columns=stocks_to_remove)

cleaned_closing_prices.interpolate(method = 'linear',inplace = True)

# cleaned DataFrame
print(cleaned_closing_prices.shape)

cleaned_closing_prices.to_csv('nikkei225_cleaned.csv')


# In[ ]:





# ## 5. Plot all the log return-time series [r(t)=lnP(t)-lnP(t-1)] in 3X3 subplots as discussed in class.

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Assuming 'cleaned_closing_prices' is the DataFrame after data cleaning

# Compute the log return series for each stock
log_returns = np.log(cleaned_closing_prices) - np.log(cleaned_closing_prices.shift(1))

# Drop the first row as it will contain NaN values due to the shift
log_returns = log_returns.dropna()
print(log_returns.head())

# Number of stocks per subplot grid
group_size = 9

# Calculate the number of 3x3 grids needed
num_groups = math.ceil(len(log_returns.columns) / group_size)

# Plotting the log return groups
for i in range(num_groups):
    # Select the next group of nine stocks
    group = log_returns.iloc[:, i * group_size:(i + 1) * group_size]

    # Create a 3x3 subplot grid
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    for j, col in enumerate(group.columns):
        group[col].plot(ax=axes[j], title=col)
        axes[j].set_xlabel('Date')
        axes[j].set_ylabel('Log Return')

    # Hide any unused subplots
    for k in range(j + 1, 9):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()


# In[ ]:





# ## 6. Remove all the stocks with |r(t)| > 0.8.

# In[9]:


#Remove all the stocks with |r(t)| > 0.8.

import numpy as np
import pandas as pd

# Assuming 'log_returns' is the DataFrame containing the log return series

# Set the threshold for extreme volatility
threshold = 0.8

# Identify stocks where any log return exceeds the threshold
volatile_stocks = log_returns.columns[log_returns.abs().max() > threshold]

# Display the list of volatile stocks to be removed
print(f"Stocks with extreme volatility to be removed: {volatile_stocks.tolist()}")

# Remove these stocks from the log_returns DataFrame
filtered_log_returns = log_returns.drop(columns=volatile_stocks)

# filtered DataFrame
print(filtered_log_returns.head())

# saved the filtered DataFrame to a new CSV file
filtered_log_returns.to_csv('nikkei_filtered_log_returns.csv')


# In[ ]:





# ## 7. Construct a correlation matrix for the full-time horizon of 10 years using log returns.

# In[10]:


#Construct a correlation matrix for the full-time horizon of 10 years using log returns.

import pandas as pd
import seaborn as sns

#'filtered_log_returns' is the DataFrame after filtering out volatile stocks

# Compute the correlation matrix for the log returns
correlation_matrix = filtered_log_returns.corr()

sns.heatmap(correlation_matrix,vmin=-1,vmax=1)

# Display the correlation matrix
print(correlation_matrix)

# saved the correlation matrix to a CSV file
correlation_matrix.to_csv('nikkei_log_returns__correlation_matrix.csv')


# In[ ]:





# ## Number of trading days in that decade.

# In[11]:


print(f'Total number of rows (trading days): {filtered_log_returns.shape[0]}')


# ## 8. Construct correlation matrices using r(t) of epoch size 20 working days and with a shift of 10 days.

# In[12]:


#Construct correlation matrices using r(t) of epoch size 20 working days and with a shift of 10 days.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_correlation_matrices(data, window_size, shift_size):
    """
    Compute correlation matrices for rolling windows in the data.

    Parameters:
    - data: The DataFrame containing filtered log returns.
    - window_size: The number of days (rows) in each window (epoch).
    - shift_size: The shift step between windows.

    Returns:
    - correlation_matrices: A list of correlation matrices for each window.
    """
    correlation_matrices = []
    num_days = len(data)

    # Iterate through data using rolling windows
    for start in range(0, num_days - window_size + 1, shift_size):
        window_data = data.iloc[start:start + window_size]
        correlation_matrix = window_data.corr()
        correlation_matrices.append(correlation_matrix)

    return correlation_matrices

def visualize_correlation_matrix(correlation_matrix, start_day, window_size):
    """
    Visualize a correlation matrix using a heatmap.

    Parameters:
    - correlation_matrix: The correlation matrix to visualize.
    - start_day: The starting day of the window for title reference.
    - window_size: The window size for title reference.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm', annot=False)
    plt.title(f'Correlation Matrix from day {start_day} to {start_day + window_size}')
    plt.show()

def save_correlation_matrices(correlation_matrices):
    """
    Save each correlation matrix to a CSV file.

    Parameters:
    - correlation_matrices: The list of correlation matrices.
    """
    for i, matrix in enumerate(correlation_matrices):
        matrix.to_csv(f'correlation_matrix_epoch_{i}.csv')

# Main function to drive the process
def main(filtered_log_returns, window_size, shift_size):
    correlation_matrices = compute_correlation_matrices(filtered_log_returns, window_size, shift_size)

    # Visualize each correlation matrix (optional)
    for i, matrix in enumerate(correlation_matrices):
        start_day = i * shift_size
        visualize_correlation_matrix(matrix, start_day, window_size)

    # Save the correlation matrices to CSV (optional)
    save_correlation_matrices(correlation_matrices)

# Set your parameters
window_size = 20  # Epoch size of 20 days
shift_size = 10   # Shift size of 10 days

#'filtered_log_returns' is the DataFrame containing filtered log returns
main(filtered_log_returns, window_size, shift_size)


# In[ ]:





# ## 9. Construct a similarity matrix S(t1,t2)= <| C(t1) - C(t2) |>.

# In[13]:


#Construct a similarity matrix S(t1,t2)= <| C(t1) - C(t2) |>.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def compute_similarity_matrix(filtered_log_returns):
    """
    Compute the similarity matrix from the correlation matrix of the filtered log returns.

    Parameters:
    - filtered_log_returns: The DataFrame containing filtered log returns.

    Returns:
    - similarity_matrix: The computed similarity matrix.
    - similarity_df: A DataFrame version of the similarity matrix for easier viewing and saving.
    """
    # Compute the correlation matrix for the entire dataset
    correlation_matrix = filtered_log_returns.corr()

    #  Compute the similarity matrix as 1 - |correlation_matrix|
    similarity_matrix = 1 - np.abs(correlation_matrix)

    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix,
                                 columns=correlation_matrix.columns,
                                 index=correlation_matrix.index)

    return similarity_matrix, similarity_df

def visualize_similarity_matrix(similarity_matrix):
    """
    Visualize the similarity matrix using a heatmap with a vibrant color palette.

    Parameters:
    - similarity_matrix: The computed similarity matrix to visualize.
    """
    # Visualize the similarity matrix
    plt.figure(figsize=(12, 10))

    # Using 'coolwarm' colormap for better distinction of values
    sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False, cbar=True,
                linewidths=0.5, linecolor='white')

    plt.title('Colorful Similarity Matrix for Filtered Log Returns', fontsize=16)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)   # Keep y-axis labels horizontal


    plt.tight_layout()
    plt.show()

def save_similarity_matrix(similarity_df, filename='similarity_matrix.csv'):
    """
    Save the similarity matrix DataFrame to a CSV file.

    Parameters:
    - similarity_df: The similarity matrix as a DataFrame.
    - filename: The name of the file to save the matrix. Default is 'similarity_matrix.csv'.
    """
    similarity_df.to_csv(filename)
    print(f"Similarity matrix saved to {filename}")

# Main function to drive the process
def main(filtered_log_returns):
    # Compute similarity matrix
    similarity_matrix, similarity_df = compute_similarity_matrix(filtered_log_returns)

    # Displaying the similarity DataFrame
    print(similarity_df)

    # Visualize the similarity matrix
    visualize_similarity_matrix(similarity_matrix)

    # Save the similarity matrix to a CSV file
    save_similarity_matrix(similarity_df)

# 'filtered_log_returns' is the DataFrame containing filtered log returns
main(filtered_log_returns)


# In[ ]:





# ## 10. Construct a 3D Multidimensional scaling plot using the Similarity matrix “S”.

# In[14]:


#Construct a 3D Multidimensional scaling plot using the Similarity matrix “S”.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def compute_similarity_matrix(filtered_log_returns):
    """
    Compute the similarity matrix from the correlation matrix of the filtered log returns.

    Parameters:
    - filtered_log_returns: The DataFrame containing filtered log returns.

    Returns:
    - similarity_matrix: The computed similarity matrix.
    - similarity_df: A DataFrame version of the similarity matrix for easier viewing and saving.
    """
    correlation_matrix = filtered_log_returns.corr()
    similarity_matrix = 1 - np.abs(correlation_matrix)
    similarity_df = pd.DataFrame(similarity_matrix,
                                 columns=correlation_matrix.columns,
                                 index=correlation_matrix.index)
    return similarity_matrix, similarity_df

def symmetrize_matrix(similarity_df):
    """
    Symmetrize the similarity matrix by averaging it with its transpose.

    Parameters:
    - similarity_df: The DataFrame containing the original similarity matrix.

    Returns:
    - symmetrized_matrix: The symmetrized similarity matrix as a NumPy array.
    """
    return (similarity_df.values + similarity_df.values.T) / 2

def apply_mds(similarity_matrix, n_components=3, random_state=42):
    """
    Apply MDS for dimensionality reduction.

    Parameters:
    - similarity_matrix: The symmetric similarity matrix.
    - n_components: Number of dimensions to reduce to (default is 3).
    - random_state: Random state for reproducibility (default is 42).

    Returns:
    - mds_coordinates: The transformed coordinates after applying MDS.
    """
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=random_state)
    return mds.fit_transform(similarity_matrix)

def plot_mds_3d(mds_coordinates, asset_names):
    """
    Create a 3D plot of MDS coordinates with asset names annotated.

    Parameters:
    - mds_coordinates: The coordinates after applying MDS.
    - asset_names: The list of asset names corresponding to the coordinates.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(mds_coordinates[:, 0], mds_coordinates[:, 1], mds_coordinates[:, 2])

    # Annotate each point with its corresponding asset name
    for i, asset in enumerate(asset_names):
        ax.text(mds_coordinates[i, 0], mds_coordinates[i, 1], mds_coordinates[i, 2], asset)

    # Set labels for the axes
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_zlabel('MDS Dimension 3')


    plt.title('3D MDS Plot based on Similarity Matrix')
    plt.show()

def main(filtered_log_returns):
    # Ensure only numeric columns are used (e.g., filter out any 'date' or non-numeric columns)
    numeric_log_returns = filtered_log_returns.select_dtypes(include=[np.number])

    _, similarity_df = compute_similarity_matrix(numeric_log_returns)

    symmetrized_matrix = symmetrize_matrix(similarity_df)

    # Apply MDS to the symmetrized matrix
    mds_coordinates = apply_mds(symmetrized_matrix)

    #Plot the MDS results in 3D
    plot_mds_3d(mds_coordinates, similarity_df.columns)

#  'filtered_log_returns' is the DataFrame containing filtered log returns
# 'filtered_log_returns' with your actual data.
# filtered_log_returns = pd.read_csv('your_filtered_log_returns.csv')
main(filtered_log_returns)


# In[ ]:





# ## 11. Find an optimum number of clusters using 1000 different conditions for the elbow method of k means clustering.
# 
# ## 12. Plot k-mean clustering results with optimum value of clustering found in No. 10 (above)

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def symmetrize_matrix(similarity_df):
    return (similarity_df.values + similarity_df.values.T) / 2

def apply_mds(similarity_matrix, n_components=3, random_state=42):
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=random_state)
    return mds.fit_transform(similarity_matrix)

def calculate_wcss(data, max_clusters):
    """
    Ensure that max_clusters does not exceed the number of samples.
    """
    max_clusters = min(max_clusters, data.shape[0])  # Ensure max_clusters <= number of samples
    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS
    return wcss

def plot_elbow_curve(avg_wcss_values, max_k):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), avg_wcss_values, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method (Average WCSS) for Optimal k (Number of Clusters)', fontsize=14)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average WCSS')
    plt.grid(True)
    plt.show()

def perform_kmeans_clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(data)

def add_random_noise(matrix, noise_level=0.05):
    noise = np.random.normal(0, noise_level, matrix.shape)
    perturbed_matrix = matrix + noise
    perturbed_matrix = (perturbed_matrix + perturbed_matrix.T) / 2  # Re-symmetrize
    np.fill_diagonal(perturbed_matrix, 1)  # Ensure the diagonal is 1
    return perturbed_matrix

def main(similarity_df, iterations=1000, max_clusters=20, noise_level=0.05):
    # Symmetrize the similarity matrix
    similarity_matrix = symmetrize_matrix(similarity_df)

    # Initialize an array to accumulate WCSS values
    total_wcss = np.zeros(min(max_clusters, similarity_df.shape[0]))  # Make sure WCSS fits the data

    # Iterate 1000 times for different conditions
    for iteration in range(iterations):
        # Add random noise to simulate different conditions
        perturbed_matrix = add_random_noise(similarity_matrix, noise_level=noise_level)

        # Apply MDS to the perturbed matrix
        mds_coordinates = apply_mds(perturbed_matrix)

        # Calculate WCSS for this iteration, ensuring max_clusters <= samples
        wcss_values = calculate_wcss(mds_coordinates, max_clusters=max_clusters)

        # Accumulate WCSS values
        total_wcss[:len(wcss_values)] += np.array(wcss_values)

        if iteration % 100 == 0:
            print(f"Iteration {iteration} complete")

    # Average the WCSS over all iterations
    avg_wcss_values = total_wcss / iterations

    # Plot the Elbow curve based on the average WCSS
    plot_elbow_curve(avg_wcss_values, len(avg_wcss_values))

    # Choose the optimal k (this should be done manually based on the Elbow plot)
    k_optimal = np.argmin(np.diff(np.diff(avg_wcss_values))) + 2  # Simple heuristic for elbow point

    # Perform K-Means clustering with the optimal number of clusters
    clusters = perform_kmeans_clustering(mds_coordinates, k_optimal)

    # Plot the final clustering result
    plot_3d_clustering(mds_coordinates, clusters, k_optimal)

def plot_3d_clustering(mds_coordinates, clusters, k):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(mds_coordinates[:, 0], mds_coordinates[:, 1], mds_coordinates[:, 2],
                         c=clusters, cmap='viridis', marker='o')

    # Add color bar for cluster labels
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_zlabel('MDS Dimension 3')
    plt.title(f'3D K-Means Clustering (Optimal k = {k})')
    plt.show()


# Example similarity matrix for testing
np.random.seed(42)
size = 10  # Size of the matrix
random_similarity_matrix = np.random.rand(size, size)
random_similarity_matrix = (random_similarity_matrix + random_similarity_matrix.T) / 2  # Symmetrize the matrix
np.fill_diagonal(random_similarity_matrix, 1)  # Fill diagonal with 1s for similarity

# Convert to DataFrame
similarity_df = pd.DataFrame(random_similarity_matrix)

# Run the main function with 1000 iterations and 20 maximum clusters
main(similarity_df, iterations=1000, max_clusters=20, noise_level=0.05)


# In[ ]:





# ## 13. Plot typical correlation structures of different states,
# ## 14. Arrange all the correlation matrices for the market states (y-axis) and time (x-axis),

# In[22]:


#Plot typical correlation structures of different states
#Arrange all the correlation matrices for the market states (y-axis) and time (x-axis),
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def preprocess_data(df):
    """
    Preprocess the DataFrame to keep only numeric columns.

    Parameters:
    - df: DataFrame containing log returns.

    Returns:
    - processed_df: DataFrame with only numeric columns.
    """
    # Keep only numeric columns
    return df.select_dtypes(include=[np.number])

def calculate_matrix_stats(correlation_matrix):
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    upper_triangle = correlation_matrix.where(mask).values.flatten()
    upper_triangle = upper_triangle[~np.isnan(upper_triangle)]

    mean_corr = np.mean(np.abs(upper_triangle))
    std_corr = np.std(upper_triangle)

    return mean_corr, std_corr

def classify_correlation_state(correlation_matrix):
    mean_corr, _ = calculate_matrix_stats(correlation_matrix)

    if mean_corr < 0.3:
        return 1    # S1: Low correlation
    elif mean_corr < 0.5:
        return 2    # S2: Medium-low correlation
    elif mean_corr < 0.7:
        return 3    # S3: Medium-high correlation
    else:
        return 4    # S4: High correlation

# Preprocess the filtered_log_returns DataFrame
filtered_log_returns_numeric = preprocess_data(filtered_log_returns)

# Set the window size and shift size for rolling correlation
window_size = 20
shift_size = 10

# List to store correlation matrices and corresponding dates
correlation_matrices = []
matrix_dates = []

# Iterate through the data using a rolling window
for start in range(0, len(filtered_log_returns_numeric) - window_size + 1, shift_size):
    window_data = filtered_log_returns_numeric.iloc[start:start + window_size]
    correlation_matrix = window_data.corr()
    correlation_matrices.append(correlation_matrix)

    # Store the middle date of the window
    middle_date = window_data.index[window_size // 2]
    matrix_dates.append(middle_date)

# Classify states
classified_states = [classify_correlation_state(matrix) for matrix in correlation_matrices]

# Create a time series plot of states with actual dates
plt.figure(figsize=(15, 5))
plt.plot(matrix_dates, classified_states, 'b.')
plt.yticks([1, 2, 3, 4], ['S1', 'S2', 'S3', 'S4'])
plt.xlabel('Time (Year)')
plt.ylabel('Market State')
plt.title('Market States Over Time')

# Format x-axis to show years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.grid(True)
plt.tight_layout()
plt.show()

# Function to display example matrices for each state
def display_example_matrices():
    examples = {1: None, 2: None, 3: None, 4: None}
    example_dates = {1: None, 2: None, 3: None, 4: None}

    for state in range(1, 5):
        state_indices = [i for i, s in enumerate(classified_states) if s == state]
        if state_indices:
            examples[state] = correlation_matrices[state_indices[0]]
            example_dates[state] = matrix_dates[state_indices[0]]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for state, ax in zip(range(1, 5), axes):
        if examples[state] is not None:
            sns.heatmap(examples[state], vmin=-1, vmax=1, cmap='coolwarm', ax=ax)

            ax.set_title(f'State S{state}')
    plt.tight_layout()
    plt.show()

# Display example matrices
display_example_matrices()

# Print state distribution
state_counts = pd.Series(classified_states).value_counts().sort_index()
print("\nState Distribution:")
for state, count in state_counts.items():
    print(f"State S{state}: {count} occurrences ({count/len(classified_states)*100:.2f}%)")


# In[ ]:





# ## 15. Show the transition of consecutive market states,

# In[21]:


#Show the transition of consecutive market states,
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_transition_matrix(states, num_states=4):
    """
    Create a transition matrix from a list of classified states.

    Parameters:
    - states: List of classified states (e.g., [1, 2, 2, 3, ...]).
    - num_states: Number of unique states (default is 4 for S1-S4).

    Returns:
    - transition_counts: 2D numpy array representing the transition matrix.
    """
    # Create pairs of consecutive states
    state_pairs = list(zip(states[:-1], states[1:]))

    # Initialize transition count matrix
    transition_counts = np.zeros((num_states, num_states), dtype=int)

    # Count transitions
    for from_state, to_state in state_pairs:
        transition_counts[from_state-1, to_state-1] += 1

    return transition_counts

def plot_transition_matrix(transition_counts):
    """
    Plot a heatmap of the transition matrix.

    Parameters:
    - transition_counts: 2D numpy array representing the transition matrix.
    """
    plt.figure(figsize=(10, 8))

    # Create a mask for zero values (to hide zero transitions in the heatmap)
    mask = transition_counts == 0

    # Plot heatmap
    ax = sns.heatmap(transition_counts,
                     annot=True,
                     fmt='d',
                     cmap='Blues',
                     mask=mask,
                     cbar_kws={'label': 'Number of Transitions'},
                     xticklabels=['S1', 'S2', 'S3', 'S4'],
                     yticklabels=['S1', 'S2', 'S3', 'S4'])

    plt.title('Market State Transitions')
    plt.xlabel('To State')
    plt.ylabel('From State')

    plt.tight_layout()
    plt.show()

def print_transition_statistics(transition_counts):
    """
    Print the transition counts between market states.

    Parameters:
    - transition_counts: 2D numpy array representing the transition matrix.
    """
    print("\nTransition Statistics:")
    for from_state in range(transition_counts.shape[0]):
        for to_state in range(transition_counts.shape[1]):
            count = transition_counts[from_state, to_state]
            if count > 0:
                print(f"S{from_state+1} to S{to_state+1}: {count} transitions")

def calculate_transition_probabilities(transition_counts):
    """
    Calculate the transition probabilities based on the transition matrix.

    Parameters:
    - transition_counts: 2D numpy array representing the transition matrix.

    Returns:
    - transition_probabilities: 2D numpy array representing the transition probabilities.
    """
    # Normalize the transition counts to obtain probabilities
    transition_probabilities = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    return transition_probabilities

def print_transition_probabilities(transition_probabilities):
    """
    Print the transition probabilities between market states.

    Parameters:
    - transition_probabilities: 2D numpy array representing the transition probabilities.
    """
    print("\nTransition Probabilities:")
    for from_state in range(transition_probabilities.shape[0]):
        for to_state in range(transition_probabilities.shape[1]):
            prob = transition_probabilities[from_state, to_state]
            if prob > 0:
                print(f"Probability S{from_state+1} to S{to_state+1}: {prob:.3f}")

def main(classified_states):
    """
    Main function to calculate transition matrix, plot heatmap, and print statistics and probabilities.

    Parameters:
    - classified_states: List of classified states (S1-S4).
    """
    # Step 1: Create the transition matrix
    transition_counts = create_transition_matrix(classified_states)

    # Step 2: Plot the transition matrix heatmap
    plot_transition_matrix(transition_counts)

    # Step 3: Print transition statistics
    print_transition_statistics(transition_counts)

    # Step 4: Calculate and print transition probabilities
    transition_probabilities = calculate_transition_probabilities(transition_counts)
    print_transition_probabilities(transition_probabilities)

# Example usage:
# Assuming 'classified_states' is the list of states from your previous calculations
main(classified_states)


# In[ ]:





# In[ ]:





# In[ ]:




