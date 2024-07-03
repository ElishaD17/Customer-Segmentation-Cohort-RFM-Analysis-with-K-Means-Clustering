# Customer Segmentation, Cohort & RFM Analysis with K-Means Clustering

## Overview
This project involves analyzing customer transactions to segment customers and optimize sales strategies. The workflow includes data cleaning, exploratory data analysis, RFM analysis, customer segmentation, K-Means clustering, and cohort analysis.

## Project Structure
1. **Data Cleaning & Exploratory Data Analysis**
   - Observe data structure and missing values using EDA and visualization techniques.
   - Perform descriptive analysis to understand feature relationships, clear noise, and handle missing values.

2. **RFM Analysis**
   - Analyze Orders, Customers, and Countries distribution to help develop sales policies and optimize resources.
   - Focus on UK transactions due to its high sales revenue and customer count.
   - Calculate Recency, Frequency, and Monetary values for UK customers and create an RFM table.

3. **Customer Segmentation**
   - Develop an RFM Segmentation Table using the RFM values.
   - Label customer segments based on their recency, frequency, and monetary values (e.g., "Big Spenders", "Lost Customers").

4. **Applying K-Means Clustering**
   - Pre-process data by examining feature correlations and distributions.
   - Normalize data for K-Means clustering.
   - Use Elbow method and Silhouette Analysis to determine the optimal number of clusters.
   - Apply K-Means clustering, visualize the cluster distribution using scatter plots, and analyze cluster properties with boxplots.

5. **Create Cohort and Conduct Cohort Analysis**
   - Conduct Cohort Analysis to break user data into related groups and analyze metrics such as retention, churn, and lifetime value. This aids in further customer segmentation and strategy development.

## What The Problem Is
To effectively segment customers and develop sales strategies, you must:
- Observe data structure and missing values using EDA and visualization.
- Perform descriptive analysis to clear noise and handle missing values.
- Analyze Orders, Customers, and Countries distribution to develop sales policies.
- Focus on UK transactions for detailed analysis due to its high sales revenue and customer count.
- Perform RFM Analysis to segment customers based on their purchasing behavior.
- Use K-Means Clustering to refine customer segments.
- Conduct Cohort Analysis to track retention, churn, and lifetime value metrics.

## Detailed Steps

### Data Cleaning & Exploratory Data Analysis
First, observe the structure of the data and identify missing values using exploratory data analysis and data visualization techniques. Conduct descriptive analysis to understand feature relationships, clear noise, and handle missing values. This prepares the dataset for further analysis.

### RFM Analysis
Before starting RFM Analysis, analyze the distribution of Orders, Customers, and Countries. This helps develop sales policies and optimize resource use. Focus on UK transactions for subsequent analysis due to its high sales revenue and customer count. Calculate Recency, Frequency, and Monetary values for UK customers and create an RFM table.

### Customer Segmentation
Create an RFM Segmentation Table using the RFM table. Segment customers by labeling them based on their recency, frequency, and monetary values (e.g., "Big Spenders", "Lost Customers").

### Applying K-Means Clustering
Perform data pre-processing by examining feature correlations and distributions. Normalize the data for K-Means clustering. Determine the optimal number of clusters using the Elbow method and Silhouette Analysis. Apply K-Means clustering, visualize the cluster distribution using scatter plots, and analyze cluster properties with boxplots. Tag clusters and interpret the results.

### Create Cohort and Conduct Cohort Analysis
Conduct Cohort Analysis to break user data into related groups and analyze metrics such as retention, churn, and lifetime value. This analysis aids in further customer segmentation and strategy development.

## Code Setup

```python
# Import Libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import missingno as msno 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import make_column_transformer

# Scaling
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

# Modelling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import plotly.graph_objects as go

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

# Figure & Display options
plt.rcParams["figure.figsize"] = (16, 9)
pd.set_option('max_colwidth',200)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Import additional libraries
import colorama
from colorama import Fore, Style
from termcolor import colored
from termcolor import cprint
import ipywidgets
from ipywidgets import interact
import pandas_profiling
from pandas_profiling.report.presentation.flavours.html.templates import create_html_assets
from wordcloud import WordCloud
import squarify as sq
```
## Conclusion
This project demonstrates the application of various data science techniques to segment customers and improve sales strategies. By analyzing customer transactions, performing RFM analysis, and utilizing K-Means clustering, we can identify distinct customer segments and develop targeted marketing strategies. Cohort analysis further enhances our understanding of customer behavior, aiding in retention and optimizing resource allocation.

## References
- Scikit-learn Documentation
- Pandas Documentation
- Plotly Documentation
- Kaggle Datasets

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.


