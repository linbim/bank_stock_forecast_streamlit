import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date, datetime, timedelta
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from datetime import timedelta
import statsmodels.api as sm
import itertools
import streamlit as st


# Function to load stock data
def load_stock_data(symbol):
    Bank_stock = f'{symbol}_monthly_stock_data.csv'
    data = pd.read_csv(Bank_stock)
    data['date'] = pd.to_datetime(data['date'])
    data['% change']=data['Adjusted_close']/data['Adjusted_close'].shift(1)
    return data[['date', 'Adjusted_close', '% change']]
#st.write()

# Function to load and preprocess economic data
def load_and_preprocess_economic_data():
    # Load CPI data
    CPI_data = pd.read_csv('C:/Users/linda/Desktop/CPI_data.csv')
    CPI_data['date'] = pd.to_datetime(CPI_data['date']) + pd.offsets.MonthEnd(0)
    CPI_data.rename(columns={'value': 'cpi_value'}, inplace=True)

    # Load unemployment data
    unemployment_data = pd.read_csv('C:/Users/linda/Desktop/unemployment_data.csv')
    unemployment_data['value'] = pd.to_numeric(unemployment_data['value'])
    unemployment_data['date'] = pd.to_datetime(unemployment_data['date']) + pd.offsets.MonthEnd(0)
    unemployment_data.rename(columns={'value': 'unemployment'}, inplace=True)

    # Load retail data
    retail_data = pd.read_csv('C:/Users/linda/Desktop/retail_data.csv')
    retail_data['value'] = pd.to_numeric(retail_data['value'])
    retail_data['date'] = pd.to_datetime(retail_data['date']) + pd.offsets.MonthEnd(0)
    retail_data.rename(columns={'value': 'Retail'}, inplace=True)

    # Load federal fund (interest) rate data
    interest_data = pd.read_csv('C:/Users/linda/Desktop/Interest_data.csv')
    interest_data['value'] = pd.to_numeric(interest_data['value'])
    interest_data['date'] = pd.to_datetime(interest_data['date']) + pd.offsets.MonthEnd(0)
    interest_data.rename(columns={'value': 'Interest'}, inplace=True)

    # Load GDP data
    gdp_data = pd.read_csv('C:/Users/linda/Desktop/gdp_data.csv')
    gdp_data['value'] = pd.to_numeric(gdp_data['value'], errors='coerce')
    gdp_data['date'] = pd.to_datetime(gdp_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    gdp_data.rename(columns={'value': 'GDP'}, inplace=True)
    gdp_data.set_index('date', inplace=True)
    rgdp_df = gdp_data.resample('M').interpolate(method='linear')
    gdp_sorted = rgdp_df.sort_index(ascending=False)
    r_gdp = gdp_sorted.reset_index()
    r_gdp['GDP'] = r_gdp['GDP'].round(2)

    # Load treasury yield data
    yield_data = pd.read_csv('C:/Users/linda/Desktop/yield_data.csv')
    yield_data['value'] = pd.to_numeric(yield_data['value'], errors='coerce')
    yield_data['date'] = pd.to_datetime(yield_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    yield_data.rename(columns={'value': 'treasure_yield'}, inplace=True)
    yield_data=yield_data.drop (columns=['Unnamed: 0'])

    # Load durables data
    durables_data = pd.read_csv('C:/Users/linda/Desktop/durables_data.csv')
    durables_data['value'] = pd.to_numeric(durables_data['value'], errors='coerce')
    durables_data['date'] = pd.to_datetime(durables_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    durables_data.rename(columns={'value': 'durables'}, inplace=True)
    durables_data=durables_data.drop(columns=['Unnamed: 0'])

    # Load payroll data
    payroll_data = pd.read_csv('C:/Users/linda/Desktop/payroll_data.csv')
    payroll_data['value'] = pd.to_numeric(payroll_data['value'], errors='coerce')
    payroll_data['date'] = pd.to_datetime(payroll_data['date'], errors='coerce') + pd.offsets.MonthEnd(0)
    payroll_data.rename(columns={'value': 'payroll'}, inplace=True)
    payroll_data=payroll_data.drop(columns=['Unnamed: 0'])

    # Merge all economic data
    dataframe = [CPI_data, unemployment_data, retail_data, interest_data, r_gdp, yield_data, durables_data, payroll_data]
    economic_data = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dataframe)
    return economic_data


# Setting up the initial dates and periods
START = '1999-11-30'
END = '2024-04-30'

# Streamlit App
st.title('Stock Price Analysis and Forecasting')

# Sidebar for stock selection
stock_symbol = st.sidebar.selectbox("Select a stock symbol:", ['JPM', 'MS', 'GS'])

# Slider to select prediction years
n_years = st.sidebar.slider('Years of prediction:', 1, 3)
period = n_years * 12

# Load stock data
stock_data = load_stock_data(stock_symbol)

# Load economic data
economic_data = load_and_preprocess_economic_data()

# Merge stock and economic data
data = pd.merge(stock_data, economic_data, on='date')
data_cleaned = data.dropna(subset=['GDP', 'durables'])
data_cleaned.set_index('date', inplace=True)

st.write(f'{stock_symbol} Stock data with economic indicators')
st.write(data_cleaned)

# Sort by 'date' in ascending order and reset the index
data_sorted = data.sort_values(by='date', ascending=True).reset_index(drop=True)

# Set 'date' as the index
data_sorted.set_index('date', inplace=True)


# Plot stock prices
st.subheader(f'{stock_symbol} Stock Price')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_cleaned.index, y=data_sorted['Adjusted_close'], mode='lines', name='Stock Price'))
fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

st.subheader(f' Effects of Major Economic Events on {stock_symbol}')
# visualising effect some major events on stock prices
economic_events = {
   # "ERM Crisis": "1992-09-16",
   # "Asian Financial Crisis": "1997-07-02",
    #"Russian Financial Crisis": "1998-08-17",
    "Dot-Com Bubble Burst": "2000-03-10",
    "9/11 Attacks": "2001-09-11",
    "Global Financial Crisis": "2008-09-15",
    "European Debt Crisis": "2010-05-09",
    "U.S. Debt Ceiling Crisis": "2011-08-02",
    "Oil Price Collapse": "2014-06-30",
    "Brexit Vote": "2016-06-23",
    "U.S.-China Trade War": "2018-03-22",
    "COVID-19 Pandemic": "2020-03-11",
    "Russian Invasion of Ukraine": "2022-02-24",
    "Global Inflation Surge": "2022-01-01",
    "Silicon Valley Bank Collapse": "2023-03-10"
}
# Convert event dates to datetime
economic_events = {event: pd.to_datetime(date) for event, date in economic_events.items()}

# Plotting the stock data
plt.figure(figsize=(14, 8))
plt.plot(data['date'], data['Adjusted_close'], label=stock_symbol)

# Add vertical lines for major economic events
for event, date in economic_events.items():
    plt.axvline(x=date, color='red', linestyle='--', alpha=0.7)
    plt.text(date, plt.ylim()[1] * 0.9, event, rotation=90, verticalalignment='bottom', fontsize=8, color='red')

# Customize the plot
#plt.title('Stock Prices with Major Economic Events')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)

# Show plot in Streamlit
st.pyplot(plt)

# Skewness transformation function
def safe_log1p(series):
    return np.log1p(series[series > 0])

def safe_sqrt(series):
    return np.sqrt(series[series >= 0])

def apply_skewness_transformations(data):
    transformed_data = pd.DataFrame(index=data.index)
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            skewness = skew(data[col].dropna())
            if skewness > 1:  
                transformed_data[col + '_log'] = safe_log1p(data[col])
            elif 0.5 < skewness <= 1:
                transformed_data[col + '_sqrt'] = safe_sqrt(data[col])
            else:
                transformed_data[col] = data[col]
        else:
            transformed_data[col] = data[col]
    return transformed_data.fillna(0)

# Apply transformations
data_transformed = apply_skewness_transformations(data_sorted)
st.subheader('transformed data')
st.write(data_transformed)

# Correlation Heatmap
st.subheader('Correlation Matrix')
corr_matrix = data_transformed.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
st.pyplot(fig)

# Feature Importance (Random Forest)
st.subheader('Feature Importance using Random Forest')

# #lets use the correct target column based on transformation
# if stock_symbol in ['BAC']:
#     target_column=['Adjusted_close'] #for BAC
# elif stock_data=='BAC':
#     target_column = 'Adjusted_close' #no
# else:
#     target_column=['Adjusted_close_log']#for other stock

features = data_transformed.drop(columns=['Adjusted_close_log'])
target = data_transformed['Adjusted_close_log']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(features, target)
importances = rf.feature_importances_
indices = np.argsort(importances)
fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], color='b', align='center')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features.columns[i] for i in indices])
ax.set_xlabel('Relative Importance')
st.pyplot(fig)

# Prophet Forecasting
st.subheader(f'{stock_symbol} Stock Price Forecast Prophet')
# Reset the index to access the 'date' column
data_transformed_reset = data_transformed.reset_index()

columns_to_select = ['Adjusted_close_log', 'date']
columns_to_select.extend(features.columns[indices[:6]])
data_ready = data_transformed_reset[columns_to_select]
data_ready.rename(columns={'date': 'ds', 'Adjusted_close_log': 'y'}, inplace=True)

model = Prophet()
for feature in features.columns[indices[:6]]:
    model.add_regressor(feature)
model.fit(data_ready)

# Getting Future forecast
future = model.make_future_dataframe(periods=period, freq='M')
for feature in features.columns[indices[:6]]:
    future[feature] = np.append(data_ready[feature].values, [data_ready[feature].values[-1]] * (len(future) - len(data_ready)))

forecast = model.predict(future)
fig = model.plot(forecast)
st.pyplot(fig)

data_transformed_reset = data_transformed.reset_index()

# Calculate RMSE and MAE for Prophet
y_true = data_transformed_reset['Adjusted_close_log']  # Actual values
y_pred_prophet = forecast['yhat'].iloc[:len(y_true)]   # Predicted values from Prophet

# RMSE for Prophet
rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
# MAE for Prophet
mae_prophet = mean_absolute_error(y_true, y_pred_prophet)

#visualise in streamlit
st.write(f'Prophet Model RMSE: {rmse_prophet}')
st.write(f'Prophet Model MAE: {mae_prophet}')


# --- ARIMA MODEL ---
# ARIMA Model setup and training
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
exog_vars = data_transformed[['GDP', 'unemployment_log', 'cpi_value', 'Retail_sqrt', '% change_log']]
aic_values = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data_transformed['Adjusted_close_log'],
                                            exog=exog_vars,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=False)
            aic_values.append((param, param_seasonal, results.aic))
        except Exception:
            continue

best_pdq, best_seasonal_pdq, best_aic = sorted(aic_values, key=lambda x: x[2])[0]

best_model = sm.tsa.statespace.SARIMAX(data_transformed['Adjusted_close_log'],
                                       exog=exog_vars,
                                       order=best_pdq,
                                       seasonal_order=best_seasonal_pdq,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
best_results = best_model.fit(disp=False)

# Start forecasting from '2020-03-31'
start_date = pd.to_datetime('2020-03-31')

# Generate predictions starting from '2020-03-31'
pred = results.get_prediction(start=start_date, dynamic=False)

# Get the predicted mean values
predicted_mean = pred.predicted_mean

# Extracting  the confidence intervals
pred_ci = pred.conf_int()

# to get the actual data from the forecast start date for comparison
actual_data =data_transformed.loc[start_date:, 'Adjusted_close_log']

st.subheader(f'{stock_symbol} forward forecast ARIMA')
fig, ax= plt.subplots()
# lets Plot the observed data and predicted values
ax = data_transformed['Adjusted_close_log'].plot(label='Observed', figsize=(14, 7))
predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=0.7)

# Plot the confidence intervals
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=0.2)

# we Add labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing stock')
plt.legend()
st.pyplot(fig)

# Calculate the Mean Squared Error (MSE) for the forecasted period
mse_Arima = mean_squared_error(actual_data, predicted_mean)
Mae_Arima = mean_absolute_error (actual_data, predicted_mean)
st.write(f'Mean Squared Error (MSE) from {start_date.date()}: {mse_Arima}')
st.write(f'Mean absolute Error (Mae) from {start_date.date()}: {Mae_Arima}')


# lets get the last known values for each exogenous variable (from data2_reset)
last_known_exog = data_transformed[['GDP', 'unemployment_log', 'cpi_value', 'Retail_sqrt', '% change_log']].iloc[-1]

# Create a DataFrame of future exog values by repeating the last known values for 45 months
future_exog = pd.DataFrame([last_known_exog] * period, columns=['GDP', 'unemployment_log', 'cpi_value', 'Retail_sqrt', '% change_log'])

# Forecast with the repeated future exog values
pred_uc = results.get_forecast(steps=period, exog=future_exog)

# Get the confidence intervals of the forecast
pred_ci = pred_uc.conf_int()

# Plot the historical data and forecasted values
st.subheader(f'{stock_symbol} Stock Price Forecast ARIMA')
fig, ax = plt.subplots()
ax = data_transformed['Adjusted_close_log'].plot(label='Historic', figsize=(14, 7))

# Plot the forecasted mean
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

# Plot the confidence intervals
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

# Add labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted_close')
plt.legend()
st.pyplot(fig)
