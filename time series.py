import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#-------------------------------------CO2----------------------------------------
data = pd.read_csv("CO2.csv")
data = data[data['CO2']>0]
data['time'] = data['Yr']+(data['Mn']+0.5)/12 - 1958

train_size = int(len(data) * 0.8+1)
train_data = data[:train_size]
test_data = data[train_size:]

##linear
x=train_data['time'].to_numpy().reshape(-1, 1)
y=train_data['CO2'].to_numpy().reshape(-1, 1)

model = LinearRegression()
model.fit(x,y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

y_pred = model.predict(x)
residuals = y - y_pred
plt.scatter(x, residuals)
plt.xlabel('time')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual plot')
plt.show()

x=test_data['time'].to_numpy().reshape(-1, 1)
y=test_data['CO2'].to_numpy().reshape(-1, 1)
y_pred = model.predict(test_data['time'].to_numpy().reshape(-1, 1))
residuals = y - y_pred

mse = mean_squared_error(y, y_pred)
rmse = np.power(mse,0.5)
mape = mean_absolute_percentage_error(y, y_pred)

print("RMSE:", rmse)
print("MAPE:", mape)

##quadratic
x=train_data['time'].to_numpy().reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
x_2 = poly.fit_transform(x)
y=train_data['CO2'].to_numpy().reshape(-1, 1)

model = LinearRegression()
model.fit(x_2,y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

y_pred = model.predict(x_2)
residuals = y - y_pred
plt.scatter(x, residuals)
plt.xlabel('time')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual plot')
plt.show()

x=test_data['time'].to_numpy().reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
x_2 = poly.fit_transform(x)
y=test_data['CO2'].to_numpy().reshape(-1, 1)
y_pred = model.predict(x_2)
residuals = y - y_pred

mse = mean_squared_error(y, y_pred)
rmse = np.power(mse,0.5)
mape = mean_absolute_percentage_error(y, y_pred)

print("RMSE:", rmse)
print("MAPE:", mape)

## cubic
np.set_printoptions(suppress=True)
x=train_data['time'].to_numpy().reshape(-1, 1)
poly = PolynomialFeatures(degree=3)
x_3 = poly.fit_transform(x)
y=train_data['CO2'].to_numpy().reshape(-1, 1)

model = LinearRegression()
model.fit(x_3,y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

y_pred = model.predict(x_3)
residuals = y - y_pred
plt.scatter(x, residuals)
plt.xlabel('time')
plt.ylabel('Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual plot')
plt.show()

x=test_data['time'].to_numpy().reshape(-1, 1)
poly = PolynomialFeatures(degree=3)
x_3 = poly.fit_transform(x)
y=test_data['CO2'].to_numpy().reshape(-1, 1)
y_pred = model.predict(x_3)
residuals = y - y_pred

mse = mean_squared_error(y, y_pred)
rmse = np.power(mse,0.5)
mape = mean_absolute_percentage_error(y, y_pred)

print("RMSE:", rmse)
print("MAPE:", mape)
## P jan，feb（quadratic）
x=train_data['time'].to_numpy().reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
x_2 = poly.fit_transform(x)
y=train_data['CO2'].to_numpy().reshape(-1, 1)

model = LinearRegression()
model.fit(x_2,y)

y_pred = model.predict(x_2)
residuals = y - y_pred
train_data['residuals'] = y - y_pred

p_jan = train_data.loc[data['Mn'] == 1, 'residuals'].mean()
p_feb = train_data.loc[data['Mn'] == 2, 'residuals'].mean()

print("p_jan:", p_jan)
print("p_feb:", p_feb)

## writing report
period_signal = []
for i in range(1,13):
    p = train_data.loc[data['Mn'] == i, 'residuals'].mean()
    period_signal.append(p)

plt.plot(range(1, 13), period_signal)
plt.xlabel('Month')
plt.ylabel('Residual CO2')
plt.title('Periodic Signal')
plt.show()
##-------------------------------------------------------

x_all = data['time'].to_numpy().reshape(-1, 1)
poly_all = PolynomialFeatures(degree=2)
x_all_2 = poly_all.fit_transform(x_all)
predicted_all = np.dot(x_all_2, model.coef_.T) + model.intercept_
data['predicted_all'] = predicted_all
data['period_signal'] = data['Mn'].apply(lambda x: period_signal[x-1])
data['final_fit'] = data['predicted_all']+data['period_signal']
plt.plot(train_data['time'], train_data['CO2'], label='Training Data')
plt.plot(test_data['time'], test_data['CO2'], label='Testing Data')
plt.plot(test_data['time'], test_data['final_fit'], label='Final Fit')
plt.xlabel('Time')
plt.ylabel('CO2')
plt.title('Final Fit Trend+Periodic Signal')
plt.legend()
plt.show()
##--------------------------------------------------------
train_size = int(len(data) * 0.8+1)
train_data = data[:train_size]
test_data = data[train_size:]

y = test_data['CO2']
y_pred = test_data['final_fit']

mse = mean_squared_error(y, y_pred)
rmse = np.power(mse,0.5)
mape = mean_absolute_percentage_error(y, y_pred)
print("RMSE:", rmse)
print("MAPE:", mape)
##--------------------------------------------------------
rangef=max(data['predicted_all'])-min(data['predicted_all'])
rangep=max(period_signal)-min(period_signal)
ranger=max(data['CO2']-data['final_fit'])-min(data['CO2']-data['final_fit'])
print(rangef/rangep)
print(rangep/ranger)

#-------------------------------------CPI&BER part1----------------------------------------

data1 = pd.read_csv("CPI.csv")
data1['date'] = pd.to_datetime(data1['date'])
data1['month'] = data1['date'].dt.month
data1['year'] = data1['date'].dt.year
data1['time'] = (data1['year']- 2008)*12+data1['month']-7
data1 = data1.groupby([data1['year'], data1['month']]).first().reset_index()
train_data1 = data1.loc[data1['time'] < 62]


x=train_data1['time'].to_numpy().reshape(-1, 1)
y=train_data1['CPI'].to_numpy().reshape(-1, 1)

plt.scatter(x, y)
plt.xlabel('time')
plt.ylabel('CPI')

model = LinearRegression()
model.fit(x,y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

y_pred = model.predict(x)
residuals = y - y_pred
print("max R:",np.max(residuals))


plot_pacf(residuals)
model = sm.tsa.AutoReg(residuals, lags=[1, 2], trend='c', old_names=False)
results = model.fit()
print(results.summary())


data1['trend_pred'] = 96.72932633 + 0.16104348*data1['time']
data1['residual'] = data1['CPI']-data1['trend_pred']
data1['residual_pred'] = 1.3237*data1['residual'].shift(1) + -0.5308*data1['residual'].shift(2)
data1['y_pred'] = data1['trend_pred'] + data1['residual_pred']

test_data1 = data1.loc[data1['time'] >= 62].dropna(subset=['CPI'])
y = test_data1['CPI']
y_pred = test_data1['y_pred']

mse = mean_squared_error(y, y_pred)
rmse = np.power(mse,0.5)
print("RMSE:", rmse)

#-------------------------------------CPI&BER part2----------------------------------------
data1 = pd.read_csv("CPI.csv")
data1['date'] = pd.to_datetime(data1['date'])
data1['month'] = data1['date'].dt.month
data1['year'] = data1['date'].dt.year
data1 = data1.groupby([data1['year'], data1['month']]).first().reset_index()

data1['IR'] = (data1['CPI']-data1['CPI'].shift(1))*100/data1['CPI']
data1['IR_ln'] = (np.log(data1['CPI'])-np.log(data1['CPI'].shift(1)))*100

data2 = pd.read_csv("T10YIE.csv")
data2['T10YIE'] = data2['T10YIE']/100
data2['DATE'] = pd.to_datetime(data2['DATE'])
data2['month'] = data2['DATE'].dt.month
data2['year'] = data2['DATE'].dt.year
data2 = data2.groupby([data2['year'], data2['month']]).mean().reset_index()
data2['time'] = (data2['year']- 2008)*12+data2['month']-7
data2['T10YIE'] =  (np.power(data2['T10YIE']+1,1/12)-1)*100


##writing report---------------------------------------------------------------------------

data1 = pd.read_csv("CPI.csv")
data1['date'] = pd.to_datetime(data1['date'])
data1['month'] = data1['date'].dt.month
data1['year'] = data1['date'].dt.year
data1 = data1.groupby([data1['year'], data1['month']]).first().reset_index()


data1['IR_ln'] = (np.log(data1['CPI'])-np.log(data1['CPI'].shift(1)))*100
data1 = data1.dropna(subset=['IR_ln'])

data1['time'] = (data1['year']- 2008)*12+data1['month']-7
train_data1 = data1.loc[data1['time'] < 62]
test_data1 = data1.loc[data1['time'] >= 62]


x=train_data1['time'].to_numpy().reshape(-1, 1)
y=train_data1['IR_ln'].to_numpy().reshape(-1, 1)

plt.plot(x, y)
plt.xlabel('time')
plt.ylabel('IR_ln')


model = sm.tsa.AutoReg(y, lags=[1], trend='c', old_names=False).fit()
print(model.summary())
data1['y_pred'] = 0.5178*data1['IR_ln'].shift(1)


plt.plot(train_data1['time'], train_data1['IR_ln'], label='Training Data')
plt.plot(test_data1['time'], test_data1['IR_ln'], label='Testing Data')
plt.plot(test_data1['time'], test_data1['y_pred'], label='Final Fit')
plt.xlabel('Time')
plt.ylabel('IR_ln')
plt.title('Final Fit Trend+Periodic Signal')
plt.legend()
plt.show()


mse = mean_squared_error(y, y_pred)
rmse = np.power(mse,0.5)
print("RMSE:", rmse)


y=test_data1['IR_ln'].to_numpy().reshape(-1, 1)
lag = []
rsme = []

for i in range(1, 20):
    lag.append(i)
    model = sm.tsa.AutoReg(y, lags=i, trend='c', old_names=False).fit()
    y_pred = model.forecast(steps=74)
    error = np.power(mean_squared_error(y, y_pred),0.5)
    rsme.append(error)

# Plot the forecast errors against different lag orders
plt.plot(lag,rsme)
plt.xlabel('lag')
plt.ylabel('RSME')
plt.show()

data2=data2[data2['time']>-1]
data2['BER']=data2['T10YIE']

plt.plot(data1['time'], data1['IR'], label='IR_CPI')
plt.plot(data1['time'], data1['IR_ln'], label='IR_CPI_Ln')
plt.plot(data2['time'], data2['T10YIE'], label='IR_BER')
plt.xlabel('Time')
plt.ylabel('IR_ln')
plt.title('Final Fit Trend+Periodic Signal')
plt.legend()
plt.show()
##6--------------------------
merge_data = pd.merge(data1, data2, on='time')
xcorr = np.correlate(merge_data['IR_ln'], merge_data['T10YIE'], mode='full')
plt.xcorr(merge_data['IR_ln'], merge_data['T10YIE'], maxlags=20)
plt.title('Cross-correlation function between IR_CPI_Ln and IR_BER')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.show()


merge_data['y_pred'] = 7.1138*merge_data['BER']+0.3751*merge_data['IR_ln'].shift(1)-0.6709

train_merge_data = merge_data.loc[merge_data['time'] < 62]
test_merge_data = merge_data.loc[merge_data['time'] >= 62]


x=train_merge_data['time'].to_numpy().reshape(-1, 1)
y=train_merge_data['IR_ln'].to_numpy().reshape(-1, 1)

model = sm.tsa.statespace.SARIMAX(train_merge_data['IR_ln'], exog=train_merge_data[['BER']], order=(1,0,0), trend='c').fit()
print(model.summary())



plt.plot(train_merge_data['time'], train_merge_data['IR_ln'], label='Training Data')
plt.plot(test_merge_data['time'], test_merge_data['IR_ln'], label='Testing Data')
plt.plot(test_merge_data['time'], test_merge_data['y_pred'], label='Final Fit')
plt.xlabel('Time')
plt.ylabel('IR_ln')
plt.title('Final Fit Trend+Periodic Signal')
plt.legend()
plt.show()

mse = mean_squared_error(test_merge_data['IR_ln'], test_merge_data['y_pred'])
rmse = np.power(mse,0.5)
print("RMSE:", rmse)