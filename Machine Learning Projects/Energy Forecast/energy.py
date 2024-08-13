import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv('01 renewable-share-energy.csv')
aus = df[df.Entity == 'Australia']
aus.columns = ['Entity', 'Code', 'Year', 'Renewable']
x = aus.Renewable
diff1 = np.diff(x,n=1)
diffD1 = np.diff(diff1, n=2)
bm = SARIMAX(x, order=(2,1,2)).fit()
forecast = bm.forecast(15)
fore = pd.DataFrame({'Forecasted %':forecast.round(2)})
fore['Year']=pd.Series()
for i in range(0,15):
    fore['Year'].iloc[i]= 2022+i
import streamlit as st
forecast_df = fore
aus = aus
st.title('Forecasting Future Renewable %')

st.subheader('Forecast Plot')
fig,ax = plt.subplots()
ax.plot(aus.Year, x)
ax.plot(fore.Year, fore['Forecasted %'], label='forecasted %')
ax.set_xlim(2000,2032)
ax.legend()
st.pyplot(fig)

st.subheader('Forecast Table')
st.write(forecast_df)