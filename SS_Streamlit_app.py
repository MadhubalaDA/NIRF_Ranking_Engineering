#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from Excel file
SS1 = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=0)

# Create input and output data
y = SS1[['SS cal=Criteria*Ratio']]
x = SS1[['SI', 'TE']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=31)
y_train = y_train.values.ravel()

# Train model
model = RandomForestRegressor(n_estimators=1000, random_state=31)
rf = model.fit(x_train, y_train)

# Create Streamlit app
st.title("NIRF Engineering SS (student)Calculation App")

# Display data from Excel file
st.write("Input data:")
st.write(SS1)

# Get user input
st.sidebar.header("Enter values for prediction")
si = st.sidebar.number_input("SI value", value=0.0)
te = st.sidebar.number_input("TE value", value=0.0)

# Make prediction
input_data = [[si, te]]
Student_score = model.predict(input_data)
st.write("Prediction:")
st.write(Student_score[0])

# Compute the Mean Squared Error
#mse = mean_squared_error(y_test['SS cal=Criteria*Ratio'], prediction)
#st.write("Mean Squared Error:", mse)

y_test_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_test_pred)
st.write("Mean Squared Error:", mse)

#PhD calculation
# Load data from Excel file
SS2 = pd.read_excel(r"C:\Users\madhu\Documents\NIRF Engineering calculation.xlsx", sheet_name=11)

# Create input and output data
y11=SS2[['Score']]
x11=SS2[['PhD pursuing till 2021 FT','PhD pursuing till 2021 PT']]

# Train-test split
x_train11, x_test11, y_train11, y_test11 = train_test_split(x11, y11, test_size=0.2, random_state=31)
y_train11 = y_train11.values.ravel()

# Train model
model11 = RandomForestRegressor(n_estimators=1000, random_state=31)
rf11 = model11.fit(x_train11, y_train11)

# Create Streamlit app
st.title("NIRF Engineering SS(PhD) Calculation App")

# Display data from Excel file
st.write("Input data:")
st.write(SS2)

# Get user input
st.sidebar.header("Enter values for prediction")
FT = st.sidebar.number_input("No. of PhD pursuing Full time", value=0.0)
PT = st.sidebar.number_input("No. of PhD pursuing Part time", value=0.0)

# Make prediction
input_data = [[FT, PT]]
PhD_score = model.predict(input_data)
st.write("Prediction:")
st.write(PhD_score[0])

# Compute the Mean Squared Error
#mse = mean_squared_error(y_test['SS cal=Criteria*Ratio'], prediction)
#st.write("Mean Squared Error:", mse)

y_test_pred11 = model11.predict(x_test11)
mse11 = mean_squared_error(y_test11, y_test_pred11)
st.write("Mean Squared Error:", mse11)

# Calculate the sum of SS1 and SS2
SS = Student_score + PhD_score

# Display the value of SS in Streamlit
st.write("SS:", SS)


# In[ ]:




