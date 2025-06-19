import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("machine.data", sep=",", header=None)
data.columns = ["vendor name","Model","MYCT","MMIN", "MMAX","CACH", "CHMIN","CHMAX","PRP","ERP"]
data = data[["MYCT","MMIN", "MMAX","CACH", "CHMIN","CHMAX","PRP"]]

# Split Data
X = np.array(data.drop(["PRP"], axis=1))
y = np.array(data["PRP"])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Train model or load if exists
try:
    with open("hardware_performance.pickle", "rb") as f:
        model = pickle.load(f)
except:
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    with open("hardware_performance.pickle", "wb") as f:
        pickle.dump(model, f)

# App Interface
st.title("Computer Hardware Performance Predictor 🖥️")
st.write("Dataset: Computer Hardware Data Set from UCI Machine Learning Repository")

# User Inputs
MYCT = st.number_input("Machine cycle time (ns)", min_value=1, value=125)
MMIN = st.number_input("Minimum main memory (KB)", min_value=1, value=640)
MMAX = st.number_input("Maximum main memory (KB)", min_value=1, value=32000)
CACH = st.number_input("Cache memory (KB)", min_value=0, value=32)
CHMIN = st.number_input("Min channels (units)", min_value=0, value=1)
CHMAX = st.number_input("Max channels (units)", min_value=0, value=4)

# Predict
if st.button("Predict Performance"):
    features = np.array([[MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX]])
    prediction = model.predict(features)
    st.success(f"Estimated Relative Performance (PRP): {prediction[0]:.2f}")

# Show Scatter Plot (Optional)
st.subheader("Explore Attribute vs Performance")
attr = st.selectbox("Select Attribute", ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"])
st.write(f"Scatter Plot: {attr} vs PRP")
st.scatter_chart(data[[attr, "PRP"]])

