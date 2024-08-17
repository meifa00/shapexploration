import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load the dataset
customer = pd.read_csv("Mall_Customers.csv")

# Create a synthetic target variable (considering high spending score as "churn")
customer['Churn'] = np.where(customer['Spending_Score'] > 50, 1, 0)

# Preprocessing
X = customer.drop(["Churn", "CustomerID"], axis=1)
y = customer['Churn']

# Convert categorical variable 'Genre' into dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Customer Churn")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.text("Classification Report")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Summary plot for class 0
st.subheader("Summary Plot for Class 0")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[0], X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=X_test[feature].mean())

# Create a DataFrame from input data
input_df = pd.DataFrame(input_data, index=[0])

# Make prediction
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]  # Probability of churn

# Display prediction
st.write(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
st.write(f"**Churn Probability:** {probability:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)

# Force plot
st.subheader("Force Plot")
st_shap(shap.force_plot(explainer.expected_value[0], shap_values_input[0], input_df), height=400, width=1000)

# Decision plot
st.subheader("Decision Plot")
st_shap(shap.decision_plot(explainer.expected_value[0], shap_values_input[0], X_test.columns))
