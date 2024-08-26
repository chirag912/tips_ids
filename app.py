import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
def load_data():
    data = pd.read_csv('tips.csv')
    return data

# Build the model
def build_model(data):
    X = data.drop('tip', axis=1)
    y = data['tip']

    categorical_cols = ['sex', 'smoker', 'day', 'time']
    numerical_cols = ['total_bill', 'size']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    
    return model

# Prediction function
def predict_tip(model, total_bill, sex, smoker, day, time, size):
    input_df = pd.DataFrame({
        'total_bill': [total_bill],
        'sex': [sex],
        'smoker': [smoker],
        'day': [day],
        'time': [time],
        'size': [size]
    })

    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit UI
def main():
    st.title("Tip Amount Prediction")

    # Input fields
    total_bill = st.number_input("Total Bill ($)", min_value=0.0, value=10.0)
    sex = st.selectbox("Gender", options=["Male", "Female"])
    smoker = st.selectbox("Smoker", options=["Yes", "No"])
    day = st.selectbox("Day of the Week", options=["Thur", "Fri", "Sat", "Sun"])
    time = st.selectbox("Time of Day", options=["Lunch", "Dinner"])
    size = st.number_input("Party Size", min_value=1, max_value=20, value=2)

    # Load data and build model
    data = load_data()
    model = build_model(data)

    # Predict button
    if st.button("Predict Tip"):
        tip = predict_tip(model, total_bill, sex, smoker, day, time, size)
        st.success(f"Predicted Tip Amount: ${tip:.2f}")

if __name__ == "__main__":
    main()

