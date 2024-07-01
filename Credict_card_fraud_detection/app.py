import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('model1.pkl', 'rb'))

# Define the HTML styling for the app
html_attribution = """
    <div style="background-color:#28a745;padding:20px;margin-bottom:20px">
    <p style="color:white;text-align:center;font-size:22px;">Developed by Mr Steve</p>
    </div>
    """
st.markdown(html_attribution, unsafe_allow_html=True)

html_temp_subtitle = """
    <div style="background-color:#ff6347;padding:10px;margin-bottom:20px">
    <h2 style="color:white;text-align:center;">Credit Card Fraud Detection</h2>
    </div>
    """
st.markdown(html_temp_subtitle, unsafe_allow_html=True)

# Function to preprocess input data
def preprocess_input(input_data):
    # Add preprocessing steps here as per your model requirements
    return input_data

# Function to predict using the model
def predict(model, input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    # Make prediction
    prediction = model.predict(processed_data)
    return prediction

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # Input form for user input
    st.subheader("Enter the input data:")
    # Example input fields (modify as per your dataset)
    category = st.number_input("Category", min_value=0, max_value=10)
    amt = st.number_input("Amount", min_value=0.0, max_value=1000.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.number_input("City Code", min_value=0, max_value=1000)
    zip_code = st.number_input("Zip Code", min_value=0, max_value=1000)
    age = st.number_input("Age", min_value=18, max_value=100)

    # Convert gender to numerical value (example, modify as per your dataset)
    gender_numeric = 0 if gender == "Male" else 1

    # Create a dictionary with input data
    input_data = {
        'category': category,
        'amt': amt,
        'gender': gender_numeric,
        'city': city,
        'zip': zip_code,
        'age': age
    }

    # Convert input data to DataFrame (modify as per your model input requirements)
    input_df = pd.DataFrame([input_data])

    # Predict button
    if st.button("Predict"):
        prediction = predict(model, input_df)
        if prediction[0] == 0:
            st.write("Prediction: Not a Fraud")
        else:
            st.write("Prediction: Is a Fraud")

if __name__ == "__main__":
    main()
