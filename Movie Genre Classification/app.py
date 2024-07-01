import streamlit as st
import pandas as pd
import pickle


html_attribution = """
    <div style="background-color:#28a745;padding:20px;margin-bottom:20px">
    <p style="color:white;text-align:center;font-size:22px;">Developed by Mr Steve</p>
    </div>
    """
st.markdown(html_attribution, unsafe_allow_html=True)

html_temp_subtitle = """
    <div style="background-color:#ff6347;padding:10px;margin-bottom:20px">
    <h2 style="color:white;text-align:center;">SPAM Detection</h2>
    </div>
    """
st.markdown(html_temp_subtitle, unsafe_allow_html=True)

st.subheader('Enter Text')

# Load the saved models
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('nb_model.pkl', 'rb') as f:
    NB = pickle.load(f)

# Function to predict genre
def class_predict(text):
    # Ensure the text is in a list
    text_vec = tfidf_vectorizer.transform([text])
    # Predict the genre
    y_pred = NB.predict(text_vec)
    return y_pred

# Streamlit app
def main():
    st.title("Movie Genre Classification")

    # Input form for user input
    st.subheader("Enter the movie plot summary:")
    text = st.text_area("Text", "Enter your movie plot summary here...")

    # Predict button
    if st.button("Predict Genre"):
        prediction = class_predict(text)
        st.write(f"Predicted Genre: {prediction}")

if __name__ == "__main__":
    main()
