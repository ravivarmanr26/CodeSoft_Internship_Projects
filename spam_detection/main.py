import streamlit as st
import pickle
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the saved model
with open('spam_classifier_using_spacy_model.pkl', 'rb') as file:
    clf = pickle.load(file)


def predict_spam(email_text):
    # Process the text with spaCy
    doc = nlp(email_text)

    # Get the vector representation of the document
    email_vector = doc.vector

    # Reshape the vector to be 2D
    email_vector_2d = email_vector.reshape(1, -1)

    # Use the classifier to predict
    prediction = clf.predict(email_vector_2d)

    return prediction[0]  # Return the prediction (0 for not spam, 1 for spam)


# Streamlit app
def main():
    st.title("Spam Email Classifier")

    email_text = st.text_area("Enter the email text:")

    if st.button("Predict"):
        if email_text:
            prediction = predict_spam(email_text)
            st.write("Prediction: ", "Spam" if prediction == 1 else "Not Spam")
        else:
            st.write("Please enter some email text to predict.")


if __name__ == "__main__":
    main()