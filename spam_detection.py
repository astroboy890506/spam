# First, import Streamlit
import streamlit as st
from joblib import load
import pandas as pd

# Load your logistic regression model and CountVectorizer
lr_loaded = load('logistic_regression_model.joblib')
cv_loaded = load('count_vectorizer.joblib')

# Streamlit application starts here
def main():
    # Title of your web app
    st.title("Spam/Ham Prediction App")

    # Text box for user input
    user_input = st.text_input("Enter a sentence to check if it's spam or ham:")

    # Predict button
    if st.button('Predict'):
        if user_input:  # Check if the input is not empty
            # Transform the user input
            df = pd.DataFrame([user_input], columns=['text'])
            Snew = cv_loaded.transform(df['text'])

            # Make a prediction
            result = lr_loaded.predict(Snew)
            if result == 1:
                st.success("This message is likely Spam.")
            else:
                st.success("This message is likely Ham.")
        else:
            st.error("Please enter a sentence for prediction.")

if __name__ == '__main__':
    main()
