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

    # File upload widget
    uploaded_file = st.file_uploader("Choose a file (Excel or .txt). For Excel, ensure your data is in the first column.", type=['xlsx', 'txt'])
    
    # Placeholder for displaying the dataframe and predictions
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            # Read the txt file
            df = pd.read_csv(uploaded_file, header=None, names=['text'])
        
        # Ensure the dataframe has the correct format
        if 'text' not in df.columns:
            st.error("Please make sure your data is in a column named 'text'.")
            return

        # Process the dataframe for predictions
        Snew = cv_loaded.transform(df['text'])
        predictions = lr_loaded.predict(Snew)
        
        # Add predictions to the dataframe
        df['Prediction'] = predictions
        # df['Prediction'] = df['Prediction'].map({0: 'Ham', 1: 'Spam'})  # Convert numeric predictions to labels
        
        # Display the dataframe with predictions
        st.write(f"Predicted: {result[0]}")
        st.dataframe(df)

# Predict button and single sentence input (optional feature)
def single_sentence_prediction():
    user_input = st.text_input("Or, enter a single sentence to check if it's spam or ham:")
    if st.button('Predict Single Sentence'):
        if user_input:  # Check if the input is not empty
            # Transform the user input
            df = pd.DataFrame([user_input], columns=['text'])
            Snew = cv_loaded.transform(df['text'])

            # Make a prediction
            result = lr_loaded.predict(Snew)
            st.write(f"Predicted: {result[0]}")
        else:
            st.error("Please enter a sentence for prediction.")

if __name__ == '__main__':
    main()
    single_sentence_prediction()
