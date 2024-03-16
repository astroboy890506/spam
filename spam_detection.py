# # First, import Streamlit
# import streamlit as st
# from joblib import load
# import pandas as pd

# # Load your logistic regression model and CountVectorizer
# lr_loaded = load('logistic_regression_model.joblib')
# cv_loaded = load('count_vectorizer.joblib')

# # Streamlit application starts here
# def main():
#     # Title of your web app
#     st.title("Spam/Ham Prediction App")

#     # Text box for user input
#     user_input = st.text_input("Enter a sentence to check if it's spam or ham:")

#     # Predict button
#     if st.button('Predict'):
#         if user_input:  # Check if the input is not empty
#             # Transform the user input
#             df = pd.DataFrame([user_input], columns=['text'])
#             Snew = cv_loaded.transform(df['text'])

#             # Make a prediction
#             result = lr_loaded.predict(Snew)
#             st.write(f"Predicted value: {result[0]}")

#         else:
#             st.error("Please enter a sentence for prediction.")

# if __name__ == '__main__':
#     main()

#============================================(file upload)

# import streamlit as st
# from joblib import load
# import pandas as pd

# # Load your logistic regression model and CountVectorizer
# lr_loaded = load('logistic_regression_model.joblib')
# cv_loaded = load('count_vectorizer.joblib')

# # Streamlit application starts here
# def main():
#     # Title of your web app
#     st.title("Spam/Ham Prediction App")

#     # File upload widget
#     uploaded_file = st.file_uploader("Choose a file (Excel or .txt). For Excel, ensure your data is in the first column.", type=['xlsx', 'txt'])
    
#     # Placeholder for displaying the dataframe and predictions
#     if uploaded_file is not None:
#         if uploaded_file.name.endswith('.xlsx'):
#             # Read the Excel file
#             df = pd.read_excel(uploaded_file)
#         elif uploaded_file.name.endswith('.txt'):
#             # Read the txt file
#             df = pd.read_csv(uploaded_file, header=None, names=['text'])
        
#         # Ensure the dataframe has the correct format
#         if 'text' not in df.columns:
#             st.error("Please make sure your data is in a column named 'text'.")
#             return

#         # Process the dataframe for predictions
#         Snew = cv_loaded.transform(df['text'])
#         predictions = lr_loaded.predict(Snew)
        
#         # Add predictions to the dataframe
#         df['Prediction'] = predictions
#         # df['Prediction'] = df['Prediction'].map({0: 'Ham', 1: 'Spam'})  # Convert numeric predictions to labels
        
#         # Display the dataframe with predictions
#         st.write(f"Predicted:")
#         st.dataframe(df)

# # Predict button and single sentence input (optional feature)
# def single_sentence_prediction():
#     user_input = st.text_input("Or, enter a single sentence to check if it's spam or ham:")
#     if st.button('Predict Single Sentence'):
#         if user_input:  # Check if the input is not empty
#             # Transform the user input
#             df = pd.DataFrame([user_input], columns=['text'])
#             Snew = cv_loaded.transform(df['text'])

#             # Make a prediction
#             result = lr_loaded.predict(Snew)
#             st.write(f"Predicted: {result[0]}")
#         else:
#             st.error("Please enter a sentence for prediction.")

# if __name__ == '__main__':
#     main()
#     single_sentence_prediction()

#======================================================(bar chart)
# First, import Streamlit, Pandas, and necessary libraries
import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt

# Load your logistic regression model and CountVectorizer
lr_loaded = load('logistic_regression_model.joblib')
cv_loaded = load('count_vectorizer.joblib')

# Streamlit application starts here
def main():
    # Title of your web app
    st.title("Spam/Ham Prediction App")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        # Text box for user input
        user_input = st.text_input("Enter a sentence to check if it's spam or ham:")

        # Predict button
        if st.button('Predict'):
            if user_input:  # Check if the input is not empty
                predict_and_display([user_input])  # Single sentence prediction
            else:
                st.error("Please enter a sentence for prediction.")
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume text file
                data = pd.read_table(uploaded_file, header=None, names=['text'])

            # Check if the file has content
            if not data.empty:
                sentences = data['text'].tolist()
                predict_and_display(sentences)  # File-based prediction

def predict_and_display(sentences):
    # Transform the sentences
    transformed_sentences = cv_loaded.transform(sentences)

    # Make predictions
    results = lr_loaded.predict(transformed_sentences)

    # Tabulate results
    counts = pd.Series(results).value_counts()
    st.write("Predictions:")
    st.table(counts)

    # Display histogram
    st.write("Histogram of Predictions:")
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_title("Number of Spam and Ham Predictions")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    st.pyplot(fig)

if __name__ == '__main__':
    main()

