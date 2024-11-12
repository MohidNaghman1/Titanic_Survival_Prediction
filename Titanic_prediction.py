import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('/mount/src/titanic_survival_prediction/titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def make_prediction(input_data):
    
    prediction = model.predict(input_data)
    return prediction

# Streamlit app layout with enhanced styling
st.markdown(
    """
    <style>
    body {
        background-color: #c4b086;  
        color: #333;  /* Dark text color */
        font-family: 'Arial', sans-serif;  /* Font style for the app */
        padding: 20px;  /* Padding around the app */
    }
    .stButton>button {
        background-color: #4CAF50;  /* Green background for buttons */
        color: white;  /* White text for buttons */
        font-size: 16px;  /* Button text size */
        border-radius: 5px;  /* Rounded corners */
        padding: 10px;  /* Padding inside buttons */
        transition: background-color 0.3s;  /* Smooth transition */
    }
    .stButton>button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }
    .stSelectbox, .stNumberInput {
        border: 1px solid #ccc;  /* Light gray border for input fields */
        border-radius: 5px;  /* Rounded corners for input fields */
        padding: 10px;  /* Padding inside input fields */
        font-size: 14px;  /* Input text size */
    }
    h1 {
        color: #2E8B57;  /* Dark green for title */
        text-align: center;  /* Centered title */
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 12px;
        color: #777;
    }
      .image-container {
        text-align: center;  /* Center the image */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit app layout
st.title("Titanic Survival Prediction")


# Center the image using st.image
st.image("titanic.jpeg", width=120, caption="Titanic", use_container_width=True)


# Input features
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
class_ = st.selectbox("Class", ["First", "Second", "Third"])
who = st.selectbox("Who", ["man", "woman", "child"])
adult_male = st.selectbox("Adult Male", ["Yes", "No"])
alone = st.selectbox("Alone", ["Yes", "No"])

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'pclass': [pclass],
    'sex': [sex],
    'age': [age],
    'sibsp': [sibsp],
    'parch': [parch],
    'fare': [fare],
    'embarked': [embarked],
    'class': [class_],
    'who': [who],
    'adult_male': [adult_male],
    'alone': [alone]
})

# Button to make prediction
if st.button("Predict"):
    prediction = make_prediction(input_data)
    
    # Show prediction result
    if prediction[0] == 0:
        st.success("The passenger did not survive.")
    else:
        st.success("The passenger survived!")

# Footer with additional information
st.markdown('<div class="footer">Mohid Naghman ❤️ </div>', unsafe_allow_html=True)
