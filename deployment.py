import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('model.pkl')

# Define features used in the model
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 
            'Sex_female', 'Sex_male', 
            'Embarked_C', 'Embarked_Q', 'Embarked_S']

st.title("Titanic Survival Prediction")
st.write("Enter passenger details below or upload a CSV file to predict survival.")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
sex = st.selectbox("Sex", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode the inputs
sex_female = 1 if sex == "Female" else 0
sex_male = 1 if sex == "Male" else 0
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Create DataFrame for single prediction
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, 
                            sex_female, sex_male, 
                            embarked_c, embarked_q, embarked_s]],
                          columns=features)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.write(f"Prediction: {result}")

uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file:
    uploaded_data = pd.read_csv(uploaded_file)

    # Fill missing Age values
    uploaded_data['Age'].fillna(uploaded_data['Age'].mean(), inplace=True)

    # One-hot encode 'Sex' and 'Embarked'
    uploaded_data = pd.get_dummies(uploaded_data, columns=['Sex', 'Embarked'])

    # Ensure all required features exist (add missing columns with 0)
    for col in features:
        if col not in uploaded_data.columns:
            uploaded_data[col] = 0

    # Reorder columns to match model's training
    uploaded_data = uploaded_data[features]

    # Make predictions
    predictions = model.predict(uploaded_data)
    uploaded_data['Survived_Prediction'] = predictions

    st.write(uploaded_data)
