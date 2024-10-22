import streamlit as st
import pandas as pd
import pickle

# Load the saved model and scaler
filename = 'sarcoma_generisk_pred1.sav'
LR_model, X, y, df, X_train_scaled, y_train, X_test_scaled, y_test, y_pred_class_on_test = pickle.load(open(filename, 'rb'))

# Load the saved scaler used during training
scaler = pickle.load(open('scaler.sav', 'rb'))

# Streamlit App Title
st.title("Sarcoma Gene Expression Predictor")

# Section for user input
st.write("Select one Gene Combination and the Corresponding Expression Value (1-10)")

Gene_names = [
    'TotalDMD&Dp427m', 
    'TotalDMD&Dp71a', 
    'TotalDMD&Dp71ab', 
    'Dp427m&Dp71a', 
    'Dp427m&Dp71ab',
    'Dp71a&Dp71ab'
]

# Step 1: Select Gene Combination from the dropdown (Single selection)
selected_gene = st.selectbox(
    "Select Gene Combination", 
    Gene_names, 
    index=0  # Default to the first combination
)

# Step 2: Select Expression Value using a Slider
Expression_level = st.slider(f"Expression value for {selected_gene}", 1.0, 10.0, 5.0)

# Function to get user input for other variables like Age and BMI
def get_user_input():      
    Age = st.sidebar.slider('Age', 20.0, 120.0, 55.0)
    BMI = st.sidebar.slider('BMI', 18.0, 47.0, 25.0)

    # Store the user input in a dictionary
    user_data = {
        'Age': Age,
        'BMI': BMI,
        'Expression Level': Expression_level
    }

    return user_data

# Get user inputs
user_input = get_user_input()

# One-hot encode the selected gene combination
def one_hot_encode_gene(selected_gene, Gene_names):
    gene_vector = [0.0] * len(Gene_names)  # Initialize with zeros
    gene_index = Gene_names.index(selected_gene)  # Find index of selected gene
    gene_vector[gene_index] = 1.0  # Set the selected gene's position to 1.0
    return gene_vector

# Get one-hot encoded gene combination
encoded_genes = one_hot_encode_gene(selected_gene, Gene_names)

# Prepare the numerical variables (Age, BMI, Expression Level)
numerical_variables = [user_input['Age'], user_input['BMI'], user_input['Expression Level']]

# Combine numerical variables with one-hot encoded genes BEFORE scaling
full_input = numerical_variables + encoded_genes

# Convert to a 2D array to pass to the scaler (as scaler expects 2D input)
full_input = [full_input]

# Apply scaling to the full input (numerical + categorical one-hot encoded)
full_input_scaled = scaler.transform(full_input)

# Debugging output to check the final predictive variables
st.write("Predictive variables after scaling and one-hot encoding:", full_input_scaled)

# Prediction
prediction = LR_model.predict(full_input_scaled)

# Map the prediction to a corresponding message
if prediction[0] == 2:
    risk_message = "Predicted High-Risk"
elif prediction[0] == 1:
    risk_message = "Predicted Low-Risk"
else:
    risk_message = "Unknown Risk Level"  # Just in case there's another prediction value

# Display the prediction result
st.success(f"Prediction: {prediction[0]}  ({risk_message})")