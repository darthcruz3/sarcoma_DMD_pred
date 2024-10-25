import streamlit as st
import pandas as pd
import pickle

# Load the saved Gradient Boosting model and scaler
filename = 'sarcoma_generisk_pred_XGB_model_all_genesfrom_manuscript.sav'
best_model, y_pred_class_on_test = pickle.load(open(filename, 'rb'))

# Load the saved scaler used during training
scaler = pickle.load(open('scaler_all_genes_from_manuscript.sav', 'rb'))

# Streamlit App Title
st.title("DMD transcript expression related outcome risk-prediction AI-simulator (using Haz-scores)")

# Section for user input
st.write("Select one Gene Combination, Tissue Type, and the Corresponding Expression Value (1-10)")

# added 'dummy_gene' to satisfy the 12 variable input
Gene_names = [
    'Dp71a', 
    'Dp71a_and_Dp71bb', 
    'Dp71ab', 
    'Dp71b', 
    'Dp427m',
    'Dp427m_and_Dp71a',
    'Total_DMD_and_Dp427m',
    'Total_DMD_gene',
    'dummy_gene'
]

# Step 1: Select Gene Combination from the dropdown (Single selection)
selected_gene = st.selectbox(
    "Select Gene Combination", 
    Gene_names, 
    index=0  # Default to the first combination
)

# Step 2: Select Expression Value using a Slider
Expression_level = st.slider(f"Expression value for {selected_gene}", 0.0, 10.0, 6.0)

# Step 3: Select Tissue Type from the dropdown (Single selection)
Tissue_types = ['Leiomyosarcomas', 'Non-myogenic_sarcomas']  # Example tissue types; replace with actual types
selected_tissue = st.selectbox(
    "Select Tissue Type", 
    Tissue_types, 
    index=0  # Default to the first tissue type
)

# Function to get user input for the Gene Combination
def get_user_input():      
    # Store the user input in a dictionary
    user_data = {
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

# One-hot encode the selected tissue type
def one_hot_encode_tissue(selected_tissue, Tissue_types):
    tissue_vector = [0.0] * len(Tissue_types)  # Initialize with zeros
    tissue_index = Tissue_types.index(selected_tissue)  # Find index of selected tissue
    tissue_vector[tissue_index] = 1.0  # Set the selected tissue's position to 1.0
    return tissue_vector

# Get one-hot encoded gene combination and tissue type
encoded_genes = one_hot_encode_gene(selected_gene, Gene_names)
encoded_tissue = one_hot_encode_tissue(selected_tissue, Tissue_types)

# Prepare the numerical variables (Expression Level)
numerical_variables = [user_input['Expression Level']]


# Debugging print statements to verify dimensions
# st.write("Numerical variables:", numerical_variables)
# st.write("One-hot encoded gene vector:", encoded_genes)
# st.write("One-hot encoded tissue vector:", encoded_tissue)
# st.write("Total features expected:", len(numerical_variables) + len(encoded_genes) + len(encoded_tissue))


# Combine numerical variables with one-hot encoded genes and tissue types BEFORE scaling
full_input = numerical_variables + encoded_genes + encoded_tissue

# Convert to a 2D array to pass to the scaler (as scaler expects 2D input)
full_input = [full_input]

# Apply scaling to the full input (numerical + categorical one-hot encoded)
full_input_scaled = scaler.transform(full_input)

# Debugging output to check the final predictive variables
st.write("Predictive variables after scaling and one-hot encoding:", full_input_scaled)

# Prediction
prediction = best_model.predict(full_input_scaled)

# Map the prediction to a corresponding message
if prediction[0] == 2:
    risk_message = "Predicted High-Risk"
elif prediction[0] == 1:
    risk_message = "Predicted Low-Risk"
else:
    risk_message = "Unknown Risk Level"  # Just in case there's another prediction value

# Display the prediction result
st.success(f"Prediction: {prediction[0]}  ({risk_message})")