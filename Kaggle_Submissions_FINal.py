# Standard libraries for data handling
import numpy as np
import pandas as pd

# Install chemistry and ML-specific libraries
!pip install rdkit-pypi tqdm ase dscribe

# Core imports for working with molecules, descriptors, and ML
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from ase.io import read
from dscribe.descriptors import SOAP

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# --------------------------------------------------
# Define paths to training data provided by Kaggle
# --------------------------------------------------
BASE_PATH = "/kaggle/input/molecular-property-prediction-challenge"
TRAIN_XYZ_DIR = os.path.join(BASE_PATH, "structures_train")
TRAIN_CSV = os.path.join(BASE_PATH, "dipole_moments_train.csv")

train_df = pd.read_csv(TRAIN_CSV)

# --------------------------------------------------
# Initialize SOAP descriptor
# --------------------------------------------------
soap = SOAP(
    species=["H", "C", "N", "O", "F"],
    periodic=False,
    r_cut=4.5,
    n_max=10,
    l_max=8,
    sigma=0.2,
    sparse=False
)

# Function to compute the SOAP vector for a molecule
def get_soap_vector_from_xyz(xyz_path):
    try:
        atoms = read(xyz_path)
        soap_vec = soap.create(atoms, n_jobs=1)
        return np.mean(soap_vec, axis=0)  # Average to get a fixed-length vector
    except Exception as e:
        print(f"Failed on {xyz_path}: {e}")
        return None

# --------------------------------------------------
# Extract SOAP features for each molecule
# --------------------------------------------------
soap_descriptors = []
dipole_moments = []

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing SOAP descriptors"):
    mol_id = row['molecule_name']
    xyz_path = os.path.join(TRAIN_XYZ_DIR, f"{mol_id}.xyz")
    
    if not os.path.exists(xyz_path):
        continue

    soap_vec = get_soap_vector_from_xyz(xyz_path)
    if soap_vec is None:
        continue
    
    soap_descriptors.append(soap_vec)
    dipole_moments.append(row['dipole_moment'])

# Convert feature list to NumPy arrays
X_soap = np.array(soap_descriptors)
y = np.array(dipole_moments)

print("SOAP shape:", X_soap.shape)
print("Target shape:", y.shape)

if X_soap.size == 0:
    raise ValueError("No descriptors generated. Check the input files or SOAP setup.")

# --------------------------------------------------
# Normalize the features before feeding into the model
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_soap)

# --------------------------------------------------
# Split the dataset for training and evaluation
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train an MLP Regressor on the SOAP features
# --------------------------------------------------
mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

mlp_model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate model performance on the hold-out test set
# --------------------------------------------------
y_pred = mlp_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse:.4f}")

# --------------------------------------------------
# Load test molecules and generate predictions
# --------------------------------------------------
TEST_XYZ_DIR = os.path.join(BASE_PATH, "structures_test")
TEST_CSV = os.path.join(BASE_PATH, "dipole_moments_test.csv")
test_df = pd.read_csv(TEST_CSV)

submission_ids = []
submission_preds = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating test predictions"):
    mol_id = row['ID']
    xyz_path = os.path.join(TEST_XYZ_DIR, f"{mol_id}.xyz")
    
    if not os.path.exists(xyz_path):
        print(f"Missing {xyz_path}")
        continue

    soap_vec = get_soap_vector_from_xyz(xyz_path)
    if soap_vec is None:
        continue
    
    soap_vec_scaled = scaler.transform([soap_vec])
    pred = mlp_model.predict(soap_vec_scaled)[0]
    
    submission_ids.append(mol_id)
    submission_preds.append(pred)

# --------------------------------------------------
# Save the predictions to a CSV for submission
# --------------------------------------------------
submission_df = pd.DataFrame({
    "ID": submission_ids,
    "dipole_moment": submission_preds
})

submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file saved as submission.csv")
