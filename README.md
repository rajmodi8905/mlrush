# mlrush

This repo contains code and resources for predicting molecular properties using SOAP and Coulomb (removed) matrix descriptors, trained with MLRegressors.  

Link to the Kaggle Competetion: [Molecular Property Prediction Challenge](https://www.kaggle.com/competitions/molecular-property-prediction-challenge/data)


We had to predict the dipole moment of small organic molecules using their 3D atomic structures provided in XYZ format. We were tasked with building regression model that take in molecular descriptors derived from these structures and output a continuous value representing the dipole moment.

This involved:

- Parsing and processing raw XYZ molecular structure files.
- Extracting 2D or 3D descriptors such as:
  - **SOAP descriptors** using the `DScribe` and `ASE` libraries.
  - **Coulomb matrix descriptors** to represent interatomic interactions.
- Representing molecules numerically in a machine-learnable format.
- Training and evaluating **ML regression models** such as XGBoost, SVR, and Ridge Regression.
- Submitting predicted dipole moments in a specific CSV format for leaderboard evaluation using **Mean Squared Error (MSE)**.


This challenge tested not just modeling skills, but also our ability to work with messy scientific data, handle missing values, and engineer meaningful features from complex molecular structures.


Efforts by:
- Raj Amit Modi (24110282), Computer Science & Engineering
- Suhani (24110358), Computer Science & Engineering
