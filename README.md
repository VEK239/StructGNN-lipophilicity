# mol_properties
This repository contains the models for predicting molecular LogP property.

Branches:
  - `eda` - data preprocessing
  - `baselines` - baseline models: 
    * morgan bit fingerprints
    * morgan count fingerprints
    * neural fingerprints
    * MolLogP predictor
  - `SOTA` - state of the art models:
    * Path augmented graph transformer neural network
    * Optimal transport graph neural neetwork
    * Junction tree
    * Directed message passing neural network
  - `baseline_improvements` - the branch with the improvement of D-MPNN model by adding the second D-MPNN encoder with substructures molecular graph
  - `MPAPC-65` - the branch with the improvement of D-MPNN by adding extra atom features as a separate encoder with substructures molecular graph