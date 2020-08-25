# mol_properties
## Datasets
All the collected and preprocessed data is located in /data
## Baseline models
The baselines are located in scripts/baselines and notebooks/baselines
The covered baselines are: 
  - morgan bit fingerprints
    * /notebooks/baselines/bit_morgan_fingerprint
  - morgan count fingerprints
    * /notebooks/baselines/bit_count_fingerprint
  - neural fingerprints 
    * /baselines/neuralfingerprints
    * /notebooks/baselines/neuralfingerprint
  - MolLogP predictor
    * /notebooks/baselines/MolLogP
## SOTA models
The covered SOTA models are:
  - Path augmented graph transformer neural network
    * /scripts/baselines/pa_graph_transformer
    * /notebooks/baselines/pa_graph_transformer
  - Optimal transport graph neural neetwork 
    * /scripts/baselines/otgnn
    * /notebooks/baselines/otgnn
  - Junction tree
    * /scripts/baselines/jtree
    * /notebooks/baselines/jtree
  - Directed message passing neural network
    * /scripts/baselines/dmpnn
    * /notebooks/baselines/dmpnn
## Improvements and DMPNN with substructures model
We decided to improve DMPNN model. The source code and algorithm description are located in /scripts/baseline_improvements/chemprop.