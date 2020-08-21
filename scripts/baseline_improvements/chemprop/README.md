## D-MPNN model improvement.

Added second encoder into the [D-MPNN model](https://github.com/chemprop/chemprop).

### The second encoder's algorithm

This encoder first finds all the interesting substructures in a molecule (rings, acids, amins, esters and sulfonamins) and makes a list of such substructures. After that each of the substructure and all the atoms which were not included in any of substructures are encoded into special "substructure atoms". Each "substructure atom" is evcoded into a vector with 165 features. The features are:
* 55 features for structure, each feature - count of atoms in a substructure with atomic charge `i`,
* 40 features for one-hot encoding of a substructure **external** valence,
* 60 features for one-hot encoding of a substructure count of hydrogens,
* substructure formal_charge (often zero, sometimes 1-2),
* substructure is_aromatic (0/1),
* substructure_mass \* 0.01 (for normalizing),
* sum of internal substructure internal edges \* 0.1 (for normalizing),
* one-hot for structure type (types are - ['ATOM', 'RING', 'ACID', 'AMIN', 'ESTER', 'SULFONAMID'])

The bonds which connect two **different** "substructure atoms" are present in a new molecule. All the others are skipped. 

### Running
If you are running from mol_properties directory:
--config_path ./scripts/baseline_improvements/chemprop/datasets/config.json --data_path 
./scripts/baseline_improvements/chemprop/datasets/logp_wo_averaging_train.csv --dataset_type regression

The configuration file is located in ./datasets/config.json
The command line arguments different from original D-MPNN:
- substructures/no_substructures depth - count of message passing steps for each encoder
- substructures/no_substructures hidden size - size of hidden layer size in each encoder
- substructures/no_substructures atom messages - message passing based on atom messages or not in each encoder
- substructures/no_substructures undirected - whether to use directed or undirected MPNN in each encoder
- substructures_use_substructures - whether to use additional substructures(acids, amins, sulfonamins, esters)
- substructures_merge - whether to merge neighboring cycles or not

There is also a new type of cross-validation ("one_out_crossval") which takes one of the four parts of the input dataset and uses it as validation.

### Feature generators

Some new feature generators were added.
- RDKit without MolLogP feature
- Only RDKit calculated MolLogP feature
- RDKit deatures without any fragment counting features   