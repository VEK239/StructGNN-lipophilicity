# mol_properties

# Datasets

|Name of dataset|Number of entries|Description|Preprocessing|
|---|---|---|---|
|logp_mean.csv|13759|LogP measurements for molecules in all raw data|1. Standartization of molecules in raw data, merging all datasets<br/> 2. Removing molecules with LogP  out of range [-5, 10] <br/> 3. Deleting 1 row with temperature = 257<br/> 4. Removing strange molecules (488 entries) <br/> 5. Deleting 18 SMILES with multiple LogP measurements with variance>1 <br/> 6. Averaging multiple LogP measurements of the same SMILES|
|logP_wo_parameters.csv|12631|LogP measurements for molecules in all raw data where pH and Temperature were NaN values|1. Standartization of molecules in raw data, merging all datasets<br/> 2. Removing strange molecules (485 entries) <br/> 3. Removing molecules with LogP  out of range [-5, 10] <br/> 4. Deleting 2 SMILES with multiple LogP measurements with std>1 <br/> 5. Averaging multiple LogP measurements of the same SMILES|
|logp_pH_range_mean.csv|1504|LogP measurements for molecules with known pH parameter categorized as acid (0), neutral (1) or alkali (2))||
