# Directories description

- raw - the directory with raw data collected (for logP and logD)
- 1_filtering - the data with removed strange molecules
- 2_standardize - standardized SMILES data
- 3_final_data - fully preprocessed data and subdirectiry with splitted datasets
- 4_best_predictions - the directory with predictions for the test datasets for the best models 
# logP preprocessed datasets

|Name of dataset|Number of entries|Description|Preprocessing|
|---|---|---|---|
|logp_mean.csv|13759|LogP measurements for molecules in all raw data|1. Standartization of molecules in raw data, merging all datasets<br/> 2. Removing molecules with LogP  out of range [-5, 10] <br/> 3. Deleting 1 row with temperature = 257<br/> 4. Removing strange molecules (488 entries) <br/> 5. Deleting 18 SMILES with multiple LogP measurements with variance>1 <br/> 6. Averaging multiple LogP measurements of the same SMILES|
|logP_wo_parameters.csv|12626|LogP measurements for molecules in all raw data where pH and Temperature were NaN values|1. Standartization of molecules in raw data, merging all datasets<br/> 2. Removing strange molecules (485 entries) <br/> 3. Removing molecules with LogP  out of range [-5, 10] <br/> 4. Deleting 2 SMILES with multiple LogP measurements with std>1 <br/> 5. Averaging multiple LogP measurements of the same SMILES|
|logp_pH_range_mean.csv|1504|LogP measurements for molecules with known pH parameter categorized as acid (0), neutral (1) or alkali (2))|1. Standartization of molecules in raw logP+pH data  <br />2. No strange molecules <br />3. Removing molecules with LogP out of range [-5, 10]<br />4. Categorazation of pH was made according to the rule: <br />   - pH < 6 => acid (0) <br />   - 6 <= pH < 8 => neutral (1) <br />   - ph >= 8 => alkali (2) <br />5. After dropping duplicates there were no records with unique smiles+pH_range and var(logP)>1<br />6. All the records were averaged. All in all, there 1504 unique records left.<br />7. Environments:  <br />   - 0 - 594 <br />   - 1 - 677 <br />   - 2 - 233|
|logp_wo_averaging.csv|13777|LogP measurements for molecules in all raw data without averaging when several logP measurements, but  choosing most common|1. Standartization of molecules in raw data, merging all datasets<br/> 2. Removing molecules with LogP  out of range [-5, 10] <br/> 3. Deleting 1 row with temperature = 257<br/> 4. Removing strange molecules (488 entries) <br/> 5. Choosing most common logP value from several measurements (if there is no most common -> random)|



