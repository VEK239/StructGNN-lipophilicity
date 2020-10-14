
# Datasets

| Dataset name | Number of Samples | Description | Sources |
| --- | --- | --- | --- |
| logp_wo_logp_json_wo_averaging | 13688 | All logP datasets except logp.json | Diverse1KDataset.csv, NCIDataset.csv, ochem_full.csv, physprop.csv |
| logd_Lip_wo_averaging | 4166 | Merged datasets w/o strange (very soluble) molecules and standardized SMILES. Between duplicated logD for one SMILES the most common value was chosen | Lipophilicity |
| esol | 1058 | Standardize molecules, remove strange ones and duplicates, choose most common logS value | ESOL |
| freesolv | 565 | Standardize molecules, remove strange ones and duplicates, choose most common Energy value | FreeSolv |
| logp_wo_logp_json_logd_Lip_wo_averaging | 17603 | Merged LogP wo logp.json and Lipophilicity datasets, 251 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |
| logp_wo_logp_json_251_Lip_wo_averaging | 13688 | Add to logP dataset 251 logD values for molecules common in two datasets | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |
| logd_251_logp_wo_logp_json_wo_averaging | 4166 | Add to logD dataset 251 logP values for molecules common in two datasets | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |
| logp_wo_logp_json_esol_wo_averaging | 13960 | Merged LogP wo logp.json and esol datasets, 786 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>esol |
| logp_wo_logp_json_FreeSolv_wo_averaging | 13851 | Merged LogP wo logp.json and freesolv datasets, 402 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>freesolv |


# Models

1. [OTGNN](https://github.com/jbr-ai-labs/mol_properties/tree/SOTA/scripts/SOTA/otgnn)
2. [PAGTNN](https://github.com/jbr-ai-labs/mol_properties/tree/SOTA/scripts/SOTA/pa-graph-transformer-neural-network)
3. [DMPNN + StructGNN](https://github.com/jbr-ai-labs/mol_properties/tree/SOTA/scripts/SOTA/dmpnn)

