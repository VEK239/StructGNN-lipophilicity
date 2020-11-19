import pandas as pd

train = pd.read_csv('./data/3_final_data/split_data/logp_wo_logp_json_wo_averaging_train.csv')
val = pd.read_csv('./data/3_final_data/split_data/logp_wo_logp_json_wo_averaging_validation.csv')
train_val = pd.concat([train, val], ignore_index=True)
train_val = train_val.drop(columns=['Unnamed: 0'])
train_val.to_csv('./data/3_final_data/split_data/logp_wo_logp_json_wo_averaging_train_val_dataset.csv', index=False)